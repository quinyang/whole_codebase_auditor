"""Stage 4 -- Context packer. The actual contribution of this project.

Linear scaling is a statement about *asymptotics*, not about free VRAM: a Mamba
prefill over 200k tokens still costs real memory and minutes of wall clock on a
T4, and Falcon3-Mamba-7B-Instruct was trained at 32k. So the packer treats the
context as a fixed budget to be allocated, not a bucket to be filled.

Three jobs:

  1. ORDER    Group files by symbol-graph connected component and order within a
              component by dependency depth, so code that interacts is adjacent
              in the stream. An SSM compresses history into a fixed-size state --
              distance between two related facts directly costs recall.

  2. DEMOTE   Spend full bodies on files that are (a) high risk-score, or (b) on
              a dependency path from a high-risk file. Everything else is
              reduced to a signature skeleton: imports + def lines. Cheap
              context that still lets the model resolve a cross-file name.

  3. MANIFEST Record, for every emitted segment, the mapping from stream
              character offset back to (path, line). Without this, model output
              cannot be grounded, and a finding without a location is unusable.

Token accounting uses the real tokenizer when one is supplied, and a
chars-per-token estimate otherwise so the stage stays CPU-runnable.
"""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from wca.graph import SymbolGraph
from wca.parse import ParsedFile

# Source code tokenizes denser than prose. Measured against Falcon3-Mamba's
# tokenizer this sits around 3.2-3.8 for Python/Go/JS; 3.5 is a safe default and
# `pack(tokenizer=...)` replaces it with exact counts when a tokenizer exists.
CHARS_PER_TOKEN = 3.5

FULL = "full"
SIGNATURE = "signature"
OMITTED = "omitted"

_QUOTES = str.maketrans({'"': "'", "“": "'", "”": "'", "‘": "'", "’": "'"})


def _normalise_code(line: str) -> str:
    """Collapse the differences a model introduces when re-quoting a line.

    Quote style, smart quotes, markdown backticks and whitespace are all things
    a model changes freely while still faithfully quoting real code.
    """
    return re.sub(r"\s+", " ", line.translate(_QUOTES).replace("`", "")).strip()


@dataclass
class Segment:
    """One emitted region of the packed stream."""

    path: str
    mode: str  # full | signature
    char_start: int
    char_end: int
    line_start: int  # 0-indexed line in the ORIGINAL file at char_start
    # For signature mode the emitted lines are non-contiguous, so we carry an
    # explicit map: offset-within-segment -> original line number.
    line_map: list[tuple[int, int]] = field(default_factory=list)
    est_tokens: int = 0


@dataclass
class PackedContext:
    text: str
    segments: list[Segment]
    budget_tokens: int
    used_tokens: int
    n_full: int
    n_signature: int
    n_omitted: int
    omitted_paths: list[str] = field(default_factory=list)

    # ---- offset resolution -------------------------------------------------

    def __post_init__(self) -> None:
        self._starts = [s.char_start for s in self.segments]
        self._norm_cache: list[tuple[int, str]] | None = None

    def resolve(self, offset: int) -> tuple[str, int] | None:
        """Map a character offset in `text` back to (path, 1-indexed line)."""
        if not self.segments:
            return None
        i = bisect.bisect_right(self._starts, offset) - 1
        if i < 0:
            return None
        seg = self.segments[i]
        if offset >= seg.char_end:
            return None
        local = offset - seg.char_start
        if seg.line_map:
            j = bisect.bisect_right([m[0] for m in seg.line_map], local) - 1
            j = max(j, 0)
            return seg.path, seg.line_map[j][1] + 1
        # full mode: count newlines from the segment start
        newlines = self.text.count("\n", seg.char_start, seg.char_start + local)
        return seg.path, seg.line_start + newlines + 1

    def resolve_snippet(self, snippet: str, *, min_ratio: float = 0.85) -> tuple[str, int] | None:
        """Locate a snippet the model quoted back, then resolve it to (path, line).

        Exact substring matching is too brittle to be the only strategy. Models
        reliably re-emit a line with the quote style swapped, wrapped in
        markdown backticks, or with whitespace reflowed -- all of which are
        faithful quotations of real code that a `str.find` rejects. Grounding is
        what separates a real finding from a hallucination, so a false negative
        here is expensive: it silently reclassifies correct work as invented.

        Four passes, most trustworthy first:
          1. exact substring
          2. exact match of any single line of the snippet
          3. normalised match (quotes unified, whitespace collapsed, backticks
             stripped) against normalised stream lines
          4. fuzzy match above `min_ratio`, longest normalised line first

        Pass 4 is deliberately gated: a genuine hallucination shares almost no
        structure with any real line and scores far below the threshold.
        """
        s = snippet.strip().strip("`").strip()
        if not s:
            return None

        idx = self.text.find(s)
        if idx != -1:
            return self.resolve(idx)

        lines = [ln.strip().strip("`").strip() for ln in s.splitlines()]
        lines = [ln for ln in lines if len(ln) >= 4]
        for ln in lines:
            idx = self.text.find(ln)
            if idx != -1:
                return self.resolve(idx)

        targets = [_normalise_code(ln) for ln in lines]
        targets = [t for t in targets if len(t) >= 6]
        if not targets:
            return None

        best: tuple[float, int] | None = None
        for offset, norm_line in self._normalised_lines():
            if len(norm_line) < 6:
                continue
            for t in targets:
                if t == norm_line or (len(t) >= 12 and t in norm_line):
                    return self.resolve(offset)
                ratio = SequenceMatcher(None, t, norm_line).ratio()
                if ratio >= min_ratio and (best is None or ratio > best[0]):
                    best = (ratio, offset)
        return self.resolve(best[1]) if best else None

    def closest_line(self, snippet: str) -> tuple[str, int, float, str] | None:
        """Best-matching stream line for a snippet: (path, line, ratio, text).

        Diagnostic for ungrounded findings. "0% grounded" is not actionable on
        its own -- a near-miss at ratio 0.8 means the model paraphrased a real
        line and the threshold is too tight, while a best match at 0.2 means it
        invented the evidence outright. Those need opposite responses.
        """
        s = _normalise_code(snippet.strip().strip("`"))
        if len(s) < 4:
            return None
        best: tuple[float, int, str] | None = None
        for offset, norm_line in self._normalised_lines():
            if len(norm_line) < 4:
                continue
            ratio = SequenceMatcher(None, s, norm_line).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, offset, norm_line)
        if best is None:
            return None
        resolved = self.resolve(best[1])
        if resolved is None:
            return None
        return resolved[0], resolved[1], round(best[0], 3), best[2][:100]

    def _normalised_lines(self) -> list[tuple[int, str]]:
        """[(offset_of_line_start, normalised_text)] over the packed stream."""
        if self._norm_cache is None:
            cache: list[tuple[int, str]] = []
            offset = 0
            for line in self.text.splitlines(keepends=True):
                cache.append((offset, _normalise_code(line)))
                offset += len(line)
            self._norm_cache = cache
        return self._norm_cache

    def manifest(self) -> dict:
        return {
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "n_chars": len(self.text),
            "counts": {
                "full": self.n_full,
                "signature": self.n_signature,
                "omitted": self.n_omitted,
            },
            "omitted_paths": self.omitted_paths,
            "segments": [asdict(s) for s in self.segments],
        }

    def write_manifest(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest(), fh, indent=2)

    def stats_line(self) -> str:
        pct = 100.0 * self.used_tokens / self.budget_tokens if self.budget_tokens else 0.0
        return (
            f"packed {self.n_full} full / {self.n_signature} sig / "
            f"{self.n_omitted} omitted -> ~{self.used_tokens:,} tok "
            f"({pct:.1f}% of {self.budget_tokens:,})"
        )


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _dependency_order(paths: list[str], graph: SymbolGraph) -> list[str]:
    """Kahn's algorithm on import edges; cycles fall back to degree order."""
    subset = set(paths)
    indeg = {p: 0 for p in paths}
    adj: dict[str, list[str]] = {p: [] for p in paths}
    for e in graph.edges:
        if e.kind != "import" or e.src not in subset or e.dst not in subset:
            continue
        # dst is the dependency; emit it before src
        adj[e.dst].append(e.src)
        indeg[e.src] += 1

    degree = graph.degree()
    ready = sorted([p for p in paths if indeg[p] == 0], key=lambda p: -degree.get(p, 0))
    out: list[str] = []
    while ready:
        p = ready.pop(0)
        out.append(p)
        for nxt in adj[p]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                bisect.insort(ready, nxt, key=lambda x: -degree.get(x, 0))
    # anything left is in a cycle
    remaining = [p for p in paths if p not in set(out)]
    remaining.sort(key=lambda p: -degree.get(p, 0))
    return out + remaining


def order_files(graph: SymbolGraph) -> list[str]:
    """Components ordered by peak risk, files within a component by dependency."""
    ordered: list[str] = []
    comps = graph.components()
    comps.sort(
        key=lambda c: max((graph.symbols[p].risk_score for p in c), default=0.0),
        reverse=True,
    )
    for comp in comps:
        ordered.extend(_dependency_order(comp, graph))
    return ordered


# --------------------------------------------------------------------------- #
# Priority
# --------------------------------------------------------------------------- #


def compute_priority(graph: SymbolGraph, *, hops: int = 1) -> dict[str, float]:
    """Risk score, propagated `hops` edges outward at decaying weight.

    A file with a hardcoded credential is interesting; so is every file that
    imports it or calls into it -- that pair *is* the cross-file vulnerability.
    """
    base = {p: fs.risk_score for p, fs in graph.symbols.items()}
    score = dict(base)
    frontier = {p for p, v in base.items() if v > 0}
    weight = 0.5
    for _ in range(hops):
        nxt: set[str] = set()
        for p in frontier:
            for n in graph.neighbors(p):
                if n in score:
                    score[n] += weight * base.get(p, 0.0)
                    nxt.add(n)
        frontier = nxt
        weight *= 0.5
    degree = graph.degree()
    for p in score:
        score[p] += 0.25 * degree.get(p, 0)  # hubs are worth seeing in full
    return score


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_signature(pf: ParsedFile, graph: SymbolGraph) -> tuple[str, list[tuple[int, int]]]:
    """Skeleton view: import lines + one line per definition, elisions marked.

    Returns (text, line_map) where line_map is [(offset_in_text, orig_line)].
    """
    fs = graph.symbols.get(pf.path)
    lines = pf.text.splitlines()
    keep: set[int] = set()

    for i, line in enumerate(lines[:200]):
        s = line.lstrip()
        if s.startswith(("import ", "from ", "#include", "use ", "package ", "require(")):
            keep.add(i)
    if fs:
        for d in fs.defs:
            keep.add(d.start_line)
        for i, line in enumerate(lines):
            if any(sec[:40] and sec[:40] in line for sec in fs.secretish[:20]):
                keep.add(i)

    out: list[str] = []
    line_map: list[tuple[int, int]] = []
    offset = 0
    prev = -1
    for i in sorted(keep):
        if prev != -1 and i > prev + 1:
            gap = f"    ... {i - prev - 1} lines elided ...\n"
            out.append(gap)
            offset += len(gap)
        rendered = lines[i].rstrip() + "\n"
        line_map.append((offset, i))
        out.append(rendered)
        offset += len(rendered)
        prev = i
    return "".join(out), line_map


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


def _default_counter(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _make_counter(tokenizer) -> Callable[[str], int]:
    if tokenizer is None:
        return _default_counter

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


HEADER = """<repository name="{name}" files_full="{n_full}" files_signature="{n_sig}">
Files marked mode="signature" show imports and declarations only; their bodies
are elided. Files marked mode="full" are complete.
"""
FOOTER = "</repository>\n"


def _render(pf: ParsedFile, mode: str, graph: SymbolGraph) -> tuple[str, list[tuple[int, int]], int]:
    """(body, line_map, line_start) for a file in a given mode."""
    if mode == FULL:
        return pf.text, [], 0
    body, line_map = render_signature(pf, graph)
    return body, line_map, (line_map[0][1] if line_map else 0)


def _chunk_tags(pf: ParsedFile, mode: str) -> tuple[str, str]:
    return f'\n<file path="{pf.path}" lang="{pf.lang}" mode="{mode}">\n', "\n</file>\n"


def pack(
    parsed: Iterable[ParsedFile],
    graph: SymbolGraph,
    *,
    budget_tokens: int = 24_000,
    tokenizer=None,
    repo_name: str = "repo",
    reserve_tokens: int = 2_000,
    full_body_share: float = 0.70,
) -> PackedContext:
    """Fit the repository into `budget_tokens`, emitting an offset manifest.

    Selection and ordering are deliberately separate concerns:

      SELECT  by priority (which files get a full body, which get a signature,
              which are dropped) -- decided globally, before anything is emitted.
      ORDER   by symbol graph (where each surviving file sits in the stream).

    Conflating them -- emitting greedily in stream order -- means a low-priority
    file early in the dependency order eats budget that a high-risk file later in
    the order needed, and the signature tier never gets reached at all.

    `full_body_share` caps what full bodies may consume, so a signature tier
    always exists: a repo where 40% of files are entirely absent from the stream
    cannot support a cross-file claim, even if the 60% present are complete.
    `reserve_tokens` is held back for the instruction preamble and the model's
    own output, so budget_tokens can be set to the model's real context length.
    """
    by_path = {pf.path: pf for pf in parsed}
    order = [p for p in order_files(graph) if p in by_path]
    order += [p for p in by_path if p not in set(order)]

    priority = compute_priority(graph)
    count = _make_counter(tokenizer)

    header = HEADER.format(name=repo_name, n_full=0, n_sig=0)
    usable = max(budget_tokens - reserve_tokens, 1024) - count(header) - count(FOOTER)

    # ---- selection ---------------------------------------------------------
    ranked = sorted(order, key=lambda p: (-priority.get(p, 0.0), p))
    mode_of: dict[str, str] = {}
    spent = 0
    full_cap = int(usable * full_body_share)

    def cost(path: str, mode: str) -> int:
        pf = by_path[path]
        body, _, _ = _render(pf, mode, graph)
        if mode == SIGNATURE and not body.strip():
            return -1  # nothing worth emitting
        o, c = _chunk_tags(pf, mode)
        return count(o + body + c)

    for path in ranked:  # highest priority first
        c = cost(path, FULL)
        if spent + c <= full_cap:
            mode_of[path] = FULL
            spent += c

    for path in ranked:  # everything not already full -> signature tier
        if path in mode_of:
            continue
        c = cost(path, SIGNATURE)
        if c < 0:
            continue
        if spent + c <= usable:
            mode_of[path] = SIGNATURE
            spent += c

    # Breadth before depth: only once every file is already present in the stream
    # is it worth spending leftover budget upgrading signatures to full bodies.
    # Promoting while files are still omitted trades coverage the cross-file claim
    # depends on for detail on files that are already visible.
    if len(mode_of) == len(order):
        for path in ranked:
            if mode_of.get(path) != SIGNATURE:
                continue
            delta = cost(path, FULL) - cost(path, SIGNATURE)
            if spent + delta <= usable:
                mode_of[path] = FULL
                spent += delta

    # ---- emission (stream order) -------------------------------------------
    parts: list[str] = [header]
    offset = len(header)
    used = count(header)
    segments: list[Segment] = []
    omitted: list[str] = []
    n_full = n_sig = 0

    for path in order:
        mode = mode_of.get(path)
        if mode is None:
            omitted.append(path)
            continue
        pf = by_path[path]
        body, line_map, line_start = _render(pf, mode, graph)
        open_tag, close_tag = _chunk_tags(pf, mode)
        chunk = open_tag + body + close_tag

        char_start = offset + len(open_tag)
        parts.append(chunk)
        offset += len(chunk)
        used += count(chunk)
        segments.append(
            Segment(
                path=path,
                mode=mode,
                char_start=char_start,
                char_end=char_start + len(body),
                line_start=line_start,
                line_map=line_map,
                est_tokens=count(chunk),
            )
        )
        if mode == FULL:
            n_full += 1
        else:
            n_sig += 1

    parts.append(FOOTER)
    used += count(FOOTER)

    return PackedContext(
        text="".join(parts),
        segments=segments,
        budget_tokens=budget_tokens,
        used_tokens=used,
        n_full=n_full,
        n_signature=n_sig,
        n_omitted=len(omitted),
        omitted_paths=omitted,
    )
