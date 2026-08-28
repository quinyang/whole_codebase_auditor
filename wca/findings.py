"""Stage 6 -- Findings.

Turn raw model text into a validated JSON schema, and ground every finding in a
real (file, line) using the packer's manifest.

Grounding is the honesty mechanism of this project. A finding whose `evidence`
line cannot be located in the packed stream is marked `grounded=False`. Those
are reported separately and excluded from the precision number -- an ungrounded
finding is a hallucination by construction, and the eval must say so.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field

from wca.pack import PackedContext

SEVERITIES = ("critical", "high", "medium", "low")
CATEGORIES = (
    "hardcoded_secret",
    "injection",
    "auth_bypass",
    "unsafe_deserialization",
    "path_traversal",
    "data_exposure",
    "other",
)


@dataclass
class Finding:
    title: str
    severity: str
    category: str
    files: list[str] = field(default_factory=list)
    evidence: str = ""
    why_cross_file: str = ""
    confidence: float = 0.0
    # resolved
    location: str | None = None  # "path:line"
    grounded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_cross_file(self) -> bool:
        return len(set(self.files)) >= 2

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        mark = "OK " if self.grounded else "?? "
        loc = self.location or ", ".join(self.files) or "unlocated"
        return (
            f"{mark}[{self.severity.upper():8}] {self.title}\n"
            f"      {loc}  ({self.category}, conf {self.confidence:.2f})\n"
            f"      {self.why_cross_file.strip()[:180]}"
        )


def _loads_lenient(blob: str) -> object | None:
    """Parse one JSON-ish value, tolerating the ways models deviate from JSON.

    Observed from Falcon3-Mamba: correct findings with exact evidence lines,
    emitted as `"evidence": 'logger.info(...)'` -- Python string syntax, not
    JSON. `json.loads` rejects the whole array and every correct finding in it
    is silently discarded. Being strict here means scoring the model's best work
    as a total miss, which is the expensive direction to be wrong in.

    `ast.literal_eval` accepts both quote styles natively, so it is the fallback.
    JSON's bare `true`/`false`/`null` are not Python literals, so those are
    mapped first -- outside string bodies only.
    """
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        pass

    # trailing commas: {"a": 1,} / [1,]
    repaired = re.sub(r",(\s*[}\]])", r"\1", blob)
    for candidate in (repaired, _map_json_literals(repaired)):
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, RecursionError, MemoryError):
            continue
    return None


def _map_json_literals(blob: str) -> str:
    """true/false/null -> True/False/None, skipping anything inside a string."""
    out: list[str] = []
    i, n = 0, len(blob)
    quote: str | None = None
    while i < n:
        ch = blob[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(blob[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        for word, repl in (("true", "True"), ("false", "False"), ("null", "None")):
            if blob.startswith(word, i) and not (
                i and (blob[i - 1].isalnum() or blob[i - 1] == "_")
            ):
                nxt = i + len(word)
                if nxt >= n or not (blob[nxt].isalnum() or blob[nxt] == "_"):
                    out.append(repl)
                    i = nxt
                    break
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _balanced_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    """Balanced `open_ch..close_ch` spans, respecting both quote styles.

    Tracking `'` as well as `"` matters: a Python-quoted value containing a
    double quote -- `'logger.info("x", y)'` -- desynchronises a scanner that
    only knows about `"`, which then mis-detects where the object ends.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth:
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start : i + 1])
                start = -1
        i += 1
    return spans


def extract_json_array(text: str) -> list[dict]:
    """Pull findings out of model output, tolerating near-JSON and truncation.

    Four passes, most faithful first:
      1. the whole fenced block, or the whole text, as one array
      2. the first balanced [...] span
      3. individual balanced {...} objects (survives a truncated array --
         `max_new_tokens` cutting off the final object should not discard the
         complete ones before it)
      4. nothing
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    # an unterminated fence is the common truncation case
    unterminated = re.search(r"```(?:json)?\s*(.+)$", text, re.DOTALL)
    if unterminated:
        candidates.append(unterminated.group(1))
    candidates.append(text)

    for cand in candidates:
        for span in _balanced_spans(cand, "[", "]")[:1]:
            parsed = _loads_lenient(span)
            if isinstance(parsed, list):
                dicts = [p for p in parsed if isinstance(p, dict)]
                if dicts:
                    return dicts

    # Object-by-object salvage. Keeps whatever completed before a cutoff.
    out: list[dict] = []
    seen: set[str] = set()
    for cand in candidates:
        spans = _balanced_spans(cand, "{", "}")
        for span in spans:
            obj = _loads_lenient(span)
            if isinstance(obj, dict) and "title" in obj:
                key = str(obj.get("title"))
                if key not in seen:
                    seen.add(key)
                    out.append(obj)
        # A generation cut off by max_new_tokens leaves a final object with
        # several complete fields and no closing brace. Those fields are real
        # model output; discarding them loses recall for a formatting reason.
        tail_start = cand.rfind(spans[-1]) + len(spans[-1]) if spans else 0
        obj = _repair_truncated_object(cand[tail_start:])
        if obj is not None and str(obj.get("title")) not in seen:
            seen.add(str(obj.get("title")))
            out.append(obj)
        if out:
            break
    return out


def _repair_truncated_object(fragment: str) -> dict | None:
    """Recover a dict from an object the model never finished.

    Walks back one line at a time, dropping the incomplete tail, until the
    remainder closes into something parseable with a `title`.
    """
    start = fragment.find("{")
    if start == -1:
        return None
    lines = fragment[start:].splitlines()
    for stop in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:stop]).rstrip().rstrip(",")
        if not chunk.endswith("}"):
            chunk += "}"
        obj = _loads_lenient(chunk)
        if isinstance(obj, dict) and "title" in obj:
            return obj
    return None


# Shapes seen when a model authors an illustrative example instead of quoting.
# Observed on pallets/click: evidence of "# In my_script.sh:\nexport
# SECRET_TOKEN='supersecret...'" -- a file that does not exist, a secret that was
# invented. Distinguishing "fabricated" from "quoted but unmatched" matters: the
# first is a prompt problem, the second is a matcher problem.
_FABRICATION_HINTS = (
    re.compile(r"^\s*#\s*(in|example|file)\b", re.IGNORECASE),
    re.compile(r"^\s*(//|/\*|<!--)"),
    re.compile(r"\b(for example|e\.g\.|such as|imagine|suppose|would look like)\b", re.IGNORECASE),
    re.compile(r"\.\.\.$"),
)


def looks_fabricated(evidence: str) -> bool:
    """Heuristic: does this read as authored illustration rather than a quotation?

    Multi-line evidence with comment scaffolding is the strongest signal -- real
    code lines are single lines without a "# In file.py:" preamble.
    """
    s = evidence.strip()
    if not s:
        return False
    if "\n" in s and any(ln.strip().startswith("#") for ln in s.splitlines()):
        return True
    return any(p.search(s) for p in _FABRICATION_HINTS)


def _coerce(raw: dict) -> Finding:
    sev = str(raw.get("severity", "medium")).lower().strip()
    cat = str(raw.get("category", "other")).lower().strip()
    files = raw.get("files") or []
    if isinstance(files, str):
        files = [files]
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Finding(
        title=str(raw.get("title", "(untitled)"))[:200],
        severity=sev if sev in SEVERITIES else "medium",
        category=cat if cat in CATEGORIES else "other",
        files=[str(f) for f in files][:8],
        evidence=str(raw.get("evidence", ""))[:500],
        why_cross_file=str(raw.get("why_cross_file", ""))[:1000],
        confidence=min(max(conf, 0.0), 1.0),
    )


def parse_findings(model_text: str, packed: PackedContext) -> list[Finding]:
    """Parse, validate, and ground findings against the packed stream."""
    known_paths = {s.path for s in packed.segments}
    findings: list[Finding] = []

    for raw in extract_json_array(model_text):
        f = _coerce(raw)

        # Drop file paths that were never in the stream -- the model made them up.
        real, fake = [], []
        for p in f.files:
            (real if p in known_paths else fake).append(p)
        if fake:
            f.notes.append(f"paths not in stream: {', '.join(fake[:4])}")
        f.files = real

        # Ground on the evidence line.
        if f.evidence:
            hit = packed.resolve_snippet(f.evidence)
            if hit:
                path, line = hit
                f.location = f"{path}:{line}"
                f.grounded = True
                if path not in f.files:
                    f.files.insert(0, path)
            elif looks_fabricated(f.evidence):
                f.notes.append("evidence appears AUTHORED, not quoted (illustrative example)")
            else:
                f.notes.append("evidence line not found in stream")
        else:
            f.notes.append("no evidence provided")

        if not f.is_cross_file and f.category != "hardcoded_secret":
            f.notes.append("single-file finding (out of scope for the cross-file claim)")

        findings.append(f)

    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: (not f.grounded, order.get(f.severity, 9), -f.confidence))
    return findings


@dataclass
class AuditReport:
    repo: str
    model: str
    findings: list[Finding]
    pack_stats: dict
    gen_stats: dict

    @property
    def grounded(self) -> list[Finding]:
        return [f for f in self.findings if f.grounded]

    @property
    def cross_file(self) -> list[Finding]:
        return [f for f in self.grounded if f.is_cross_file]

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "model": self.model,
            "summary": {
                "total": len(self.findings),
                "grounded": len(self.grounded),
                "cross_file": len(self.cross_file),
            },
            "pack": self.pack_stats,
            "generation": self.gen_stats,
            "findings": [f.to_dict() for f in self.findings],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def pretty(self) -> str:
        head = (
            f"\n=== WCA audit: {self.repo} ===\n"
            f"{len(self.findings)} findings; {len(self.grounded)} grounded to a real "
            f"file:line; {len(self.cross_file)} genuinely cross-file\n"
        )
        if not self.findings:
            return head + "\n(no findings)\n"
        body = "\n\n".join(f.pretty() for f in self.findings)
        tail = "\n\n'??' = the model's evidence line was not found in the stream; treat as unverified.\n"
        return head + "\n" + body + tail
