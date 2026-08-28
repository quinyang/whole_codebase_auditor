"""Stage 6 -- Findings.

Turn raw model text into a validated JSON schema, and ground every finding in a
real (file, line) using the packer's manifest.

Grounding is the honesty mechanism of this project. A finding whose `evidence`
line cannot be located in the packed stream is marked `grounded=False`. Those
are reported separately and excluded from the precision number -- an ungrounded
finding is a hallucination by construction, and the eval must say so.
"""

from __future__ import annotations

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


def extract_json_array(text: str) -> list[dict]:
    """Pull the first well-formed JSON array out of model output.

    Models wrap JSON in prose or fences, and sometimes truncate mid-array. This
    tries a fenced block, then a bracket-balanced scan, then a salvage pass that
    collects individually-parseable objects.
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    for cand in candidates:
        start = cand.find("[")
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cand)):
            ch = cand[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cand[start : i + 1])
                        if isinstance(parsed, list):
                            return [p for p in parsed if isinstance(p, dict)]
                    except json.JSONDecodeError:
                        break
    # salvage: individual objects, e.g. from a truncated generation
    out: list[dict] = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "title" in obj:
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


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
