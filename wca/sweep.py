"""Session 2 -- context sweep.

Answers the question the project has been blocked on: **how far does context
length scale before output quality or memory gives out?**

Deliberately lives in the package rather than in notebook cells. A Colab
notebook is a copy; edits pushed to the repo never reach an open tab, so logic
in cells silently goes stale while `pip install --upgrade` does reach the
package. One call from one cell keeps the two in sync.

    from wca.sweep import run_demo, run_sweep
    run_demo()                       # toy fixture, scored against ground truth
    run_sweep("pallets/click")       # degradation curve across budgets

Every row records four independent things, because they fail independently:
  json_valid     did the model emit parseable structured output
  n_findings     did it find anything
  grounding_rate did its evidence resolve to real source lines
  peak_gib       what it cost
"""

from __future__ import annotations

import gc
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from wca.findings import Finding, extract_json_array, looks_fabricated, parse_findings
from wca.fixtures import GROUND_TRUTH, materialise_toy_vuln
from wca.graph import build_graph
from wca.ingest import ingest
from wca.pack import pack
from wca.parse import LanguageDispatcher, parse_files

DEFAULT_BUDGETS = (2_000, 4_000, 8_000, 16_000, 24_000)


@dataclass
class SweepRow:
    budget: int
    prompt_tokens: int = 0
    n_files_full: int = 0
    n_files_omitted: int = 0
    json_valid: bool = False
    n_findings: int = 0
    n_grounded: int = 0
    n_cross_file: int = 0
    n_fabricated: int = 0
    peak_gib: float = 0.0
    prefill_s: float = 0.0
    total_s: float = 0.0
    error: str = ""
    raw_output: str = field(default="", repr=False)
    findings: list[Finding] = field(default_factory=list, repr=False)
    ungrounded: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def grounding_rate(self) -> float:
        return self.n_grounded / self.n_findings if self.n_findings else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("findings", None)
        d["grounding_rate"] = round(self.grounding_rate, 3)
        return d


def _peak_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        pass
    return 0.0


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def audit_once(
    auditor,
    parsed_files,
    graph,
    budget: int,
    *,
    repo_name: str = "repo",
    max_new_tokens: int = 512,
) -> SweepRow:
    """One audit at one budget. Never raises -- OOM is data, not a crash."""
    row = SweepRow(budget=budget)
    _reset_peak()

    packed = pack(
        parsed_files, graph, budget_tokens=budget,
        tokenizer=auditor.tokenizer, repo_name=repo_name,
    )
    row.n_files_full = packed.n_full
    row.n_files_omitted = packed.n_omitted

    try:
        gen = auditor.generate(packed.text, max_new_tokens=max_new_tokens)
    except Exception as exc:  # OOM, context overflow, anything
        row.error = f"{type(exc).__name__}: {str(exc)[:160]}"
        row.peak_gib = _peak_gib()
        return row

    row.prompt_tokens = gen.prompt_tokens
    row.prefill_s = round(gen.prefill_seconds, 1)
    row.total_s = round(gen.total_seconds, 1)
    row.peak_gib = _peak_gib()
    row.raw_output = gen.text

    # Parsed-as-JSON and grounded are separate questions; measure both.
    row.json_valid = bool(extract_json_array(gen.text))
    findings = parse_findings(gen.text, packed)
    row.findings = findings
    row.n_findings = len(findings)
    row.n_grounded = sum(f.grounded for f in findings)
    row.n_cross_file = sum(f.grounded and f.is_cross_file for f in findings)
    row.n_fabricated = sum(
        not f.grounded and looks_fabricated(f.evidence) for f in findings
    )

    # Why did anything fail to ground? A near-miss and a fabrication both
    # show up as 0%, and they need opposite fixes.
    for f in findings:
        if f.grounded or not f.evidence:
            continue
        near = packed.closest_line(f.evidence)
        row.ungrounded.append(
            {
                "evidence": f.evidence[:120],
                "closest": f"{near[0]}:{near[1]}" if near else None,
                "ratio": near[2] if near else 0.0,
                "closest_text": near[3] if near else "",
            }
        )
    return row


def print_table(rows: list[SweepRow]) -> None:
    print(
        f"\n{'budget':>7} {'prompt':>7} {'full':>5} {'omit':>5} "
        f"{'json':>5} {'find':>5} {'grnd':>5} {'rate':>6} "
        f"{'peakGiB':>8} {'prefill':>8} {'note'}"
    )
    print("-" * 88)
    for r in rows:
        note = r.error[:28] if r.error else ""
        print(
            f"{r.budget:>7,} {r.prompt_tokens:>7,} {r.n_files_full:>5} {r.n_files_omitted:>5} "
            f"{'yes' if r.json_valid else 'NO':>5} {r.n_findings:>5} {r.n_grounded:>5} "
            f"{r.grounding_rate:>5.0%} {r.peak_gib:>8.2f} {r.prefill_s:>7.1f}s {note}"
        )


def _prepare(target: str, ref: str = "main", quiet: bool = False):
    bundle = ingest(target, ref)
    parsed = parse_files(bundle.files, LanguageDispatcher())
    graph = build_graph(parsed.files)
    if not quiet:
        print(bundle.summary())
        print(parsed.summary())
        print(graph.summary())
    return bundle, parsed, graph


def run_sweep(
    target: str,
    *,
    ref: str = "main",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    auditor=None,
    max_new_tokens: int = 512,
    stop_after_errors: int = 2,
) -> list[SweepRow]:
    """Audit one repo at increasing budgets; report where quality/memory break.

    Ascending order so the ceiling is found rather than assumed. Stops early
    after consecutive failures -- past the memory wall nothing recovers.
    """
    from wca.infer import MambaAuditor

    if auditor is None:
        auditor = MambaAuditor()

    bundle, parsed, graph = _prepare(target, ref)

    rows: list[SweepRow] = []
    consecutive = 0
    for budget in budgets:
        print(f"\n--- budget {budget:,} ---")
        row = audit_once(
            auditor, parsed.files, graph, budget,
            repo_name=bundle.name, max_new_tokens=max_new_tokens,
        )
        rows.append(row)
        if row.error:
            consecutive += 1
            print(f"  FAILED: {row.error}")
            if consecutive >= stop_after_errors:
                print(f"  stopping -- {consecutive} consecutive failures")
                break
        else:
            consecutive = 0
            print(
                f"  {row.prompt_tokens:,} tok | json={row.json_valid} | "
                f"{row.n_findings} findings, {row.n_grounded} grounded "
                f"({row.grounding_rate:.0%}) | peak {row.peak_gib:.2f} GiB | "
                f"{row.total_s:.0f}s"
            )

    print_table(rows)
    _interpret(rows)
    return rows


def _interpret(rows: list[SweepRow]) -> None:
    """Say what the numbers mean, so a run is self-explaining later."""
    ok = [r for r in rows if not r.error]
    print("\n=== reading the sweep ===")
    if not ok:
        print("  Every budget failed. Check `wca env` and the first error above.")
        return

    largest = max(ok, key=lambda r: r.prompt_tokens)
    print(f"  Largest context that completed: {largest.prompt_tokens:,} tokens "
          f"(peak {largest.peak_gib:.2f} GiB)")

    valid = [r for r in ok if r.json_valid]
    if valid and len(valid) < len(ok):
        last_ok = max(valid, key=lambda r: r.prompt_tokens)
        print(f"  JSON compliance holds to {last_ok.prompt_tokens:,} tokens, then breaks.")
        print("  -> shorten the schema, or move instructions AFTER the code (recency")
        print("     helps an SSM), before blaming the model.")
    elif valid:
        print("  JSON compliance held at every budget tested. Push budgets higher.")
    else:
        print("  No budget produced parseable JSON -- treat the prompt as the suspect.")

    grounded = [r for r in ok if r.n_findings]
    if grounded:
        best = max(grounded, key=lambda r: r.grounding_rate)
        worst = min(grounded, key=lambda r: r.grounding_rate)
        if best.grounding_rate == worst.grounding_rate:
            print(f"  Grounding rate flat at {best.grounding_rate:.0%} across all budgets.")
        else:
            print(f"  Grounding rate {worst.grounding_rate:.0%} at {worst.prompt_tokens:,} tok "
                  f"-> {best.grounding_rate:.0%} at {best.prompt_tokens:,} tok.")
        if best.grounding_rate - worst.grounding_rate > 0.25:
            print("  -> a clear degradation with context length: recall of exact")
            print("     detail decays as SSM state fills.")

    diags = [d for r in ok for d in r.ungrounded]
    if diags:
        print("\n  WHY FINDINGS DID NOT GROUND (closest line in the stream):")
        for d in diags[:6]:
            print(f"    ratio {d['ratio']:.2f}  said: {d['evidence'][:70]!r}")
            print(f"                 near: {d['closest']} {d['closest_text'][:66]!r}")
        top = max(d['ratio'] for d in diags)
        if top >= 0.6:
            print("    -> near-misses: the model is paraphrasing real code. Tighten the")
            print("       prompt ('copy the line character-for-character'), not the matcher.")
        else:
            print("    -> no close match: the model is describing code rather than quoting")
            print("       it. Most likely it cited a file the packer OMITTED -- check the")
            print("       omit column; at these budgets most of the repo is not in context.")

    if len(ok) >= 2:
        a, b = min(ok, key=lambda r: r.prompt_tokens), largest
        if a.prompt_tokens and b.prompt_tokens > a.prompt_tokens:
            per_tok = (b.peak_gib - a.peak_gib) / (b.prompt_tokens - a.prompt_tokens)
            weights = b.peak_gib - per_tok * b.prompt_tokens
            print("\n  MEMORY MODEL (the headline measurement):")
            print(f"    peak_GiB = {weights:.2f} + {per_tok * 1024:.2f} MiB/token x context")
            print(f"    intercept {weights:.2f} GiB = model weights "
                  f"({'4-bit working' if weights < 6 else 'NOT quantised?'})")
            print(f"    slope {per_tok * 1024:.2f} MiB/token = activations. Near 0 would mean")
            print("      fused kernels; a linear slope is the eager path materialising")
            print("      [batch, d_inner, seq_len, d_state].")
            try:
                import torch
                total = torch.cuda.get_device_properties(0).total_memory / 2**30
                ceiling = int((total * 0.92 - weights) / per_tok)
                print(f"    => ceiling on this {total:.1f} GiB GPU: ~{ceiling:,} tokens")
            except Exception:  # noqa: S110 -- GPU total is a nice-to-have
                pass


def run_demo(auditor=None, *, budget: int = 4_000, max_new_tokens: int = 512) -> SweepRow:
    """Audit the embedded toy repo and score against its recorded ground truth.

    The smallest thing that exercises the cross-file claim end to end. Use it to
    confirm a working setup before spending time on a large sweep.
    """
    from wca.infer import MambaAuditor

    if auditor is None:
        auditor = MambaAuditor()

    root = materialise_toy_vuln(tempfile.mkdtemp(prefix="wca_toy_"))
    _bundle, parsed, graph = _prepare(str(root), quiet=True)
    print(f"toy_vuln: {len(parsed.files)} files, {graph.summary()}")

    t0 = time.perf_counter()
    row = audit_once(
        auditor, parsed.files, graph, budget,
        repo_name="toy_vuln", max_new_tokens=max_new_tokens,
    )
    if row.error:
        print(f"FAILED: {row.error}")
        return row

    print(f"\n{row.prompt_tokens:,} tokens | {time.perf_counter() - t0:.0f}s | "
          f"peak {row.peak_gib:.2f} GiB")
    print("\n=============== RAW MODEL OUTPUT ===============")
    print(row.raw_output)
    print("================================================")

    score_against_ground_truth(row.findings)
    return row


def score_against_ground_truth(findings: list[Finding]) -> dict[str, Any]:
    """Score findings against the toy fixture's planted vulnerabilities.

    Correctness and groundedness are reported separately on purpose. Gating one
    on the other once scored a fully correct model as having missed everything.
    """
    named = {f for fi in findings for f in fi.files}

    print("\n=== (a) correctness: right file pairs? ===")
    found = 0
    for planted in GROUND_TRUTH["planted"]:
        need = set(planted["requires_files"])
        hit = need <= named
        found += hit
        status = "FOUND" if hit else ("partial" if need & named else "missed")
        print(f"  {planted['id']:15} [{planted['severity']:8}] {status:8} needs {sorted(need)}")
    print(f"  -> {found}/{len(GROUND_TRUTH['planted'])} planted vulnerabilities identified")

    print("\n=== (b) groundedness: evidence resolves? ===")
    n_grounded = 0
    for f in findings:
        n_grounded += f.grounded
        mark = f"grounded {f.location}" if f.grounded else "UNGROUNDED"
        print(f"  {mark:32} {f.title[:50]}")
        if not f.grounded and f.evidence:
            print(f"      evidence: {f.evidence[:88]!r}")
    rate = n_grounded / len(findings) if findings else 0.0
    print(f"  -> grounding rate {n_grounded}/{len(findings)} ({rate:.0%})")

    clean = {"app/util.py"}
    fps = [f for f in findings if set(f.files) & clean]
    print(f"\nfalse positives against known-clean files: {len(fps)}")

    return {
        "identified": found,
        "planted": len(GROUND_TRUTH["planted"]),
        "grounding_rate": rate,
        "false_positives": len(fps),
    }


# --------------------------------------------------------------------------- #
# Prompt validation -- the two-sided test
# --------------------------------------------------------------------------- #


def run_validation(
    auditor=None,
    *,
    negative_control: str = "pallets/click",
    budget: int = 4_000,
) -> dict[str, Any]:
    """Check the prompt on a repo that HAS vulnerabilities and one that does not.

    A detector is only meaningful if it does both. Measuring recall alone rewards
    a model that reports something for every input; measuring precision alone
    rewards one that reports nothing. `eval/README.md` calls negative controls
    mandatory for exactly this reason, and this is the smallest version of that.

    Expected after prompt v2:
      toy_vuln  -> 2/2 identified, evidence grounds, 0 fabricated
      clean repo -> 0 findings (an empty array is the correct answer)
    """
    from wca.infer import MambaAuditor

    if auditor is None:
        auditor = MambaAuditor()

    print("=" * 70)
    print("POSITIVE CONTROL -- toy_vuln (2 planted cross-file vulnerabilities)")
    print("=" * 70)
    root = materialise_toy_vuln(tempfile.mkdtemp(prefix="wca_toy_"))
    _b, parsed, graph = _prepare(str(root), quiet=True)
    pos = audit_once(auditor, parsed.files, graph, budget, repo_name="toy_vuln")
    if pos.error:
        print(f"FAILED: {pos.error}")
        return {"error": pos.error}
    print(f"{pos.prompt_tokens:,} tok | {pos.total_s:.0f}s | peak {pos.peak_gib:.2f} GiB")
    print("\n--- raw output ---")
    print(pos.raw_output[:1200])
    scored = score_against_ground_truth(pos.findings)

    print("\n" + "=" * 70)
    print(f"NEGATIVE CONTROL -- {negative_control} (expect NO findings)")
    print("=" * 70)
    _b2, parsed2, graph2 = _prepare(negative_control, quiet=True)
    neg = audit_once(auditor, parsed2.files, graph2, budget, repo_name=negative_control)
    if neg.error:
        print(f"FAILED: {neg.error}")
        neg_n = -1
    else:
        neg_n = neg.n_findings
        print(f"{neg.prompt_tokens:,} tok | {neg.total_s:.0f}s | "
              f"{neg_n} findings ({neg.n_fabricated} with authored evidence)")
        if neg_n:
            print("\n--- raw output ---")
            print(neg.raw_output[:900])
            for d in neg.ungrounded[:3]:
                print(f"  ratio {d['ratio']:.2f}  said: {d['evidence'][:70]!r}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    recall_ok = scored["identified"] == scored["planted"]
    ground_ok = scored["grounding_rate"] > 0
    precision_ok = neg_n == 0

    print(f"  recall     : {scored['identified']}/{scored['planted']} planted found"
          f"   {'PASS' if recall_ok else 'FAIL'}")
    print(f"  grounding  : {scored['grounding_rate']:.0%} of findings evidenced"
          f"      {'PASS' if ground_ok else 'FAIL'}")
    print(f"  precision  : {neg_n} findings on a clean repo"
          f"       {'PASS' if precision_ok else 'FAIL' if neg_n > 0 else 'SKIPPED'}")

    if recall_ok and ground_ok and precision_ok:
        print("\n  All three hold. The prompt is sound -- build the benchmark (session 3).")
    elif not ground_ok and pos.n_fabricated:
        print("\n  Model still authors evidence rather than quoting it. Shorten the")
        print("  schema further, or drop `evidence` to a line NUMBER instead of text.")
    elif not precision_ok and neg_n > 0:
        print("\n  False positives on a clean repo. This is the number that would sink")
        print("  a precision claim -- fix before scaling the benchmark.")
    elif not recall_ok:
        print("\n  Missed planted vulnerabilities that a 640-token context contains in")
        print("  full. Prompt or model capability, not context length.")

    return {
        "positive": pos.to_dict(),
        "negative": neg.to_dict() if not neg.error else {"error": neg.error},
        "recall_ok": recall_ok,
        "grounding_ok": ground_ok,
        "precision_ok": precision_ok,
    }
