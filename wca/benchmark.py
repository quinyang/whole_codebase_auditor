"""Session 3 -- the benchmark corpus, runner, and score.

Each repo is audited three ways:

    near    a cross-file vulnerability planted into tightly-coupled files
    far     the same patterns planted into distant files
    clean   the repo untouched -- a negative control

Precision is unmeasurable without the clean runs, and the near/far split is what
tests whether graph ordering earns its complexity. Both are structural, not
optional extras.

Only **grounded** findings are scored. Measured in session 2: 100% of findings
ground on a repo with planted vulnerabilities, 0% on a clean one. An ungrounded
finding is unverifiable by construction, so counting it would be counting noise.

Run `prepare_corpus()` on CPU first. It downloads, parses, injects, and checks
that every planted line resolves through the manifest -- catching a dead repo or
a broken injection for free, instead of 45 minutes into a GPU booking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from wca.corpus import SYNTHETIC_CORPUS, generate_corpus
from wca.graph import build_graph
from wca.ingest import RepoBundle, ingest
from wca.inject import BenchmarkCase, inject, verify
from wca.pack import pack
from wca.parse import LanguageDispatcher, parse_files

# Small, pure-Python repos. Chosen so most files fit inside the ~5,300-token
# ceiling measured on a T4, which keeps recall a measure of *detection* rather
# than of the packer's omission behaviour under memory pressure.
DEFAULT_CORPUS: tuple[str, ...] = (
    "pallets/itsdangerous",
    "pallets/markupsafe",
    "theskumar/python-dotenv",
    "jd/tenacity",
    "pytoolz/toolz",
    "Suor/funcy",
    "john-kurkowski/tldextract",
    "lepture/mistune",
    "psf/cachecontrol",
    "seatgeek/thefuzz",
)

# Must mirror pack.pack()'s defaults: it reserves this many tokens for the
# preamble and the model's output, then spends at most FULL_BODY_SHARE of the
# remainder on full bodies. A slice sized to the raw budget overshoots badly.
RESERVE_TOKENS = 2_000
FULL_BODY_SHARE = 0.70

# Line tolerance when matching a finding to a planted spot. The model quotes a
# line; a couple of lines of drift is still the same defect.
LINE_TOLERANCE = 3


@dataclass
class RepoCases:
    """One repo, prepared in all three variants."""

    repo: str
    ref: str
    n_files: int
    near: BenchmarkCase | None = None
    far: BenchmarkCase | None = None
    clean: BenchmarkCase | None = None
    error: str = ""
    whole_repo_coverage: float = 0.0  # planted vulns the packer keeps at full scale
    sliced: bool = False

    def variants(self) -> list[tuple[str, BenchmarkCase]]:
        return [
            (name, case)
            for name, case in (("near", self.near), ("far", self.far), ("clean", self.clean))
            if case is not None
        ]


@dataclass
class AuditOutcome:
    repo: str
    variant: str
    prompt_tokens: int = 0
    n_proposed: int = 0
    n_grounded: int = 0
    true_positives: int = 0
    false_positives: int = 0
    n_planted: int = 0
    planted_found: list[str] = field(default_factory=list)
    planted_missed: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Preparation (CPU only)
# --------------------------------------------------------------------------- #


def prepare_repo(
    spec: str,
    *,
    seed: int,
    max_files: int = 80,
    budget: int = 4_000,
    bundle: RepoBundle | None = None,
) -> RepoCases:
    """Build one repo's three variants. Never raises.

    Pass `bundle` to use an already-constructed repo (the synthetic corpus);
    otherwise `spec` is downloaded from GitHub.
    """
    ref = "synthetic"
    if bundle is None:
        last: Exception | None = None
        for ref in ("main", "master"):
            try:
                bundle = ingest(spec, ref)
                break
            except Exception as exc:  # network, 404, wrong default branch
                last = exc
        else:
            return RepoCases(repo=spec, ref="?", n_files=0, error=f"ingest failed: {last}")
    else:
        ref = bundle.ref

    parsed = parse_files(bundle.files, LanguageDispatcher())
    py = [f for f in parsed.files if f.lang == "python"]
    if len(py) < 4:
        return RepoCases(repo=spec, ref=ref, n_files=len(py), error="too few python files")
    if len(py) > max_files:
        return RepoCases(repo=spec, ref=ref, n_files=len(py), error=f"too large ({len(py)} files)")

    graph = build_graph(parsed.files)
    cases = RepoCases(repo=spec, ref=ref, n_files=len(py))
    cases.near = inject(bundle, graph, seed=seed, distance="near")
    cases.far = inject(bundle, graph, seed=seed + 1, distance="far")
    cases.clean = inject(bundle, graph, seed=seed, patterns=())

    for name, case in cases.variants():
        problems = verify(case)
        if problems:
            cases.error = f"{name}: {problems[0]}"
            return cases

    # How much would the packer surface at WHOLE-REPO scale? Recorded before
    # slicing, because it is a real property of the packer under the T4 ceiling
    # and it disappears once we slice.
    planted_cases = [c for _, c in cases.variants() if c.planted]
    if planted_cases:
        cases.whole_repo_coverage = sum(
            packer_coverage(c, budget) for c in planted_cases
        ) / len(planted_cases)

    # Slice so both halves of every planted vulnerability reach the model.
    # Otherwise a 20k-token repo on a 4k budget scores every case a miss for
    # reasons of memory, not detection.
    cases.near = slice_to_budget(cases.near, budget)
    cases.far = slice_to_budget(cases.far, budget)
    cases.sliced = True

    for name, case in cases.variants():
        if case.planted and not _planted_lines_ground(case, budget):
            need = required_tokens(case)
            capacity = int((budget - RESERVE_TOKENS) * FULL_BODY_SHARE)
            cases.error = (
                f"{name}: required files are {need:,} tok; budget {budget:,} gives the "
                f"packer ~{capacity:,} tok of full bodies. Files too large for this GPU."
            )
            return cases
    return cases


def required_tokens(case: BenchmarkCase) -> int:
    """Estimated tokens of the files a planted vulnerability spans.

    If this exceeds the budget the case is unauditable on this GPU no matter how
    good the packer is -- the two halves physically cannot both be in context.
    """
    from wca.pack import CHARS_PER_TOKEN

    by_path = {f.path: f for f in case.files}
    required = {p for planted in case.planted for p in planted.requires_files}
    return int(sum(len(by_path[p].text) for p in required if p in by_path) / CHARS_PER_TOKEN)


def packer_coverage(case: BenchmarkCase, budget: int) -> float:
    """Fraction of planted vulnerabilities whose BOTH halves survive packing.

    Reported separately from detection, because they are separate abilities and
    conflating them produces a number that means nothing. Measured on a T4 with
    a ~5,300-token ceiling, whole-repo coverage of a realistic 20k-token repo is
    poor -- and that is a fact about the hardware, not about the model.
    """
    if not case.planted:
        return 1.0
    parsed = parse_files(case.files, LanguageDispatcher())
    graph = build_graph(parsed.files)
    packed = pack(parsed.files, graph, budget_tokens=budget, repo_name=case.repo)
    covered = sum(
        all(
            packed.resolve_snippet(spot.code) == (spot.file, spot.line)
            for spot in planted.spots.values()
        )
        for planted in case.planted
    )
    return covered / len(case.planted)


def slice_to_budget(case: BenchmarkCase, budget: int, *, fill: float = 0.9) -> BenchmarkCase:
    """Keep the planted files plus their highest-priority neighbours; drop the rest.

    Without this, a realistic 20k-token repo cannot be audited on a T4 at all:
    at budget 4000 the packer has ~1,400 tokens for full bodies, and a single
    6 KB source file does not fit. Every planted vulnerability would be scored a
    miss for a reason that has nothing to do with detection.

    Slicing makes the benchmark measure one thing cleanly -- **can the model see
    a cross-file defect when both halves are in context** -- and leaves the
    separate question of whether the packer would have surfaced those files at
    whole-repo scale to `packer_coverage()`. Two honest numbers beat one
    ambiguous one.
    """
    from wca.pack import CHARS_PER_TOKEN, compute_priority

    if not case.planted:
        return case

    by_path = {f.path: f for f in case.files}
    required = {p for planted in case.planted for p in planted.requires_files}

    parsed = parse_files(case.files, LanguageDispatcher())
    graph = build_graph(parsed.files)
    priority = compute_priority(graph)

    def cost(path: str) -> int:
        return int(len(by_path[path].text) / CHARS_PER_TOKEN)

    keep = set(required)
    spent = sum(cost(p) for p in keep)

    # Size the slice to the packer's FULL-BODY capacity, not to the raw budget.
    # `pack()` holds back `reserve_tokens` for the preamble and the model's own
    # output, then spends at most `full_body_share` of what remains on full
    # bodies -- at budget 4,000 that is ~1,400 tokens, not 4,000. Slicing to a
    # fraction of the raw budget left the required files just over the line, so
    # the packer demoted one to signature mode and elided the very body line the
    # vulnerability lives on. The symptom was a file present in the stream whose
    # planted line could not be found.
    full_capacity = int((budget - RESERVE_TOKENS) * FULL_BODY_SHARE)
    ceiling = max(int(full_capacity * fill), spent)

    others = sorted(
        (p for p in by_path if p not in keep),
        key=lambda p: (-priority.get(p, 0.0), cost(p)),
    )
    for path in others:
        c = cost(path)
        if spent + c > ceiling:
            continue
        keep.add(path)
        spent += c

    sliced = BenchmarkCase(
        repo=case.repo,
        seed=case.seed,
        files=[by_path[p] for p in sorted(keep)],
        planted=case.planted,
        clean_files=case.clean_files & keep,
    )
    return sliced


def _planted_lines_ground(case: BenchmarkCase, budget: int) -> bool:
    return packer_coverage(case, budget) == 1.0


def prepare_synthetic_corpus(
    names: tuple[str, ...] = SYNTHETIC_CORPUS,
    *,
    seed: int = 20260829,
    budget: int = 4_000,
) -> list[RepoCases]:
    """The corpus that actually fits a T4. See `wca/corpus.py` for why.

    No network, so this is fast and reproducible from the seed alone.
    """
    out: list[RepoCases] = []
    for i, bundle in enumerate(generate_corpus(names, seed=seed)):
        cases = prepare_repo(
            bundle.name, seed=seed + i * 10, budget=budget, bundle=bundle
        )
        status = "OK" if not cases.error else "SKIP"
        cov = f"{cases.whole_repo_coverage:.0%}" if not cases.error else "-"
        print(f"  {status:4} {bundle.name:24} {cases.n_files:>3} py | coverage {cov:>4}")
        if cases.error:
            print(f"       {cases.error[:96]}")
        out.append(cases)

    usable = [c for c in out if not c.error]
    n_planted = sum(len(c.planted) for r in usable for _, c in r.variants())
    print(
        f"\n{len(usable)}/{len(names)} repos usable | {len(usable) * 3} audits | "
        f"{n_planted} planted vulnerabilities"
    )
    if usable:
        mean = sum(c.whole_repo_coverage for c in usable) / len(usable)
        print(f"Whole-repo packer coverage {mean:.0%} at budget {budget:,} (no slicing needed).")
    return out


def prepare_corpus(
    specs: tuple[str, ...] = DEFAULT_CORPUS,
    *,
    seed: int = 1234,
    max_files: int = 80,
    budget: int = 4_000,
) -> list[RepoCases]:
    """CPU-only preflight. Run this before booking GPU time."""
    out: list[RepoCases] = []
    for i, spec in enumerate(specs):
        cases = prepare_repo(spec, seed=seed + i * 10, max_files=max_files, budget=budget)
        status = "OK" if not cases.error else "SKIP"
        cov = f"{cases.whole_repo_coverage:.0%}" if not cases.error else "-"
        print(f"  {status:4} {spec:32} {cases.n_files:>3} py | coverage {cov:>4}")
        if cases.error:
            print(f"       {cases.error[:96]}")
        out.append(cases)

    usable = [c for c in out if not c.error]
    n_planted = sum(len(c.planted) for r in usable for _, c in r.variants())
    print(
        f"\n{len(usable)}/{len(specs)} repos usable | "
        f"{len(usable) * 3} audits | {n_planted} planted vulnerabilities"
    )
    if len(usable) < len(specs):
        print("Skipped repos are excluded from the score, not counted as misses.")
    if usable:
        mean_cov = sum(c.whole_repo_coverage for c in usable) / len(usable)
        print(
            f"\nWhole-repo packer coverage: {mean_cov:.0%} of planted vulnerabilities "
            f"would have both halves\nin context at budget {budget:,} without slicing. "
            f"Report this alongside detection --\nit is the T4 memory ceiling, measured, "
            f"not a model failure."
        )
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _matches_planted(finding, case: BenchmarkCase) -> str | None:
    """Return the id of the planted vulnerability this finding hits, or None.

    A hit requires either the grounded location to land within `LINE_TOLERANCE`
    of a planted spot, or the finding to name both files the vulnerability spans.
    Location is the stronger signal; file pairs catch the case where the model
    quotes one half and the symbol graph supplies the other.
    """
    if not finding.grounded or not finding.location:
        return None
    path, _, line_s = finding.location.rpartition(":")
    try:
        line = int(line_s)
    except ValueError:
        return None

    for planted in case.planted:
        for spot in planted.spots.values():
            if spot.file == path and abs(spot.line - line) <= LINE_TOLERANCE:
                return planted.id
    for planted in case.planted:
        if set(planted.requires_files) <= set(finding.files):
            return planted.id
    return None


def score_case(findings: list, case: BenchmarkCase, variant: str, repo: str) -> AuditOutcome:
    out = AuditOutcome(
        repo=repo, variant=variant, n_proposed=len(findings), n_planted=len(case.planted)
    )
    grounded = [f for f in findings if f.grounded]
    out.n_grounded = len(grounded)

    hit_ids: set[str] = set()
    for f in grounded:
        matched = _matches_planted(f, case)
        if matched:
            hit_ids.add(matched)
        else:
            # Anything grounded that matches nothing planted is a false positive.
            # On a clean repo every grounded finding lands here, which is exactly
            # what makes the negative controls do their job.
            out.false_positives += 1

    out.true_positives = len(hit_ids)
    out.planted_found = sorted(hit_ids)
    out.planted_missed = sorted({p.id for p in case.planted} - hit_ids)
    return out


def summarise(outcomes: list[AuditOutcome]) -> dict[str, Any]:
    ok = [o for o in outcomes if not o.error]
    tp = sum(o.true_positives for o in ok)
    fp = sum(o.false_positives for o in ok)
    planted = sum(o.n_planted for o in ok)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / planted if planted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("near", "far", "clean"):
        rows = [o for o in ok if o.variant == variant]
        v_tp = sum(o.true_positives for o in rows)
        v_fp = sum(o.false_positives for o in rows)
        v_planted = sum(o.n_planted for o in rows)
        by_variant[variant] = {
            "audits": len(rows),
            "planted": v_planted,
            "true_positives": v_tp,
            "false_positives": v_fp,
            "recall": round(v_tp / v_planted, 3) if v_planted else None,
            "proposed": sum(o.n_proposed for o in rows),
            "grounded": sum(o.n_grounded for o in rows),
        }

    return {
        "audits": len(ok),
        "failed_audits": len(outcomes) - len(ok),
        "planted_total": planted,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "by_variant": by_variant,
    }


def print_summary(outcomes: list[AuditOutcome]) -> dict[str, Any]:
    s = summarise(outcomes)
    print(f"\n{'repo':32} {'variant':8} {'plant':>5} {'prop':>5} {'grnd':>5} {'TP':>4} {'FP':>4}")
    print("-" * 76)
    for o in outcomes:
        note = f"  {o.error[:24]}" if o.error else ""
        print(
            f"{o.repo[:32]:32} {o.variant:8} {o.n_planted:>5} {o.n_proposed:>5} "
            f"{o.n_grounded:>5} {o.true_positives:>4} {o.false_positives:>4}{note}"
        )

    print("\n" + "=" * 76)
    print("RESULT (grounded findings only)")
    print("=" * 76)
    print(f"  precision {s['precision']:.1%}   ({s['true_positives']} TP / "
          f"{s['true_positives'] + s['false_positives']} reported)")
    print(f"  recall    {s['recall']:.1%}   ({s['true_positives']} / {s['planted_total']} planted)")
    print(f"  F1        {s['f1']:.3f}")

    near, far = s["by_variant"]["near"], s["by_variant"]["far"]
    if near["recall"] is not None and far["recall"] is not None:
        print("\n  ABLATION -- does proximity in the packed stream matter?")
        print(f"    near-distance recall {near['recall']:.1%}  ({near['true_positives']}/{near['planted']})")
        print(f"    far-distance  recall {far['recall']:.1%}  ({far['true_positives']}/{far['planted']})")
        delta = near["recall"] - far["recall"]
        if abs(delta) < 0.1:
            print(f"    delta {delta:+.1%} -- no clear effect. Report it honestly; a null")
            print("      result on your own ablation is more credible than a claim.")
        elif delta > 0:
            print(f"    delta {delta:+.1%} -- proximity helps, which is the packer's whole")
            print("      premise: an SSM's fixed-size state loses distant detail.")
        else:
            print(f"    delta {delta:+.1%} -- proximity HURT. Worth investigating before")
            print("      claiming graph ordering is a benefit.")

    clean = s["by_variant"]["clean"]
    print(f"\n  NEGATIVE CONTROLS: {clean['audits']} clean repos, "
          f"{clean['proposed']} findings proposed, {clean['grounded']} grounded, "
          f"{clean['false_positives']} false positives")
    if clean["proposed"] and not clean["grounded"]:
        print("    Grounding rejected every proposal on clean repos -- the filter,")
        print("    not the model, is what delivers precision. State it that way.")
    return s


# --------------------------------------------------------------------------- #
# Runner (GPU)
# --------------------------------------------------------------------------- #


def run_benchmark(
    corpus: list[RepoCases],
    *,
    auditor=None,
    budget: int = 4_000,
    max_new_tokens: int = 1024,
    checkpoint_dir: str | None = "/content/wca_runs",
) -> tuple[list[AuditOutcome], dict[str, Any]]:
    """Audit every prepared variant. Checkpoints after each one.

    Colab sessions die; a 45-minute run that loses everything on a disconnect is
    a 45-minute run you have to do again.
    """
    from wca.findings import enrich_with_graph, parse_findings
    from wca.infer import load_auditor

    todo = [(r, name, case) for r in corpus if not r.error for name, case in r.variants()]
    if not todo:
        # Emitting "precision 0.0%" for an empty corpus produces a number that
        # looks like a result and is not one. Refuse instead.
        skipped = [r for r in corpus if r.error]
        msg = [
            f"Nothing to audit: {len(corpus)} repos in, {len(skipped)} skipped, 0 usable.",
            "Run prepare_corpus() first (cell 2) and check its output.",
        ]
        for r in skipped[:10]:
            msg.append(f"  SKIP {r.repo}: {r.error[:70]}")
        raise ValueError("\n".join(msg))

    if auditor is None:
        auditor = load_auditor()

    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)

    outcomes: list[AuditOutcome] = []
    usable = len({r.repo for r, _, _ in todo})
    print(f"{len(todo)} audits queued across {usable} repos\n")

    for i, (repo_cases, variant, case) in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] {repo_cases.repo} :: {variant}"
        parsed = parse_files(case.files, LanguageDispatcher())
        graph = build_graph(parsed.files)
        packed = pack(
            parsed.files, graph, budget_tokens=budget,
            tokenizer=auditor.tokenizer, repo_name=repo_cases.repo,
        )
        try:
            gen = auditor.generate(packed.text, max_new_tokens=max_new_tokens)
        except Exception as exc:
            outcome = AuditOutcome(
                repo=repo_cases.repo, variant=variant, n_planted=len(case.planted),
                error=f"{type(exc).__name__}: {str(exc)[:120]}",
            )
            outcomes.append(outcome)
            print(f"{label}: FAILED {outcome.error[:60]}")
            continue

        findings = enrich_with_graph(parse_findings(gen.text, packed), graph)
        outcome = score_case(findings, case, variant, repo_cases.repo)
        outcome.prompt_tokens = gen.prompt_tokens
        outcome.seconds = round(gen.total_seconds, 1)
        outcomes.append(outcome)
        print(
            f"{label}: {outcome.n_proposed} proposed -> {outcome.n_grounded} grounded | "
            f"TP {outcome.true_positives}/{outcome.n_planted} FP {outcome.false_positives} | "
            f"{outcome.seconds:.0f}s"
        )

        if ckpt:
            (ckpt / "outcomes.json").write_text(
                json.dumps([o.to_dict() for o in outcomes], indent=2)
            )
            (ckpt / f"raw_{repo_cases.repo.replace('/', '__')}_{variant}.txt").write_text(gen.text)

    summary = print_summary(outcomes)
    if ckpt:
        (ckpt / "summary.json").write_text(json.dumps(summary, indent=2))
        (ckpt / "ground_truth.json").write_text(
            json.dumps(
                {f"{r.repo}::{n}": c.ground_truth() for r in corpus if not r.error
                 for n, c in r.variants()},
                indent=2,
            )
        )
        print(f"\nsaved -> {ckpt}/  (outcomes.json, summary.json, ground_truth.json, raw_*.txt)")
    return outcomes, summary
