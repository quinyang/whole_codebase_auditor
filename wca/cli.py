"""`wca` command line entrypoint.

    wca graph <target>            symbol graph stats only (CPU, seconds)
    wca pack  <target>            build + dump the packed context (CPU)
    wca scan  <target>            full audit, requires a GPU
    wca env                       report GPU / dtype / kernel availability

<target> is either 'owner/repo', a GitHub URL, or a local directory path.

Everything through `pack` runs without torch installed, which is what makes the
packer independently testable and the Colab notebook a thin driver rather than
the program.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from wca import __version__
from wca.graph import build_graph
from wca.ingest import ingest
from wca.pack import pack
from wca.parse import LanguageDispatcher, parse_files


def _pipeline(args, tokenizer=None):
    """Ingest -> parse -> graph -> pack. Shared by pack/scan."""
    t0 = time.perf_counter()
    bundle = ingest(args.target, args.ref)
    print(f"[1/4] ingest    {bundle.summary()}  ({time.perf_counter() - t0:.1f}s)")

    t = time.perf_counter()
    parsed = parse_files(bundle.files, LanguageDispatcher(quiet=not args.verbose))
    print(f"[2/4] parse     {parsed.summary()}  ({time.perf_counter() - t:.1f}s)")
    if parsed.failed and args.verbose:
        for p, e in parsed.failed[:10]:
            print(f"        ! {p}: {e}")
    if not parsed.files:
        sys.exit("no parseable source files found")

    t = time.perf_counter()
    graph = build_graph(parsed.files)
    print(f"[3/4] graph     {graph.summary()}  ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    packed = pack(
        parsed.files,
        graph,
        budget_tokens=args.budget,
        tokenizer=tokenizer,
        repo_name=bundle.name,
    )
    counter = "exact" if tokenizer is not None else "estimated"
    print(f"[4/4] pack      {packed.stats_line()} [{counter}]  ({time.perf_counter() - t:.1f}s)")
    return bundle, parsed, graph, packed


def cmd_graph(args) -> int:
    bundle = ingest(args.target, args.ref)
    print(bundle.summary())
    parsed = parse_files(bundle.files, LanguageDispatcher(quiet=not args.verbose))
    print(parsed.summary())
    graph = build_graph(parsed.files)
    print(graph.summary())

    risky = sorted(graph.symbols.values(), key=lambda s: -s.risk_score)[:15]
    print("\ntop files by risk score (heuristic, not a vulnerability judgement):")
    for fs in risky:
        if fs.risk_score <= 0:
            break
        hits = ", ".join(sorted(fs.danger_hits)[:6]) or "-"
        print(f"  {fs.risk_score:6.1f}  {fs.path}")
        print(f"          secrets={len(fs.secretish)} sinks=[{hits}]")

    print("\nsample cross-file edges:")
    for e in graph.edges[:20]:
        print(f"  {e.kind:8} {e.src} -> {e.dst}   ({e.detail[:50]})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "files": {
                        p: {
                            "lang": s.lang,
                            "imports": s.imports,
                            "defs": [d.name for d in s.defs],
                            "danger_hits": sorted(s.danger_hits),
                            "secretish": s.secretish,
                            "risk_score": s.risk_score,
                        }
                        for p, s in graph.symbols.items()
                    },
                    "edges": [vars(e) for e in graph.edges],
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out}")
    return 0


def cmd_pack(args) -> int:
    _, _, _, packed = _pipeline(args)
    out = args.out or "context.pack.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(packed.text)
    manifest = out.replace(".txt", "") + ".manifest.json"
    packed.write_manifest(manifest)
    print(f"\nwrote {out} ({len(packed.text):,} chars) and {manifest}")

    # Sanity check that grounding actually works before spending GPU minutes.
    if packed.segments:
        seg = packed.segments[0]
        probe = seg.char_start + min(50, max(seg.char_end - seg.char_start - 1, 0))
        print(f"offset probe: {probe} -> {packed.resolve(probe)}")
    return 0


def cmd_scan(args) -> int:
    # infer.py imports torch lazily so the CPU stages stay usable without it;
    # check explicitly here rather than letting a ModuleNotFoundError surface.
    for mod, hint in (("torch", "torch"), ("transformers", "transformers")):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(
                f"`wca scan` needs {hint}, which is not installed.\n"
                f"  pip install 'wca[gpu]'   (or run the CPU stages: wca pack <target>)"
            )

    from wca.findings import AuditReport, parse_findings
    from wca.infer import DEFAULT_MODEL, MambaAuditor

    auditor = MambaAuditor(args.model or DEFAULT_MODEL, load_in_4bit=not args.no_4bit)
    bundle, _, _, packed = _pipeline(args, tokenizer=auditor.tokenizer)

    print("\n[5/5] inference (prefill dominates; expect minutes on a T4)")
    gen = auditor.generate(packed.text, max_new_tokens=args.max_new_tokens)
    print(f"        {gen.stats_line()}")

    findings = parse_findings(gen.text, packed)
    report = AuditReport(
        repo=f"{bundle.name}@{bundle.ref}",
        model=auditor.model_id,
        findings=findings,
        pack_stats={
            "budget_tokens": packed.budget_tokens,
            "used_tokens": packed.used_tokens,
            "full": packed.n_full,
            "signature": packed.n_signature,
            "omitted": packed.n_omitted,
        },
        gen_stats={
            "prompt_tokens": gen.prompt_tokens,
            "output_tokens": gen.output_tokens,
            "total_seconds": round(gen.total_seconds, 2),
        },
    )
    print(report.pretty())

    out = args.out or "findings.json"
    report.save(out)
    if args.raw:
        with open(args.raw, "w", encoding="utf-8") as fh:
            fh.write(gen.text)
    print(f"wrote {out}")
    return 0


def cmd_env(args) -> int:
    from wca.infer import describe_environment

    print(describe_environment())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wca", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"wca {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, with_budget=True):
        sp.add_argument("target", help="owner/repo, GitHub URL, or local directory")
        sp.add_argument("--ref", default="main", help="branch, tag, or SHA (default: main)")
        sp.add_argument("-o", "--out", help="output file")
        sp.add_argument("-v", "--verbose", action="store_true")
        if with_budget:
            sp.add_argument(
                "--budget",
                type=int,
                default=int(os.getenv("WCA_BUDGET", "24000")),
                help="context token budget (default 24000; model max is 32768)",
            )

    sp = sub.add_parser("graph", help="build and inspect the symbol graph")
    common(sp, with_budget=False)
    sp.set_defaults(func=cmd_graph)

    sp = sub.add_parser("pack", help="build the packed context + manifest (CPU only)")
    common(sp)
    sp.set_defaults(func=cmd_pack)

    sp = sub.add_parser("scan", help="full audit (needs a GPU)")
    common(sp)
    sp.add_argument("--model", help="HF model id")
    sp.add_argument("--no-4bit", action="store_true", help="load in fp16/bf16 instead")
    sp.add_argument("--max-new-tokens", type=int, default=1024)
    sp.add_argument("--raw", help="also dump raw model output here")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("env", help="report GPU, dtype, and kernel availability")
    sp.set_defaults(func=cmd_env)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
