# Whole-Codebase Auditor (WCA)

A pure-SSM security auditor that reads an entire repository in one linear-time
context pass. A Tree-sitter symbol graph packs cross-file dependencies into a
fixed token budget so the model can see "action-at-a-distance" vulnerabilities —
a credential defined in one file and logged in another, untrusted input entering
at one boundary and reaching a dangerous sink in a different module — that
single-file scanners structurally cannot find.

> **Status: week 1 of 4.** Stages 1–4 (ingest → parse → graph → pack) are
> implemented and tested. Stages 5–6 (inference, findings) are implemented but
> not yet evaluated. There is no precision/recall number yet; see
> [Honest limitations](#honest-limitations).

## Why an SSM

Transformer attention is O(L²) in sequence length, so whole-repo context is
priced out well before a repo of interesting size. Mamba is a state space model:
it carries a fixed-size recurrent state, so a forward pass is O(L) in time and
O(1) in state memory.

The catch, and the reason the context packer exists: **linear scaling is a claim
about asymptotics, not about free VRAM.** A prefill over 200k tokens still costs
real memory and minutes of wall clock on a T4, and Falcon3-Mamba-7B-Instruct was
trained at 32k — beyond that you are extrapolating. So the context is treated as
a fixed budget to be *allocated*, not a bucket to be filled.

## Pipeline

```
  ingest.py    one tarball fetch (codeload) or a local dir  ->  file list + bytes
     |
  parse.py     Tree-sitter AST per file, 7 languages         ->  retained trees
     |
  graph.py     imports / defs / call sites / literals        ->  cross-file edges
     |
  pack.py      budget-aware packer + offset manifest         ->  the context stream
     |
  infer.py     instruction-tuned Mamba, 4-bit                ->  raw model output
     |
  findings.py  JSON schema + location resolution             ->  grounded findings
```

Everything through `pack` runs on CPU without torch installed, which is what
makes the packer independently testable and the Colab notebook a thin driver
rather than the program itself.

### The context packer

Three jobs, and the reason this project is a systems project rather than a
prompting project:

**Order.** Files are grouped by symbol-graph connected component, and ordered
within a component by dependency depth (Kahn's algorithm over import edges), so
code that interacts is adjacent in the stream. An SSM compresses history into a
fixed-size state — distance between two related facts directly costs recall. A
transformer can attend across 100k tokens at quadratic cost; an SSM cannot, so
*layout is the mitigation*.

**Allocate.** Selection and ordering are deliberately separate passes. Selection
decides globally, by priority, which files get a full body, which are reduced to
a signature skeleton (imports + declarations, bodies elided), and which are
dropped. Only then does emission walk the graph order. Conflating the two — a
single greedy pass in stream order — means a low-priority file early in the
dependency order eats budget a high-risk file later needed, and the signature
tier is never reached at all. That was a real bug, caught at 400 files; the
regression test is `test_signature_tier_is_reached_on_a_large_repo`.

Priority is a heuristic risk score (secret-shaped literals, dangerous sinks)
propagated one hop outward along graph edges, because a file that *imports* a
credential is exactly as interesting as the one that defines it — that pair
**is** the cross-file vulnerability.

**Ground.** Every emitted region records a mapping from stream character offset
back to `(file, line)`. Without this, model output cannot be tied to real
locations, and a finding without a location is unusable. Signature mode emits
non-contiguous lines, so segments carry an explicit offset→line map rather than
counting newlines.

Grounding is also the honesty mechanism. `findings.py` marks any finding whose
evidence line cannot be located in the packed stream as `grounded=False`, and
strips file paths the model invented. Ungrounded findings are hallucinations by
construction and are reported separately from the headline number.

## Install

```bash
# CPU: ingest, parse, graph, pack
pip install -e .

# GPU: adds transformers + bitsandbytes for inference
pip install -e ".[gpu]"
```

## Use

```bash
wca graph pallets/flask            # symbol graph stats + top-risk files
wca pack  pallets/flask            # build the context + manifest (CPU, seconds)
wca scan  pallets/flask            # full audit (needs a GPU)
wca env                            # GPU, dtype, and kernel availability
wca pack  ./my-project --budget 16000
```

On Colab, open `notebooks/colab_driver.ipynb` — five cells, all logic imported
from this package.

## Model

`tiiuae/Falcon3-Mamba-7B-Instruct`. Pure Mamba-1, 64 decoder blocks, no
attention anywhere, so the O(L) claim survives — but instruction-tuned, with a
chat template, and a 32k trained context.

This replaces `state-spaces/mamba-2.8b-hf`, which is a **base LM**. The previous
prompt ended in `"1. [CRITICAL] Hardcoded Secret:"` and the model simply
continued that text. It would have emitted a confident finding for an empty
file. That is text continuation, not auditing. Generation is also greedy now:
sampling at `temperature=0.1` on a factual task manufactures findings.

## Hardware notes (Colab)

- **T4 is Turing and has no real bf16.** `bnb_4bit_compute_dtype=bfloat16`
  silently emulates and runs slow. `select_dtype()` picks fp16 on compute
  capability < 8.0 and bf16 on Ampere and later.
- **`mamba-ssm` and `causal_conv1d` are optional.** Without them transformers
  uses the eager path, which is correct and slower. Do **not**
  `pip install mamba-ssm` — it is a source build and fails on Colab routinely.
  If speed demands kernels, install a prebuilt wheel matching the exact
  `cu12x`/`torch2.x`/`cxx11abi`/`cp31x` combination, and pin it; Colab bumps
  torch and will silently break an unpinned wheel.
- **Do not reinstall torch.** Colab ships a CUDA-matched build.
- **Sessions die** (~90 min idle, 12 h max), so the notebook checkpoints the
  packed context to Drive before touching the GPU.

## Languages

Python, C, C++ (and CUDA), JavaScript, Go, Java, Rust.

Symbol extraction uses a per-language node-type table plus one generic AST walk
rather than seven tree-sitter query files. This trades precision for breadth on
purpose: the packer needs relatedness ordering, not a sound call graph, and an
untabulated grammar degrades to "no symbols" instead of crashing.

## Tests

```bash
pytest -q     # 31 tests, CPU only, ~0.1s
```

The manifest tests matter most. `test_every_offset_resolves_to_the_correct_source_line`
walks **every character offset of every segment** at three budgets and asserts
the resolved line actually contains that character. If offset resolution is
wrong, every finding points at the wrong line and any eval number is
meaningless.

Three tests are regressions against specific bugs in the pre-rework code:
a parse failure aborting the whole scan, `Parser(module.language())` raising on
tree-sitter ≥ 0.25, and the packer's collapsed selection/ordering pass.

## Honest limitations

- **No evaluation number yet.** Week 4. Until then, no claim about detection
  rate is defensible.
- Public vulnerability benchmarks (OWASP Benchmark, NIST Juliet) are
  single-file, single-function, so they do not test the cross-file claim at all.
  The plan is a purpose-built benchmark of injected cross-file patterns with
  recorded ground truth, described honestly, over ~15–25 repos.
- The risk score is a **heuristic for budget allocation**, not a vulnerability
  judgement. It decides what the model gets to read; it does not decide what is
  a bug.
- Call-graph edges are name-based and only created when a symbol is defined in
  exactly one file. Ambiguous names carry no information and are dropped, so
  recall on dynamically dispatched code is limited by construction.
- Repos larger than the budget lose files entirely. `wca pack` reports
  `omitted_paths` — read it before believing a clean result.
