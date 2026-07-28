# Evaluation (week 4)

The number this directory produces is the point of the whole rework, so the
design matters more than the code.

## Why not an existing benchmark

OWASP Benchmark and NIST Juliet are **single-file, single-function**. They do not
test the cross-file claim, which is the entire thesis of this project. Scoring
well on them would be evidence of nothing.

## Design

**Corpus.** 15–25 real repositories, mixed languages, mixed sizes — including
repos larger than the token budget, because that is the honest operating regime.

**Injection.** `inject.py` (to write) takes a clean repo and plants a
cross-file vulnerability with recorded ground truth `(file, line, category)`.
Scripted and seeded, so results reproduce. Patterns to cover:

| Pattern | Shape |
|---|---|
| `credential_leak` | secret defined in module A, logged/transmitted in module B |
| `taint_to_sink` | request input enters at A, reaches an unparameterized sink in B |
| `auth_bypass` | authorization enforced in A, a second entry point in B skips it |
| `unsafe_deser` | payload accepted in A, deserialized without validation in B |

**Distance is a variable, not a constant.** Inject each pattern at both short
and long distance in the packer's emitted order. This is what actually tests
whether graph ordering earns its complexity, and it is the most interesting
result the project can report.

**Negative controls are mandatory.** A meaningful fraction of the corpus must
carry no injected vulnerability. Without them precision cannot be computed and
the headline number is meaningless.

## Scoring

- A finding counts as a true positive only if it is **grounded** (its evidence
  line resolved to a real `file:line`) and lands within ±3 lines of ground truth.
- Ungrounded findings are counted and reported separately, never silently
  dropped — they are the hallucination rate.
- Single-file findings are excluded from the cross-file numbers.

## Ablations

Cheap to run once the harness exists, and they are what turn a score into a
systems result:

1. graph ordering vs. random file order
2. with vs. without the signature tier
3. recall as a function of token budget (4k → 32k)

## Files

- `inject.py` — plant vulnerabilities, emit ground truth (to write)
- `run_eval.py` — run `wca scan` over the corpus, score, report (to write)
- `benchmark/` — corpus manifest + ground truth JSON (to build)
