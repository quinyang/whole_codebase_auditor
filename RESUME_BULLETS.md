# Resume bullets — final (numbers are real)

## Use this

**Whole-Codebase Auditor — Mamba/SSM** *(Python, PyTorch, Tree-sitter, CUDA)*

- **Pipeline:** Built a 6-stage repository auditor (tarball ingest → Tree-sitter parse → cross-file symbol graph → context packer → SSM inference → grounded findings) as an installable CLI/Colab package; 87 tests, sub-second CPU path on 400-file repos.
- **Evaluation:** Built a seeded cross-file vulnerability benchmark (10 repositories, 40 injected defects, negative controls) because public benchmarks are single-file; measured **43.8% precision / 17.5% recall** on grounded findings, and reported a **null result** on the graph-ordering ablation rather than a favorable one.
- **GPU profiling:** Found HuggingFace's non-fused Mamba path materializes a `[batch, d_inner, seq_len, d_state]` tensor — activation memory linear in context (~1.74 MiB/token measured vs 1.5 predicted), not the O(1) state the SSM implies; derived `peak = 4.3 GiB + 1.74 MiB/token`, a ~5,300-token ceiling on a T4, and added budget guards.

---

## Why these are the right three

The second and third bullets are what distinguish this from a typical model-usage
project. Bullet 2 says you built the measurement apparatus and **reported a
negative result** — interviewers read that as someone who won't oversell. Bullet
3 says you profiled an implementation against its paper's complexity claim and
predicted the failure quantitatively.

The recall number is low and that is fine to state plainly. A candidate who
reports 17.5% with a clear methodology is more credible than one reporting 90%
with a benchmark nobody can inspect.

---

## Answers you need ready

**"17.5% recall is low. Why present it?"**
Because it is measured, and the apparatus is the contribution. The benchmark is
seeded and reproducible, has negative controls, and separates detection from
retrieval. A number I can defend beats a number I can't.

**"Why a synthetic corpus?"**
I tried real libraries first and measured why they don't work: a 350-line module
is ~5,600 tokens and the T4 ceiling is ~5,300, so both halves of a cross-file
defect physically cannot be in context. The synthetic corpus is a consequence of
that measurement. On better hardware the same harness runs on real repositories.

**"Your ablation found nothing. Doesn't that undercut the packer?"**
At 4k context on 10-module repositories, yes — I found no effect and I report it.
The argument for ordering (an SSM's fixed-size state should lose distant detail)
predicts an effect at long context on large repositories, which is the regime a
16 GB GPU can't reach. Untested, not disproven, and I'd say so either way.

**"What does grounding actually buy you?"**
Auditability, not accuracy. Every reported finding points at a real file and line
a reviewer can check in seconds. I initially claimed it filtered hallucinations —
on two repositories it looked like a perfect filter — but at n=30 the grounding
rate was 17.9% on planted repos and 20.6% on clean ones. Indistinguishable. I
corrected the claim.

*(That last answer is the strongest thing in this document. It shows you
measured your own claim, found it wrong, and said so.)*

---

## Do not write

| Don't | Why |
|---|---|
| "grounding eliminates false positives" | Measured false: 20.6% grounding on clean repos, all 7 false positives |
| "graph ordering improves recall" | Measured: −5.0%, a null result |
| "evaluated on real repositories" | None were audited |
| "achieves O(1) state memory" | Measured the opposite on the eager path |
| any recall figure above ~18% | 7 of 40 planted, on grounded findings |
