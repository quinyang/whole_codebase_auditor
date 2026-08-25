# Resume bullets — audit and rewrite

## Use this now

**Whole-Codebase Auditor — Mamba/SSM** *(Python, PyTorch, Tree-sitter, CUDA)* | In progress

- **Pipeline:** Built a 6-stage repository auditor (tarball ingest → Tree-sitter parse → symbol graph → context packer → SSM inference → findings) as an installable CLI/Colab package; 31 tests, sub-second CPU path on 400-file repos.
- **Context Packing:** Designed a token-budget packer that orders files by a cross-file symbol graph (imports, call sites, literals; 7 languages) and emits an offset→`file:line` manifest grounding every model claim to real source, validated over 40K offsets.
- **GPU Profiling:** Found HuggingFace's non-fused Mamba path materializes a `[batch, d_inner, seq_len, d_state]` tensor — activation memory linear in context (~1.5 MiB/token), not the O(1) state the SSM implies; modeled the OOM threshold within 2% and added budget guards.

---

## What was wrong with the original

| Claim | Problem |
|---|---|
| "**enabling detection** of cross-file vulnerabilities" | **Fix this first.** No successful inference run yet — an interviewer reads "detection" and asks for your rate |
| "6 languages (Python, C/C++, JavaScript, Go)" | Lists 5, says 6, code supports **7** (adds Java, Rust) |
| "Mamba-2.8B" | Now Falcon3-Mamba-7B-**Instruct**. The 2.8B was a base LM that couldn't follow instructions |
| "structured context streams" | Was *false* when written — old code discarded the AST. True now |
| "Mega Memory GitHub Auditor" | Repo is `whole_codebase_auditor`; names should match if you link it |
| "achieve O(L) scaling" | True for compute, but you've since *measured* something sharper |

The GPU-profiling bullet is the differentiator. Most new-grad resumes say they
*used* a model; yours says you profiled one, found where the reference
implementation diverges from the paper's complexity claim, and predicted the
failure quantitatively.

---

## After week 4, swap bullet 3 (keep it if you have room)

- **Evaluation:** Built a cross-file vulnerability benchmark (**[N]** repos, scripted injection with ground truth, negative controls) since public benchmarks are single-file; **[X]%** precision / **[Y]%** recall counting only findings grounded to a verified location, hallucination rate reported separately.

---

## Phrasing

| Don't | Do |
|---|---|
| "achieves O(1) state memory" | "linear-time inference; the fused kernel is what recovers constant-state memory" |
| "enabling detection of…" | "grounds findings to…" — describe the mechanism until you have a rate |
| "6 languages" | "7 languages" |
| "Mamba-2.8B" | "instruction-tuned 7B Mamba" |

---

## Three questions to expect

1. **"Why an SSM over a long-context Transformer?"** — O(L) vs O(L²). Then the
   honest counterpoint: an SSM has no random access to history, only a
   fixed-size state, which is *why* the packer orders related files adjacently.
   Layout is the mitigation for the architecture's weakness.
2. **"You say linear scaling — what did you measure?"** — Compute is linear, but
   the HF eager path's *activation memory* is also linear (~1.5 MiB/token)
   because it materializes the recurrence across all timesteps instead of
   scanning in SRAM. Predicted the OOM point within 2%.
3. **"How do you know the findings are real?"** — Every finding must quote an
   evidence line that resolves through the manifest to a real `file:line`; those
   that don't are flagged ungrounded and excluded. Limitation: call edges only
   fire when a symbol is defined in exactly one file, so dynamic dispatch is out
   of reach by construction.
