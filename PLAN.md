# WCA rework plan — solidified

Supersedes `mamba_auditor_handoff.md`. Target: a defensible portfolio piece with
a real evaluation number, runnable on Colab and as a CLI, in 4 weeks.

**Week 1 is done.** What follows records what changed against the original spec,
then the remaining three weeks with concrete done-criteria.

---

## Decisions locked in

| Question | Decision | Why |
|---|---|---|
| Cloud target | Google Colab (T4/L4), package + thin notebook driver | Zero infra cost; the package, not the notebook, is the artifact — so a Modal/RunPod port later touches only `infer.py` |
| Model | `tiiuae/Falcon3-Mamba-7B-Instruct` | Pure Mamba-1, no attention → the O(L) story survives; instruction-tuned with a chat template; **32k trained context** |
| Kernels | Eager path first, no `mamba-ssm` | Source build, fails on Colab routinely. Correctness before speed |
| Default budget | 24k tokens (32k hard cap) | Model's trained ceiling is 32k; 24k leaves room for the preamble and output |
| Decoding | Greedy | Auditing is factual; sampling manufactures findings |
| Benchmark | Purpose-built cross-file, injected ground truth | OWASP/Juliet are single-file and do not test the claim |

## Blockers from the handoff spec — resolved

**Blocker 1 — model can't follow instructions.** Swapped to
Falcon3-Mamba-7B-Instruct with a proper chat template and a JSON output schema.
The trailing-colon completion trick is gone; so is `do_sample=True`.

**Blocker 2 — Tree-sitter was decorative.** `parse.py` now retains the tree.
`graph.py` walks it to extract imports, definitions, call sites, and string
literals, and links files by import resolution, unambiguous call targets, and
shared distinctive literals. `pack.py` consumes the graph for both ordering and
priority. The AST is now load-bearing at both stages.

**Smaller bugs — all fixed**, each with a regression test where it's testable:

- Shadowed `tree` (git tree vs. parse tree) — gone with the tarball rewrite.
- `except: return` inside the file loop — now records and continues.
  → `test_bad_syntax_does_not_abort_scan`
- `.rs`/`.java` in the language map but not in requirements — all seven grammars
  are now declared dependencies, and a missing grammar degrades to "unsupported"
  instead of raising. → `test_unsupported_extension_is_skipped_not_raised`
- Per-blob API fetching with `time.sleep(0.05)` — replaced by one tarball
  request. ~500 authenticated calls → 1 unauthenticated call on a 500-file repo.

### Two problems the handoff spec did not anticipate

**`Parser(module.language())` raises on tree-sitter ≥ 0.25.** `language()` now
returns a `PyCapsule`, which must be wrapped in `Language(...)`. The old
dispatcher would have failed on a fresh install for every language — a silent
blocker sitting underneath the two known ones.
→ `test_dispatcher_loads_grammars_on_modern_tree_sitter`

**Greedy single-pass packing never demotes.** The first packer implementation
emitted full bodies in stream order until the budget ran out, then tried
signatures with nothing left. At 400 files that produced 183 full / **0
signature** / 217 omitted, and only 1 of 11 secret-bearing files survived a tight
budget. Splitting selection (by priority, global) from emission (by graph order)
fixed it: every secret-bearing file now survives at every budget tested, and the
signature tier is reached. → `test_signature_tier_is_reached_on_a_large_repo`

---

## Week 1 — restructure, ingest, bug fixes ✅

Package layout, tarball ingest, fixed parser, symbol graph, budget packer with
offset manifest, CLI, Colab driver, 31 tests.

*Done when:* pip-install in Colab and scan without editing logic in a cell. ✅

Measured (400-file synthetic repo, CPU, `wca pack`):

| Budget | Used | Full | Signature | Omitted | Secret files kept |
|---|---|---|---|---|---|
| 8k  | 5,994  | 33  | 16 | 351 | 11/11 |
| 16k | 13,935 | 78  | 39 | 283 | 11/11 |
| 24k | 21,977 | 127 | 59 | 214 | 11/11 |
| 32k | 29,973 | 174 | 78 | 148 | 11/11 |

Ingest→pack on 400 files: **0.12s**. Budget is never exceeded. 40,527 stream
offsets verified to resolve to the correct source line, zero mismatches.

---

## Week 2 — first real GPU run + packer measurement

The packer exists but has only been measured against synthetic repos with a
chars-per-token estimate. This week makes the numbers real.

1. Run the Colab notebook end-to-end against 3–5 real repos of increasing size.
2. Re-pack with the actual tokenizer and **calibrate `CHARS_PER_TOKEN`** — it is
   currently a 3.5 guess. Measure per language.
3. Record prefill wall-clock vs. context length at 4k/8k/16k/24k/32k on a T4.
   This is the plot that substantiates "linear-time" — an *empirical* O(L) curve
   from your own pipeline is worth more in an interview than the claim.
4. Find the largest budget that completes inside a Colab session, and set the
   default from evidence rather than from the model card.

*Done when:* a prefill-time-vs-context-length curve exists, and the default
budget is justified by measurement.

### Finding (first GPU run): the eager path is linear in *memory*, not just time

A 24k-token audit OOM'd on a T4 trying to allocate 10.92 GiB in a single tensor.
The cause is architectural, not a config mistake:

```python
# FalconMambaMixer.slow_forward
discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
# shape [batch, intermediate_size, seq_len, ssm_state_size], fp32
```

Without a fused kernel, transformers **materialises the discretised recurrence
for every timestep at once** rather than scanning it. So activation memory is
*linear in context length*:

```
intermediate(8192) × state(16) × 4 bytes = 0.5 MiB per token per tensor
~3 live tensors                          ≈ 1.5 MiB per token
```

Predicted 10.74 GiB at 21,986 tokens vs. 10.92 GiB observed — a 2% match, so the
mechanism is confirmed. A 24k context wants ~36 GiB of activations and **cannot
fit on a 16 GB card at any quantisation**, because this is activation memory, not
weights. 4-bit does nothing for it.

**This sharpens the project's thesis rather than damaging it.** The O(1)-state
property of an SSM is a claim about the *algorithm*; the reference implementation
doesn't deliver it. The fused `selective_scan` kernel is what makes it real — it
runs the scan in SRAM and never allocates the `seq_len` dimension. That is a
measured, defensible systems result, and a better week-2 deliverable than the
prefill-time curve alone.

Guardrails added to `infer.py`: `slow_path_bytes_per_token()`,
`estimate_max_context()`, and a fail-fast check in `generate()` so this surfaces
before a 3-minute model load instead of after.

**Revised week-2 order:**

1. Budget 4,000, eager path — prove the pipeline end-to-end today.
2. `pip install kernels` (HF's on-demand prebuilt FalconMamba kernels — this is
   what the transformers warning recommends, and it avoids the `mamba-ssm`
   source build entirely). Verify with `fast_path_available()`.
3. If the kernel loads, re-measure the ceiling — it should jump by roughly an
   order of magnitude, and *that delta is the headline measurement*.
4. If it doesn't support sm_75 (T4 is Turing; the fused kernels commonly target
   sm_80+), either move to an L4 runtime or report the T4 ceiling honestly and
   run the budget sweep within it.

Plot **both** curves — memory vs. context length, with and without kernels. Two
lines, one flat-ish and one steep, is the single best artifact this project can
produce for an interview.

## Week 3 — output quality

Inference and findings are written but never exercised against a real model.

1. Does Falcon3-Mamba-7B-Instruct actually emit valid JSON at 20k+ context? SSM
   state degrades over long contexts; instruction-following at 24k is an open
   question, not a given. **Test this early in the week** — it is the largest
   remaining unknown in the project.
2. If schema compliance is poor: shorten the schema, move the instructions
   *after* the code (recency helps an SSM), or fall back to a line-oriented
   format that `findings.py` parses instead.
3. Measure the grounding rate — what fraction of findings carry an evidence line
   that actually exists in the stream. A low rate is itself a legitimate result
   to report.
4. Hand-check findings on the 5 repos from week 2. Are they real? Are they
   genuinely cross-file, or single-file findings dressed up?

*Done when:* `wca scan <repo>` prints findings with real `file:line`, and you
know the grounding rate.

**Risk:** the model may not follow a JSON schema reliably at long context. Fall
back to a simpler output format before falling back to a smaller model — losing
the 7B instruct model costs more than losing strict JSON.

## Week 4 — benchmark, eval, docs

This is where the resume number comes from, so it needs the most care.

1. **Build the benchmark.** 15–25 real repos. For each, inject a known
   cross-file vulnerability with recorded ground truth `(file, line, category)`.
   Injection must be scripted and reproducible — `eval/inject.py`, seeded — or
   the number isn't defensible.
   - Vary the pattern: credential defined/used across files, taint source →
     cross-module sink, auth check bypassed by a second path.
   - Vary the *distance*: files adjacent in the packer's order vs. far apart.
     This directly tests whether the ordering strategy does anything, and is the
     single most interesting result the project can produce.
2. **Include negative controls.** Repos with no injected vulnerability. Without
   them precision is unmeasurable and the number is worthless.
3. **Report** precision / recall over N repos, counting only *grounded*
   findings, and report the ungrounded rate separately.
4. **Ablations** (cheap, and they are what makes it a systems result):
   - graph ordering vs. random file order
   - with vs. without the signature tier
   - budget sweep: recall as a function of token budget

*Done when:* there is a number you would defend under questioning, and a README
limitations section that states what the benchmark does not show.

---

## Resume bullets — rewrite once real

Do not write these until week 4 produces the numbers. Shape to aim for:

- Built a whole-codebase security auditor on an instruction-tuned Mamba SSM,
  ingesting entire repositories in a single linear-time context pass;
  demonstrated O(L) prefill scaling empirically from 4k to 32k tokens.
- Designed a Tree-sitter symbol graph (imports, definitions, call sites across 7
  languages) driving a budget-aware context packer that fits cross-file
  dependencies into a fixed window, with an offset→location manifest that grounds
  every model finding to a real `file:line`.
- Evaluated on a purpose-built cross-file vulnerability benchmark: **[X]%**
  precision / **[Y]%** recall over **[N]** repos; graph-ordered packing improved
  recall **[Z]** points over random file ordering.

Keep them honest. An interviewer will ask how the benchmark was built, what the
negative controls were, and what the ungrounded rate was — and the honest answer
to all three is the strongest part of the project.
