# Status — where the project stands

## Blocked on one thing right now

Your local work is **not on GitHub**. Verified against the remote:

| Branch | State |
|---|---|
| `main` | still the old 5 loose files — no `pyproject.toml` |
| `rework` | `version = "0.2.0"` — the v0.3.0 commit never pushed |
| local | v0.3.0 with `free_gpu`, footprint reporting, memory guards |

This is why Colab kept running old code. A restart couldn't help — the package
being installed was old at the source. Fix first, before anything else:

```bash
cd ~/Documents/"Cowork Playground"/whole_codebase_auditor
git status                       # see what's uncommitted
git add -A
git commit -m "v0.3.0: free_gpu, VRAM footprint report, memory guards, docs"
git push origin rework
git push origin rework:main      # promote to main
```

Verify it actually landed — this takes five seconds and would have saved an
afternoon:

```bash
git ls-remote --heads origin     # main and rework should show the SAME sha
```

Then in Colab: **Runtime → Restart session**, run cell 1, and confirm it prints
`0.3.0`. If it prints `0.2.0`, the push didn't land.

---

## What's done

**The rework of stages 1–4 is complete and verified.** Starting from 5 loose
files with 3 blocking bugs, the project is now a 6-stage installable package.

| Stage | Status |
|---|---|
| 1. `ingest.py` | Done — one tarball request replaces ~500 API calls |
| 2. `parse.py` | Done — AST retained; 7 languages, 18 extensions |
| 3. `graph.py` | Done — imports / call / literal edges across files |
| 4. `pack.py` | Done — budget packer + offset→`file:line` manifest |
| 5. `infer.py` | Written, **never successfully run** |
| 6. `findings.py` | Written, unit-tested against synthetic output only |

Measured on a 400-file repo: ingest→pack in **0.12s**, budget never exceeded,
all 11 secret-bearing files retained at every budget from 4k to 32k. **40,527
stream offsets** verified to resolve to the correct source line, zero
mismatches. 31 tests, ruff clean.

### Bugs found and fixed

Two from your handoff spec: the base-LM prompt that confabulated findings
regardless of input, and Tree-sitter being decorative (AST built, then
discarded). Plus three the spec didn't have:

- `Parser(module.language())` raises on tree-sitter ≥ 0.25 (`PyCapsule` must be
  wrapped in `Language`) — your dispatcher would have failed on a fresh install
  for every language.
- The packer's first implementation collapsed selection and ordering into one
  greedy pass. Looked perfect on 5 files; at 400 files produced **0 signature
  files** and kept 1 of 11 secret-bearing files. Split into select-by-priority
  then emit-by-graph-order.
- **The eager-path memory finding** (below), which is now the most interesting
  result the project has.

### The memory finding

A 24k audit OOM'd trying to allocate 10.92 GiB in one tensor. Cause:

```python
discrete_A = torch.exp(A[None,:,None,:] * dt[:,:,:,None])
# [batch, intermediate_size, seq_len, ssm_state_size]
```

Without a fused kernel, transformers materializes the recurrence for **every
timestep at once**. Activation memory is linear in context length
(~1.5 MiB/token), not the O(1) state the SSM formulation implies. Predicted
10.74 GiB vs 10.92 observed — a 2% match.

Weight quantization does nothing for this; it's activation memory. The fused
`selective_scan` kernel is what makes the O(1) claim real, by keeping the scan
in SRAM.

**This sharpens the thesis rather than damaging it**, and it's the strongest
interview material in the project.

## What's not done

- No successful inference run. Zero findings produced so far.
- No evaluation, no precision/recall, no benchmark.
- Unknown: whether the model emits valid JSON at long context.

---

## Next three sessions

### Session 1 — first successful audit (~1 hour)

1. Push v0.3.0, verify remote SHAs match (above).
2. Colab: restart, cell 1 must print `0.3.0`.
3. Run at `budget_tokens=4000` on a **small** repo.
4. Check the printed footprint line: `allocated ~6 GiB` = 4-bit worked. Over
   8 GiB = stale model, restart.

*Done when:* `report.pretty()` prints anything at all — even zero findings. The
goal is proving the path end-to-end, not finding a vulnerability.

### Session 2 — the JSON question (~2 hours)

The largest remaining unknown: **does Falcon3-Mamba emit valid JSON at long
context?** SSM state degrades over long sequences and instruction-following at
20k+ is not a given.

Run at 4k / 8k / 16k, print `gen.text` raw each time, and record whether
`parse_findings` returns anything. If compliance collapses: shorten the schema,
then move instructions *after* the code (recency helps an SSM), then fall back
to a line-oriented format. Try `pip install kernels` here too — it's the
no-source-build route to the fast path.

*Done when:* you know the largest context that still produces parseable output,
and the grounding rate at that context.

*Also record:* peak memory vs. context length, with and without kernels. Two
lines on one plot is the best artifact this project can produce.

### Session 3 — benchmark design (~3 hours)

Read `eval/README.md` first. Non-negotiables: **negative controls** (repos with
no injected vulnerability — without them precision is unmeasurable), **scripted
seeded injection** so results reproduce, and **varied distance** between the two
halves of each planted vulnerability, which is what actually tests whether graph
ordering earns its complexity.

*Done when:* a number you'd defend under questioning.

---

## Guardrails

- **Don't update the resume Mamba bullets** until session 3 produces numbers.
  `RESUME_BULLETS.md` has an accurate interim version — the key change is
  dropping "enabling detection," which you can't yet back.
- **Don't `pip install mamba-ssm`.** Source build, fails on Colab. Use `kernels`.
- **Don't raise the budget past 32k** — the model's trained ceiling.
- **Read the "already allocated" number in an OOM**, not "tried to allocate."
  That distinction separated the real finding from two rounds of stale-session
  noise.

## Docs

| File | Read when |
|---|---|
| `STATUS.md` | this file — where things stand |
| `CODE_MAP.md` | re-orienting on the code |
| `TECH_PRIMER.md` | any term is unfamiliar |
| `PLAN.md` | weeks 2–4 with done-criteria |
| `GETTING_STARTED.md` | environment / git setup |
| `eval/README.md` | before building the benchmark |
| `RESUME_BULLETS.md` | before touching the resume |
