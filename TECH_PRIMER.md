# Tech primer — every moving part in this project

Written for coming back cold. Each entry says *what it is*, *why this project
uses it*, and where relevant, *which error you hit because of it*.

---

# Part 1 — The model layer

## Language model, tokens, context

A language model predicts the next **token** given previous tokens. A token is a
sub-word chunk — roughly 3–4 characters of code. "Context" is how many tokens the
model can consider at once. `wca` packs an entire repo into that window.

Two distinct phases, with very different costs:

- **Prefill** — the model reads your whole prompt. All 2,000 (or 24,000) tokens
  are processed. This is where your time and memory go.
- **Decode** — the model emits output one token at a time. Comparatively cheap.

When I said "prefill dominates," this is what I meant: reading a repo is the
expensive part, writing 500 tokens of findings is not.

## Transformer vs. State Space Model — the reason this project exists

A **Transformer** uses *attention*: every token compares itself to every other
token. For a sequence of length L that's L² comparisons. Double the context,
quadruple the cost. This is why whole-repo context is priced out on Transformers.

A **State Space Model (SSM)** instead carries a fixed-size **hidden state** that
it updates as it walks the sequence — much like an RNN, but with a mathematical
structure that makes it parallelizable during training. Cost is **O(L)**: double
the context, double the cost. That linear scaling is the entire thesis of this
project.

**Mamba** is a specific SSM design whose state update is *selective* — the
parameters controlling what enters and leaves the state depend on the current
input, so it can choose to remember a variable name and forget whitespace. That
selectivity is what made SSMs competitive with Transformers.

The key trade-off, and worth having ready for an interview: a Transformer can
look directly at any earlier token via attention. An SSM only has what survived
in its fixed-size state. **Information has to survive the walk.** This is exactly
why `pack.py` orders related files adjacently — distance between two related
facts costs recall in a way it wouldn't for a Transformer.

## Falcon3-Mamba-7B-Instruct

- **7B** — 7 billion parameters (learned weights).
- **Mamba-1 architecture**, 64 decoder blocks, hidden width 4096, state size 16.
  No attention layers anywhere, so the O(L) story holds.
- **Instruct** — post-trained to follow instructions. The critical difference
  from your old `mamba-2.8b-hf`, which was a **base model**: base models only
  continue text. Your old prompt ended `"1. [CRITICAL] Hardcoded Secret:"` and
  the model dutifully continued that sentence regardless of the code.
- **32k trained context** — the ceiling. Beyond it you're extrapolating.

## Chat template

Instruct models are trained on a specific markup for turn boundaries, e.g.
`<|system|>...<|user|>...<|assistant|>`. `tokenizer.apply_chat_template()` emits
the exact format that model was trained on. Skipping it and pasting raw text is a
common cause of an instruct model behaving like a base model.

## Greedy decoding vs. sampling

- **Sampling** (`do_sample=True`, `temperature`) picks randomly among likely next
  tokens. Good for creative writing.
- **Greedy** (`do_sample=False`) always takes the most likely token.
  Deterministic, reproducible.

Your old code used `do_sample=True, temperature=0.1`. Auditing is a factual task
— randomness there literally invents vulnerabilities. `infer.py` is greedy.

---

# Part 2 — The GPU layer

This is where every error in this project came from, so it's worth real detail.

## VRAM, and the three things competing for it

Your T4 has **14.56 GiB** of VRAM. Three consumers:

| Consumer | What it is | Scales with |
|---|---|---|
| **Weights** | the 7B learned parameters | model size, quantization |
| **Activations** | intermediate tensors during a forward pass | **context length** |
| **Allocator overhead** | cached/fragmented blocks torch holds | usage pattern |

**Quantization shrinks weights only.** That distinction is the single most
important thing in this section, and it's what made your OOMs confusing.

## Quantization / 4-bit / nf4

Weights are normally fp16 (2 bytes each) → 7B × 2 = ~14 GB, which alone nearly
fills a T4. **4-bit quantization** stores each weight in 4 bits (~0.5 bytes) →
~4 GB, dequantizing on the fly during compute.

- **nf4** ("normal float 4") — a 4-bit format whose levels are spaced to match
  the bell-curve distribution weights actually follow. More accurate than naive
  4-bit at the same size.
- **double quantization** — also quantizes the per-block scaling constants.
  Saves a further ~0.4 GB.
- **compute dtype** — the precision used for actual math after dequantizing.
  Separate from storage precision.
- **bitsandbytes** — the library implementing all of this.

## fp16 vs bf16, and compute capability

Both are 16-bit floats, differing in how they split bits between range and
precision. **bf16** has the same exponent range as fp32, so it rarely overflows,
which makes it the default choice on modern GPUs.

**But bf16 needs hardware support**, which arrived with NVIDIA's Ampere
generation (compute capability 8.0). Your **T4 is Turing (sm_75)** — bf16 gets
emulated, slowly. Hence `select_dtype()` in `infer.py`: fp16 below sm_80, bf16
at or above. Your old code hardcoded `bfloat16`.

"Compute capability" is just NVIDIA's version number for a GPU's feature set.
T4 = 7.5, L4/A100 = 8.x, H100 = 9.0.

## CUDA kernels, HBM vs SRAM — why the fused kernel matters

A **kernel** is a function that runs on the GPU. A **fused** kernel does several
mathematical steps in one launch without writing intermediates back to memory.

GPU memory has two tiers:

- **HBM** — the 14.56 GiB. Large, comparatively slow.
- **SRAM** — a few MB of on-chip scratch. ~10× faster.

Mamba's official `selective_scan` kernel is *hardware-aware*: it runs the
recurrence entirely in SRAM and **never writes the per-timestep intermediates to
HBM**. That is what makes the memory footprint independent of sequence length.

**This is the root cause of your first OOM.** Without that kernel, transformers
falls back to `slow_forward`, which does the same math naively:

```python
discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
# shape [batch, intermediate_size, seq_len, ssm_state_size]
```

Note `seq_len` is a **dimension of the allocated tensor**. It materializes every
timestep at once. Cost:

```
intermediate(8192) × state(16) × 4 bytes(fp32) = 0.5 MiB per token, per tensor
~3 such tensors live                            ≈ 1.5 MiB per token
```

At 22,000 tokens that's a single 10.9 GiB allocation — which is exactly the
number in your traceback, within 2% of the prediction.

**So: the O(1)-state property of an SSM is a claim about the algorithm. The
reference implementation does not deliver it. The fused kernel is what makes it
real.** You measured that. It's the most interesting result the project has
produced so far.

## Why the second and third OOMs were different

The first OOM was architectural (real 22k context, real 10.9 GiB tensor). The
later ones requested only **1.16 GiB** and still failed — because 12–13 GiB was
*already* allocated before generation started. That's roughly two copies of the
model.

**Rule worth keeping: an OOM whose requested allocation is small is never about
the allocation.** It's about what was already resident. Read the "already
allocated" number first, always.

## Why a second copy existed

In Python, `auditor = MambaAuditor()` rebinds a name. The *old* model object
isn't freed until garbage collection runs — and even then, torch's **caching
allocator** keeps the freed VRAM blocks reserved for reuse rather than returning
them. So re-running a load cell stacks models.

Hence `free_gpu()` in `infer.py`: `gc.collect()` then `torch.cuda.empty_cache()`.
The real fix is restarting the runtime, which guarantees a clean process.

---

# Part 3 — The parsing layer

## Tree-sitter, ASTs, grammars

An **AST** (abstract syntax tree) is source code as a tree: a function
definition node with a name child and a body child, and so on. It knows
structure, where a regex only sees characters.

**Tree-sitter** is a parser generator with a separate **grammar** per language,
shipped as its own pip package (`tree-sitter-python`, `tree-sitter-go`, …). This
project uses 7 grammars across 18 file extensions.

Node types are grammar-specific — Python calls a function definition
`function_definition`, Go calls it `function_declaration`. `graph.py` keeps
per-language tables mapping those names onto four categories (definitions,
imports, calls, string literals) so one generic tree walk handles every language.

**The PyCapsule bug you'd have hit:** in tree-sitter ≥ 0.25, `module.language()`
returns a raw C pointer wrapper (a `PyCapsule`) rather than a `Language` object,
so `Parser(module.language())` raises. It must be `Parser(Language(capsule))`.
Your old dispatcher would have failed on a fresh install for every language.

## Symbol graph, topological sort

A **graph** is nodes plus edges. Here nodes are files; edges mean "imports",
"calls a symbol defined in", or "shares a distinctive string with".

- **Connected component** — a cluster of files reachable from one another. Used
  to keep related code together in the stream.
- **Topological sort** — ordering a directed graph so dependencies come before
  dependents. `_dependency_order` uses **Kahn's algorithm**: repeatedly emit any
  node with no unmet dependencies. Cycles (mutual imports) can't be ordered, so
  they fall back to a degree heuristic.

---

# Part 4 — Libraries and packaging

## Hugging Face stack

| Piece | Role |
|---|---|
| **transformers** | model architectures + `from_pretrained` loading |
| **Hub** | where weights are downloaded from; `HF_TOKEN` raises rate limits |
| **safetensors** | weight file format; safe and fast (the old `.bin` was pickle, i.e. arbitrary code) |
| **accelerate** | `device_map="auto"` — places layers across GPU/CPU/disk |
| **bitsandbytes** | the 4-bit quantization implementation |
| **kernels** | HF's on-demand prebuilt CUDA kernel loader — the no-source-build route to the fast path |

## Python packaging

- **pyproject.toml** — declares the package: name, version, dependencies,
  entry points. Its absence is why your first Colab install failed (pip cloned
  `main`, which still had the old five loose files).
- **Editable install** (`pip install -e .`) — links to your source directory
  instead of copying, so edits take effect immediately.
- **Extras** (`.[gpu]`, `.[dev]`) — optional dependency groups. This project
  deliberately keeps torch out of the core deps so stages 1–4 run on any CPU box
  and Colab's CUDA-matched torch is never clobbered.
- **PEP 508 direct reference** — `"wca[gpu] @ git+https://…"`. The older
  `#egg=wca[gpu]` form is deprecated and silently drops the extra.
- **Virtual environment** — an isolated set of installed packages. `pip` installs
  into whatever interpreter is active; the `(base)` / `(wca)` prefix in your
  prompt is the only signal of which one that is.

## pytest and ruff

**pytest** collects functions named `test_*` and reports failures; `@fixture`
supplies shared setup. The suite here is 31 tests, ~0.1s, CPU-only — fast enough
to run on every change.

**ruff** is a linter and formatter: unused imports, bad import order, suspicious
patterns. Configured in `pyproject.toml`, with `BLE001` (broad `except`)
deliberately disabled because degrading rather than aborting on a bad file is the
intended behavior here.

---

# Part 5 — Colab and git

## How Colab actually works, and the trap you hit three times

A Colab notebook runs against a **kernel** — a long-lived Python process. Cells
share state, and the kernel ID appears in tracebacks as `ipykernel_XXXX`.

Two consequences that cost you real time:

1. **`pip install` does not affect an already-imported module.** Python caches
   imports in `sys.modules`. Reinstalling `wca` mid-session leaves the *old* code
   running. **Runtime → Restart session** is the only reliable fix.
2. **A notebook open in Colab is a copy.** Editing `notebooks/colab_driver.ipynb`
   in the repo and pushing does *not* update your open tab. To get repo changes:
   **File → Open notebook → GitHub**, pick the repo and branch. Or paste the new
   cell contents in manually.

Also: sessions die after ~90 min idle / 12 h max, which is why cell 4 checkpoints
the packed context to Drive before any GPU work.

## Git terms used here

- **remote / origin** — the GitHub copy.
- **branch** — a named line of history (`main`, `rework`).
- **`git reset <commit>`** — moves HEAD and the index, **leaves working files
  alone**. This is why it worked where `checkout` failed: checkout wanted to
  overwrite your `.gitignore` with the old one.
- **`git push origin rework:main`** — push local `rework` onto remote `main`.
- **Personal access token / SSH key** — GitHub removed password auth in 2021.

---

# The five errors, in one table

Each one maps to a concept above. Worth re-reading once the pipeline runs.

| Error | Real cause | Concept |
|---|---|---|
| `checkout would overwrite .gitignore` | checkout materializes the target tree | git working tree vs. index |
| `does not appear to be a Python project` | pip cloned `main`, which had no `pyproject.toml` | branches; packaging |
| `Authentication failed` | password auth removed in 2021 | PAT / SSH |
| OOM, 10.92 GiB at 22k tokens | `slow_forward` materializes `seq_len` | fused kernels, HBM vs SRAM |
| OOM, 1.16 GiB at 2k tokens | two models resident; stale session | caching allocator; `sys.modules` |

The last two look identical in the traceback and have completely different
causes. Telling them apart is just reading the *already allocated* figure rather
than the *requested* figure.
