# Getting started — what was built, and how to pick it up

Two parts: **what changed and why** (so you can defend it in an interview), and
**what to run, in order** (so you can get moving today).

---

# Part 1 — What was built

## The problem with the old code

Your repo was 5 loose files: `run_audit.py`, `tree_parser.py`,
`language_dispatcher.py`, `notes.txt`, `requirements.txt`. Three things were
wrong, and only two of them were in your handoff spec.

**1. The model couldn't follow instructions.** `state-spaces/mamba-2.8b-hf` is a
base language model — it completes text, it doesn't take orders. Your prompt
ended with:

```python
prompt = f"""...
1. [CRITICAL] Hardcoded Secret:"""
```

The model was continuing that sentence. Feed it an empty file and it still
writes a confident vulnerability report, because "1. [CRITICAL] Hardcoded
Secret:" is a strong prefix and text continuation is all it does. Compounding
it, `do_sample=True, temperature=0.1` adds randomness to what should be a
factual task.

**2. Tree-sitter was decorative.** `tree_parser.py` built a full AST per file
and then kept three fields off it:

```python
file_data = {"root_type": root.type, "statements": root.child_count,
             "has_error": has_error, "content": code_content}
```

Then `build_mamba_prompt()` concatenated raw `file["content"]`. The AST was
thrown away. So "converting repositories into structured context streams" — the
claim on your resume — wasn't happening. It was string concatenation.

**3. (Not in your spec) The parser wouldn't even load.** On tree-sitter ≥ 0.25:

```python
language_obj = language_module.language()   # now returns a PyCapsule
parser = Parser(language_obj)               # TypeError
```

`language()` used to return a `Language`; it now returns a `PyCapsule` that must
be wrapped: `Parser(Language(capsule))`. Your dispatcher would fail on a fresh
install for *every* language. I found this by installing the grammars and
running it before writing anything.

Plus the smaller bugs your spec listed: `tree` shadowed (git tree vs. parse
tree), `except: return` inside the file loop killing the whole scan on one bad
file, `.rs`/`.java` in the language map but not in requirements, and ~500
authenticated API calls with `time.sleep(0.05)` between them.

## The architecture now

Six stages, each a module, each independently testable:

```
ingest.py    one tarball fetch or a local dir   ->  file list + bytes
parse.py     Tree-sitter AST, 7 languages       ->  retained trees
graph.py     imports/defs/calls/literals        ->  cross-file edges
pack.py      budget packer + offset manifest    ->  the context stream
infer.py     instruction-tuned Mamba, 4-bit     ->  raw model output
findings.py  JSON schema + location resolution  ->  grounded findings
```

Stages 1–4 run on CPU with no torch installed. That's deliberate: it makes the
packer testable in 0.1s without a GPU, and makes the Colab notebook a thin
driver instead of the program.

### ingest.py — one request instead of 500

GitHub serves any repo as a single gzip stream at
`codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{ref}` — no token needed
for public repos. The tarball is streamed and filtered in memory: vendor dirs,
lockfiles, minified bundles, generated protobufs, binaries, and anything over
200KB are dropped before they ever reach the parser.

On a 500-file repo that's ~500 authenticated calls and 25 seconds of pure
`sleep()` reduced to one unauthenticated request.

### parse.py — the AST is kept

Same `LanguageDispatcher` idea as yours (it was a good design — lazy loading,
cached parsers), with the `PyCapsule` fix, a missing grammar degrading to
"unsupported" instead of raising, and the `except` recording the error and
**continuing** instead of returning. `ParsedFile` now carries the actual `tree`.

### graph.py — where the cross-file claim becomes real

One DFS per file, dispatching on a per-language node-type table, extracting four
things: imports, definitions (name + byte span + line span), call sites, and
string literals. Then files get linked three ways:

| Edge | Meaning |
|---|---|
| `import` | A imports a module that resolves to B (handles dotted paths, `::`, and relative `../lib/x.js`) |
| `call` | A calls a symbol defined in **exactly one** other file, B |
| `literal` | A and B share a distinctive string — the classic "secret here, used there" |

The `call` rule matters: if a name is defined in five files, it carries no
information, so no edge is created. That's a deliberate precision-over-recall
choice, and there's a test for it.

I used a node-type table rather than seven tree-sitter query files. Trade-off:
less precise, far less surface area, and an untabulated grammar degrades to "no
symbols" instead of crashing. The packer needs relatedness ordering, not a sound
call graph.

**Verified on a planted example** — a repo where `lib/config.py` holds a
password, `app/handlers.py` logs it, and `lib/db.py` has an unparameterized
`cur.execute()`:

```
import   app/handlers.py -> lib/config.py   (lib.config)
call     app/handlers.py -> lib/config.py   (get_conn_string)
call     app/handlers.py -> lib/db.py       (execute_raw)
import   app/util.js     -> lib/helper.js   (../lib/helper.js)
```

That's the action-at-a-distance chain, recovered structurally.

### pack.py — the actual contribution

The key insight, and the thing to say in an interview: **linear scaling is a
claim about asymptotics, not about free VRAM.** A 200k-token Mamba prefill still
costs real memory and minutes on a T4, and Falcon3-Mamba was trained at 32k.
So the context is a budget to *allocate*, not a bucket to fill.

Three jobs:

**Order** — group files by connected component, order within a component by
dependency depth (Kahn's algorithm over import edges), so related code sits
adjacent. An SSM compresses history into a fixed-size state, so distance between
two related facts directly costs recall. A transformer can attend across 100k
tokens at quadratic cost; an SSM can't — *layout is the mitigation*.

**Allocate** — decide globally, by priority, which files get full bodies, which
get a signature skeleton (imports + declarations, bodies elided), which get
dropped. Priority is a risk heuristic (secret-shaped literals, dangerous sinks)
propagated one hop along graph edges, because a file that *imports* a credential
is as interesting as the one defining it — that pair **is** the vulnerability.

**Ground** — every emitted region records stream-offset → `(file, line)`. Without
this the model's output can't be tied to real locations, and a finding without a
location is unusable.

### The bug I hit here, and why it's worth telling

My first packer emitted greedily in stream order: full bodies until the budget
ran out, then signatures with nothing left. On 5 files it looked perfect. On 400
files:

```
183 full / 0 signature / 217 omitted     <- the signature tier never fired
secret-bearing files kept at 4k budget: 1 of 11
```

Both failures, same root cause: selection and ordering were the same pass, so a
low-priority file early in the dependency order ate budget a high-risk file
later needed. Splitting them — select globally by priority, *then* emit in graph
order — fixed both:

| Budget | Used | Full | Signature | Omitted | Secret files kept |
|---|---|---|---|---|---|
| 8k  | 5,994  | 33  | 16 | 351 | 11/11 |
| 16k | 13,935 | 78  | 39 | 283 | 11/11 |
| 24k | 21,977 | 127 | 59 | 214 | 11/11 |
| 32k | 29,973 | 174 | 78 | 148 | 11/11 |

There's a second rule in there worth knowing: leftover budget only upgrades
signatures to full bodies **once every file is already in the stream**. Breadth
before depth — promoting while files are still omitted trades away the coverage
the cross-file claim depends on.

### infer.py — model swap and two hardware traps

`tiiuae/Falcon3-Mamba-7B-Instruct`: pure Mamba-1, 64 decoder blocks, no
attention anywhere (so O(L) survives), instruction-tuned with a chat template,
32k trained context. Decoding is now greedy — sampling on a factual task
manufactures findings.

Two things that bite on Colab:

- **T4 is Turing and has no real bf16.** Your `bnb_4bit_compute_dtype=bfloat16`
  silently emulates and runs slow. `select_dtype()` reads compute capability and
  picks fp16 below sm_80, bf16 at or above.
- **`mamba-ssm` is optional.** Without it transformers uses the eager path, which
  is *correct*, just slower. Don't `pip install mamba-ssm` — source build, fails
  on Colab routinely.

### findings.py — the honesty mechanism

Parses JSON out of model output (handles fences, surrounding prose, and salvages
truncated arrays), then:

- strips file paths that were never in the stream — the model invented them
- looks up the `evidence` line in the packed text and resolves it to `file:line`
- marks anything it can't locate as `grounded=False`

An ungrounded finding is a hallucination by construction. They're reported
separately and kept out of the precision number. This is what makes the week-4
evaluation defensible instead of self-graded.

## What was verified

31 tests, ruff clean, ingest→pack in **0.12s** on 400 files.

The one that matters most:
`test_every_offset_resolves_to_the_correct_source_line` walks **every character
offset of every segment** at three budgets and asserts the resolved line
actually contains that character — 40,527 offsets, zero mismatches. If offset
resolution is wrong, every finding points at the wrong line and any eval number
is meaningless.

Three are regressions against real bugs: the scan-aborting `except`, the
`PyCapsule` parser failure, and the collapsed packer pass.

**Not yet verified:** anything requiring a GPU. `infer.py` and `findings.py` are
written and unit-tested against synthetic model output, but no real model has
run. That's your week 2.

---

# Part 2 — How to follow this

## Step 0 — Get the code into your repo (15 min)

The files are in your Cowork folder, not in git yet. Your GitHub repo still has
the old 5 files.

```bash
cd ~/Documents/"Cowork Playground"/whole_codebase_auditor
git init
git remote add origin https://github.com/quinyang/whole_codebase_auditor.git
git fetch origin
```

The old files don't exist in the new layout, so this is a replacement, not a
merge — but the old commits are worth keeping ("here's what it was, here's what I
changed and why" is a good interview artifact).

**Do not use `git checkout -b rework origin/main`.** Checkout tries to write the
old tree into your directory and aborts the moment an old file collides with a
new one — the old repo has a `.gitignore` and so does this one:

```
error: The following untracked working tree files would be overwritten by checkout:
        .gitignore
```

Use `git reset` instead. It moves HEAD and the index to the old commit but
**never touches your working files**, so the new code stays exactly as it is and
git simply sees it as a large diff against the old tree:

```bash
git reset origin/main    # HEAD + index = old tree; your files untouched
git branch -m rework     # name the branch
git add -A               # stage: new files added, old files recorded as deleted
git status               # verify before committing (see below)
```

`git status --short` should look like this — `A` for new files, `D` for the five
old ones, `M` for the `.gitignore` that exists in both:

```
M  .gitignore
A  CODE_MAP.md
A  GETTING_STARTED.md
A  PLAN.md
A  README.md
A  pyproject.toml
A  wca/pack.py            (and the rest of wca/, tests/, eval/, notebooks/)
D  language_dispatcher.py
D  notes.txt
D  requirements.txt
D  run_audit.py
D  tree_parser.py
```

If you see `D` next to anything you wanted to keep, stop and re-check. Otherwise:

```bash
git commit -m "Rework: 6-stage package, symbol graph, budget packer, 31 tests"
git push -u origin rework
```

`git log --oneline` should now show your rework commit sitting on top of the old
history. Then either open a PR to `main`, or promote it directly:

```bash
git push origin rework:main
```

## Step 1 — Run it locally, no GPU (10 min)

```bash
pip install -e ".[dev]"
pytest -q                       # expect: 31 passed
```

Now scan something real:

```bash
wca graph pallets/flask         # symbol graph + top-risk files
wca pack  pallets/flask         # builds context.pack.txt + manifest
```

**Open `context.pack.txt` and read it.** This is the single most useful thing
you can do today — it's exactly what the model will see. Check: are related
files adjacent? Did the signature tier elide sensibly? Did anything important
land in `omitted_paths` in the manifest?

Try your own projects too — `wca pack ./some-local-dir --budget 16000`. If the
ordering looks wrong on a real codebase, that's a `graph.py` improvement worth
making before you spend GPU time.

## Step 2 — First GPU run (Colab, ~1 hour)

Open `notebooks/colab_driver.ipynb` in Colab. **Runtime → Change runtime type →
T4 GPU.** Then run cells in order:

| Cell | What it does | Watch for |
|---|---|---|
| 1 | `pip install "wca[gpu] @ git+..."` | must point at the branch you pushed |
| 2 | `describe_environment()` | should say **float16** on a T4, and `mamba_ssm: absent` |
| 3 | Mount Drive | so a dying session doesn't lose the packed context |
| 4 | ingest → parse → graph → pack | CPU only; should take seconds |
| 5 | Load model + generate | first run downloads ~7GB; prefill takes minutes |

Start with a **small repo and a 4k budget**, not Flask at 24k. Confirm the whole
path works end-to-end, then scale up. Cell 5 also re-packs with the real
tokenizer, so the token count becomes exact rather than estimated.

## Step 3 — The question that decides week 3

The largest unknown in this project: **does Falcon3-Mamba emit valid JSON at 20k+
context?** SSM state degrades over long contexts, and instruction-following at
24k is an open question, not a given.

Test it early. Run cell 5 at 4k, 8k, 16k, 24k and check whether
`parse_findings` returns anything at each. If schema compliance collapses at
long context, the fallbacks in priority order are:

1. shorten the JSON schema
2. move instructions *after* the code (recency helps an SSM)
3. switch to a line-oriented output format and adjust `findings.py`

Only after all three fail should you consider a smaller model — losing the 7B
instruct model costs more than losing strict JSON.

Also record, at each budget, the prefill wall-clock. That gives you an
**empirical O(L) curve from your own pipeline**, which is worth more in an
interview than citing the Mamba paper.

## Step 4 — The benchmark (week 4, the resume number)

Read `eval/README.md` before writing any of it. The design decisions that matter:

- **Public benchmarks won't work.** OWASP and Juliet are single-file — they don't
  test the claim at all.
- **Negative controls are mandatory.** Repos with no injected vulnerability.
  Without them precision is unmeasurable and the number is worthless.
- **Vary the distance** between the two halves of each injected vulnerability.
  This directly tests whether graph ordering earns its complexity, and it's the
  most interesting result the project can produce.
- **Only grounded findings count**, and report the ungrounded rate separately.

## What not to do yet

**Don't update the resume.** The Mamba bullets should stay "Ongoing" until step 4
produces a number. The current bullets claim "structured context streams" driving
the model — that's true *now*, but the detection claim still has zero evidence.
`PLAN.md` has the target bullet shapes to fill in.

**Don't install `mamba-ssm`.** Only after correctness is proven and speed is
provably the bottleneck, and then only a pinned prebuilt wheel.

**Don't raise the budget past 32k.** That's the model's trained ceiling; beyond
it you're extrapolating and recall will quietly degrade.

## Where things are

| File | Read it when |
|---|---|
| `README.md` | explaining the project to someone else |
| `PLAN.md` | deciding what to do next — weeks 2–4 with done-criteria |
| `eval/README.md` | before building the benchmark |
| `wca/pack.py` | the docstring explains the core design decision |
| `tests/test_pipeline.py` | you change anything — run `pytest -q` first |
