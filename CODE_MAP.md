# Code map — how each piece works and how they connect

A re-onboarding reference. Read the mental model, then the worked trace at the
bottom; the per-module tables are for looking things up later.

---

## The mental model in one paragraph

A repo goes in, a list of grounded security findings comes out. In between, the
repo is turned into **one long string** that fits in a fixed token budget, and a
**side table** that says which part of that string came from which file and line.
The string is what the model reads. The side table is how you turn whatever the
model says back into a real `file:line`. Everything else — the AST, the symbol
graph, the priority scoring — exists to decide *what goes in the string and in
what order*, because the string is smaller than the repo.

## The data that flows between stages

Each stage consumes the previous stage's object and produces one new type. That's
the whole contract:

```
  str (repo spec)
      │  ingest()
      ▼
  RepoBundle ──── .files: list[SourceFile]      path, data(bytes), text
      │  parse_files()
      ▼
  ParseResult ─── .files: list[ParsedFile]      + lang, tree (the AST)
      │  build_graph()
      ▼
  SymbolGraph ─── .symbols: dict[path→FileSymbols]   defs/imports/calls/literals
      │           .edges:   list[Edge]                cross-file links
      │  pack()
      ▼
  PackedContext ─ .text:     str                 ← what the model reads
      │           .segments: list[Segment]       ← the offset→line side table
      │  MambaAuditor.generate()
      ▼
  GenerationResult .text: str (raw model output)
      │  parse_findings(text, packed)
      ▼
  list[Finding] ── grounded to real file:line
```

Two objects survive to the very end and get used together: **`PackedContext`**
and the model's **raw text**. `parse_findings` needs both — the text to know what
was claimed, the `PackedContext` to check whether the claim points at anything
real.

---

## Stage 1 — `ingest.py`: repo → bytes

Gets files, drops junk. No parsing, no cleverness.

| Function | What it does |
|---|---|
| `ingest(target, ref)` | **Entry point.** If `target` is a local dir → `from_local`, else → `from_github` |
| `from_github(spec, ref)` | Downloads the whole repo as one gzip from `codeload.github.com`, extracts in memory |
| `from_local(root)` | `os.walk` a directory, same filtering |
| `parse_repo_spec(spec)` | `"owner/repo"`, a URL, or `git@...` → `("owner", "repo")` |
| `_ref_candidates(ref)` | Tries `refs/heads/X`, then `refs/tags/X`, then raw SHA — so `--ref` works for branches *and* tags |
| `_bundle_from_tar(blob,…)` | Walks tar members, strips GitHub's `repo-<sha>/` prefix, applies filters |
| `_should_skip(path,size,…)` | Returns a **reason string** or `None`. Drops vendor dirs, lockfiles, minified JS, generated protobufs, files > 200KB |
| `_decode(data)` | UTF-8 → latin-1 → `None` if binary (checks for null bytes first) |

**Types:** `SourceFile(path, data, text)` and `RepoBundle(name, ref, files, skipped)`.
`skipped` is a reason→count dict, which is why `summary()` can tell you *why*
files vanished.

> The one design point: one HTTP request instead of one-per-file. Your old code
> made ~500 authenticated calls with `sleep(0.05)` between them.

## Stage 2 — `parse.py`: bytes → ASTs

| Function | What it does |
|---|---|
| `parse_files(files, dispatcher)` | **Entry point.** Loops files, parses each, returns `ParseResult`. One bad file never aborts the run |
| `LanguageDispatcher.get_parser_for_file(fn)` | Extension → cached `Parser`. Returns `(None, None)` for unsupported |
| `LanguageDispatcher._load(mod, lang)` | Imports the grammar, wraps the `PyCapsule` in `Language(...)`, caches the parser |
| `LanguageDispatcher.language_for_file(fn)` | Extension → `"python"` / `"go"` / … without loading anything |

**Types:** `ParsedFile(path, lang, source, text, tree, has_error)` — note it keeps
`tree`, the real AST. `ParseResult` splits results into `files` / `unsupported` /
`failed`, so nothing disappears silently.

`LANGUAGE_MAP` maps 18 extensions onto 7 grammars (`.cu` → C++, `.jsx` → JS, etc.).

## Stage 3 — `graph.py`: ASTs → cross-file links

The stage that makes "cross-file" real rather than aspirational.

| Function | What it does |
|---|---|
| `build_graph(parsed)` | **Entry point.** Extracts symbols per file, then links files three ways |
| `extract_symbols(pf)` | **One DFS** over the AST. Dispatches each node type against the per-language tables and fills a `FileSymbols` |
| `_find_name(node, src)` | Definition name. Tries the `name` field, else a bounded DFS for an identifier (C/C++ bury names under nested declarators) |
| `_callee_name(node, src)` | Callee of a call: `a.b.c(x)` → `"c"` (last identifier) |
| `_import_targets(node, src, lang)` | Per-language regex to pull the module string out of an import node |
| `_module_keys(path)` | All names other files might use to refer to this one: `lib/config.py` → `{lib.config, lib/config, config, …}` |
| `_resolve_import(imp, importer, idx)` | Import string → candidate files. Handles dotted (`a.b`), `::`, and relative (`../lib/x.js`, resolved against the importer's dir) |

**The four tables** at the top are the whole language-portability story —
`DEF_NODES`, `IMPORT_NODES`, `CALL_NODES`, `STRING_NODES`, one entry per language.
Add a language by adding four rows.

**The three edge kinds:**

| Kind | Rule |
|---|---|
| `import` | A imports something resolving to B |
| `call` | A calls a name defined in **exactly one** other file. Ambiguous names → no edge (a name in 5 files carries no information) |
| `literal` | A and B share a distinctive string (≥8 chars, in ≤4 files, and secret-shaped or path-like) |

**Types:** `Definition(name, kind, path, start_byte, end_byte, start_line, end_line)`,
`FileSymbols`, `Edge(src, dst, kind, detail)`, `SymbolGraph`.

`FileSymbols.risk_score` = `3 × secrets + 1 × dangerous_sinks`. This is a
**budget-allocation heuristic, not a vulnerability judgement** — it decides what
the model gets to read, not what's a bug.

`SymbolGraph` helpers used downstream: `neighbors(path)`, `degree()`,
`components()` (connected components, largest first).

## Stage 4 — `pack.py`: graph + files → one string + a side table

The core of the project. Two separate decisions that used to be one:

| Function | What it does |
|---|---|
| `pack(parsed, graph, budget_tokens=…)` | **Entry point.** Selects modes, then emits in graph order |
| `order_files(graph)` | **WHERE** things go. Components sorted by peak risk; files within a component by `_dependency_order` |
| `_dependency_order(paths, graph)` | Kahn's topological sort over import edges. Cycles fall back to degree order |
| `compute_priority(graph, hops=1)` | **WHAT** gets in. Risk score propagated one hop outward at half weight, plus `0.25 × degree` |
| `render_signature(pf, graph)` | Skeleton view: import lines + one line per def + secret lines, with `... N lines elided ...` between. Returns text **and** a `line_map` |
| `_render(pf, mode, graph)` | Dispatches full vs. signature, returns `(body, line_map, line_start)` |
| `_make_counter(tokenizer)` | Real tokenizer if given, else `len(text)/3.5`. This is what keeps stage 4 CPU-only |
| `cost(path, mode)` *(inner)* | Token cost of a file in a mode — used during selection, before anything is emitted |

**How `pack` actually runs**, in order:

1. Compute `order` (graph order) and `priority` (risk order) — two different lists.
2. **Selection pass 1:** walk in *priority* order, mark `FULL` while total stays under `full_cap` (70% of usable).
3. **Selection pass 2:** everything else → `SIGNATURE`, while under `usable`. Anything that still doesn't fit is omitted.
4. **Promotion:** leftover budget upgrades signatures to full — **only if nothing was omitted**. Breadth before depth.
5. **Emission:** walk in *graph* order, write each file in its assigned mode, recording a `Segment` for each.

> Steps 2–4 decide *what*, step 5 decides *where*. Collapsing them into one greedy
> pass was the bug: at 400 files it produced 0 signatures and kept 1 of 11
> secret-bearing files.

**Types:** `Segment(path, mode, char_start, char_end, line_start, line_map, est_tokens)`
and `PackedContext`.

**The two methods everything downstream depends on:**

- `PackedContext.resolve(offset)` → `(path, line)`. Binary-searches segments, then
  either counts newlines (full mode) or looks up `line_map` (signature mode, where
  emitted lines are non-contiguous).
- `PackedContext.resolve_snippet(text)` → finds a quoted line in the stream, then
  calls `resolve`. **This is what grounds a finding.**

## Stage 5 — `infer.py`: string → model output

| Function | What it does |
|---|---|
| `MambaAuditor(model_id, load_in_4bit=True)` | Loads tokenizer + model, picks the compute dtype |
| `.build_prompt(context)` | Wraps the packed text in `SYSTEM_PROMPT` + `USER_TEMPLATE` via the chat template |
| `.generate(context, max_new_tokens)` | Greedy generation, returns `GenerationResult` with token counts and timings |
| `.count_tokens(text)` | Exact count — pass `auditor.tokenizer` back into `pack()` for an exact budget |
| `select_dtype()` | Compute capability ≥ 8.0 → bf16, else **fp16** (T4 is Turing, no real bf16) |
| `describe_environment()` | GPU, dtype, VRAM, and whether `mamba_ssm`/`causal_conv1d` are present |

`do_sample=False` — greedy. Auditing is factual; sampling manufactures findings.
`MODEL_MAX_CONTEXT = 32_768` is the model's trained ceiling; exceeding it warns.

## Stage 6 — `findings.py`: model output → grounded findings

| Function | What it does |
|---|---|
| `parse_findings(model_text, packed)` | **Entry point.** Parse → validate → ground → sort |
| `extract_json_array(text)` | Three strategies: fenced block, bracket-balanced scan (string- and escape-aware), then salvage individual objects from a truncated array |
| `_coerce(raw)` | Clamps severity/category to the allowed sets, confidence to `[0,1]`, truncates long strings |

What `parse_findings` does to each finding, in order:

1. Any `files` entry not in the stream is **removed** and noted — the model invented it.
2. `evidence` is looked up via `packed.resolve_snippet`. Found → `location` set, `grounded=True`. Not found → `grounded=False` + a note.
3. Single-file findings (except hardcoded secrets) get flagged out of scope.
4. Sorted: grounded first, then severity, then confidence.

**Types:** `Finding` (with `.is_cross_file`, `.pretty()`) and `AuditReport`
(with `.grounded`, `.cross_file`, `.save()`).

**This stage is the honesty mechanism.** `grounded=False` means the model said
something it can't point at — a hallucination by construction. Week 4's precision
number counts only grounded findings, and reports the ungrounded rate separately.

## `cli.py` — the glue

| Function | Command |
|---|---|
| `_pipeline(args, tokenizer)` | Shared stages 1–4, with the `[1/4]…[4/4]` progress output |
| `cmd_graph` | `wca graph <target>` — graph stats + top-risk files + sample edges |
| `cmd_pack` | `wca pack <target>` — writes `context.pack.txt` + manifest |
| `cmd_scan` | `wca scan <target>` — full audit; checks torch is installed first |
| `cmd_env` | `wca env` — hardware report |

`cmd_scan` calls `_pipeline(tokenizer=auditor.tokenizer)`, so the scan path packs
with **exact** token counts while `cmd_pack` uses the estimate. Same code, both.

---

## Worked trace — one vulnerability through all six stages

Planted repo: `lib/config.py` holds a DB password, `app/handlers.py` logs it.
This is real output, not illustrative.

**1. Ingest** — 6 files kept.

```
demo@local: 6 files, 1.2 KB kept (skipped: none)
['app/handlers.py', 'app/util.js', 'lib/config.py', 'lib/db.py', 'lib/helper.js', 'lib/svc.go']
```

**2. Parse** — ASTs retained.

```
parsed 6 files (go=1, javascript=2, python=3); 0 unsupported, 0 failed
ParsedFile(lib/config.py) lang=python bytes=206 root=module children=5
```

**3. Graph** — symbols extracted, files linked.

```
6 files, 7 defs, 7 cross-file edges (call=4, import=3), 3 components

FileSymbols(lib/config.py):
  defs      = [('get_conn_string', 'function', line 6)]
  secretish = ['API_KEY = "sk-live-…"', 'DB_PASSWORD = "admin_password_123"', 'DB_URL = "postgresql://…"']
  risk_score= 15.0                      ← 3 secrets × 3.0

FileSymbols(app/handlers.py): imports=['lib.config', 'logging'] risk=1.0
```

**4a. Priority** — risk propagates one hop, so the *consumer* rises too:

```
lib/config.py  17.5     ← owns the secrets
lib/db.py      10.75    ← neighbour, inherits
app/handlers.py 10.25   ← neighbour, inherits (1.0 on its own)
app/util.js     6.5
lib/helper.js   3.5
lib/svc.go      2.0
```

`app/handlers.py` scores 1.0 alone but 10.25 after propagation — because the
*pair* is the vulnerability, and both halves must be in the context.

**4b. Order and pack** — config emitted immediately before its consumer:

```
order_files: ['lib/config.py', 'app/handlers.py', 'lib/db.py', 'lib/helper.js', 'app/util.js', 'lib/svc.go']
packed 6 full / 0 sig / 0 omitted -> ~515 tok (6.4% of 8,000)
Segment[0]: path=lib/config.py mode=full chars=245..451 line_start=0
resolve(285) -> ('lib/config.py', 3)
```

**5. Infer** — the packed string goes to the model (not run here; needs a GPU).

**6. Findings** — two claims fed in, one real and one fabricated:

```
grounded=True   location=lib/config.py:3   files=['lib/config.py', 'app/handlers.py']  notes=[]
grounded=False  location=None              files=[]
                notes=['paths not in stream: totally/fake.py',
                       'evidence line not found in stream',
                       'single-file finding (out of scope for the cross-file claim)']
```

The real finding resolved to `lib/config.py:3` — the actual line holding
`DB_PASSWORD`. The fabricated one had its invented path stripped and was flagged
ungrounded on three independent counts.

---

## If you only remember four things

1. **`PackedContext.text` is what the model sees; `.segments` is how you get back
   to real lines.** Every other stage exists to build those two.
2. **`pack()` separates *what* (priority) from *where* (graph order).** That split
   is the load-bearing design decision, and collapsing it was a real bug.
3. **`risk_score` allocates budget, it does not judge code.** Don't let it leak
   into the eval as if it were a detection signal.
4. **`grounded=False` means the model made it up.** That flag is what makes the
   week-4 number defensible instead of self-graded.
