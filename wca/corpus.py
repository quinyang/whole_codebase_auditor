"""Synthetic benchmark corpus.

Measured on a T4: the budget fits ~2,200 tokens of source across the two files a
cross-file vulnerability spans. Real libraries have 300-400 line modules -- one
of them is ~5,600 tokens on its own -- so both halves physically cannot be in
context, and every case would score a miss for reasons of memory rather than
detection. See `PLAN.md` for the measurement.

So the corpus is generated instead: ten repositories of ~8-12 modules at ~40
lines each, varied in domain, layout, and dependency shape. This is a narrower
claim than "real repositories" and should be described that way -- but it tests
the thing the project is actually about, which a corpus that does not fit in
memory does not test at all.

What is deliberately varied, so the benchmark is not ten copies of one repo:
  * module layout      flat / package / src-layout / nested
  * dependency shape   chain, star, diamond
  * domain vocabulary  web, cli, storage, auth, jobs
  * incidental risk     some repos contain sinks and secret-ish words already,
                        so the risk heuristic cannot trivially locate the plant
"""

from __future__ import annotations

import random

from wca.ingest import RepoBundle, SourceFile

LAYOUTS = ("flat", "pkg", "src", "nested")
SHAPES = ("chain", "star", "diamond")

DOMAINS: dict[str, dict[str, list[str]]] = {
    "web": {
        "modules": ["config", "routes", "views", "middleware", "serializers", "errors"],
        "verbs": ["handle", "render", "dispatch", "resolve", "encode"],
        "nouns": ["request", "response", "session", "payload", "header"],
    },
    "cli": {
        "modules": ["config", "commands", "parser", "output", "completion", "shell"],
        "verbs": ["parse", "format", "invoke", "expand", "validate"],
        "nouns": ["argument", "option", "command", "flag", "context"],
    },
    "storage": {
        "modules": ["config", "backend", "cache", "index", "serialize", "migrate"],
        "verbs": ["read", "write", "evict", "compact", "flush"],
        "nouns": ["record", "bucket", "entry", "shard", "checkpoint"],
    },
    "auth": {
        "modules": ["config", "tokens", "policy", "principals", "audit", "scopes"],
        "verbs": ["issue", "verify", "revoke", "refresh", "authorise"],
        "nouns": ["token", "claim", "grant", "scope", "principal"],
    },
    "jobs": {
        "modules": ["config", "queue", "worker", "scheduler", "retry", "metrics"],
        "verbs": ["enqueue", "consume", "schedule", "retry", "record"],
        "nouns": ["task", "job", "attempt", "backoff", "batch"],
    },
}

# Incidental noise so the risk heuristic cannot simply point at the plant.
# A real repo mentions "token" and calls "execute" without being vulnerable.
DECOYS = (
    '    # NOTE: cache_key is derived, not a secret\n    cache_key = "v1:%s" % name\n',
    "    cursor = self._conn.cursor()\n    cursor.execute(QUERY_BY_ID, (record_id,))\n",
    '    token_type = "Bearer"  # scheme name, not a credential\n',
    "    payload = json.loads(raw) if raw else {}\n",
)


def _module_path(layout: str, pkg: str, name: str) -> str:
    if layout == "flat":
        return f"{name}.py"
    if layout == "pkg":
        return f"{pkg}/{name}.py"
    if layout == "src":
        return f"src/{pkg}/{name}.py"
    return f"src/{pkg}/core/{name}.py"


def _import_stmt(layout: str, pkg: str, name: str) -> str:
    if layout == "flat":
        return f"import {name}"
    if layout == "pkg":
        return f"from {pkg} import {name}"
    if layout == "src":
        return f"from {pkg} import {name}"
    return f"from {pkg}.core import {name}"


def _deps(shape: str, i: int, n: int, rng: random.Random) -> list[int]:
    """Which earlier modules module `i` imports."""
    if i == 0:
        return []
    if shape == "chain":
        return [i - 1]
    if shape == "star":
        return [0]
    # diamond: two roots feed a join
    if i <= 2:
        return [0]
    return sorted(rng.sample(range(1, i), min(2, i - 1)))


def _module_source(
    domain: str, layout: str, pkg: str, names: list[str], i: int, deps: list[int],
    rng: random.Random,
) -> str:
    d = DOMAINS[domain]
    name = names[i]
    lines = [f'"""{name}: {domain} helpers."""', "", "import json", ""]
    for j in deps:
        lines.append(_import_stmt(layout, pkg, names[j]))
    lines.append("")
    lines.append("")

    if i == 0:  # config-ish root
        lines += [
            f'DEFAULT_TIMEOUT = {rng.choice([5, 10, 30])}',
            f'SERVICE_NAME = "{pkg}-{domain}"',
            "",
            "",
            "def settings():",
            '    """Return the effective settings mapping."""',
            "    return {",
            '        "timeout": DEFAULT_TIMEOUT,',
            '        "service": SERVICE_NAME,',
            "    }",
            "",
        ]

    for k in range(rng.randint(3, 5)):
        verb, noun = rng.choice(d["verbs"]), rng.choice(d["nouns"])
        fn = f"{verb}_{noun}_{i}{k}"
        lines += [f"def {fn}({noun}, *, strict=False):", f'    """{verb.title()} a {noun}."""']
        if rng.random() < 0.4:
            lines.append(rng.choice(DECOYS).rstrip("\n"))
        if deps and rng.random() < 0.6:
            ref = names[rng.choice(deps)]
            lines.append(f"    base = {ref}.settings() if hasattr({ref}, 'settings') else {{}}")
            lines.append(f"    return {{**base, '{noun}': {noun}, 'strict': strict}}")
        else:
            lines.append(f"    return {{'{noun}': {noun}, 'strict': strict}}")
        lines.append("")
    return "\n".join(lines) + "\n"


def generate_repo(name: str, *, seed: int, n_modules: int | None = None) -> RepoBundle:
    """Build one synthetic repository as an in-memory `RepoBundle`."""
    rng = random.Random(seed)
    domain = rng.choice(sorted(DOMAINS))
    layout = rng.choice(LAYOUTS)
    shape = rng.choice(SHAPES)
    pkg = name.split("/")[-1].replace("-", "_")

    n = n_modules or rng.randint(8, 12)
    pool = DOMAINS[domain]["modules"]
    names = [pool[i % len(pool)] + ("" if i < len(pool) else f"{i // len(pool)}") for i in range(n)]

    files: list[SourceFile] = []
    for i in range(n):
        deps = _deps(shape, i, n, rng)
        text = _module_source(domain, layout, pkg, names, i, deps, rng)
        path = _module_path(layout, pkg, names[i])
        files.append(SourceFile(path=path, data=text.encode("utf-8"), text=text))

    if layout != "flat":
        init_dirs = sorted({p.rsplit("/", 1)[0] for p in (f.path for f in files)})
        for d in init_dirs:
            t = f'"""{pkg} package."""\n'
            files.append(SourceFile(path=f"{d}/__init__.py", data=t.encode("utf-8"), text=t))

    bundle = RepoBundle(name=name, ref=f"synthetic:{seed}")
    bundle.files = sorted(files, key=lambda f: f.path)
    return bundle


SYNTHETIC_CORPUS: tuple[str, ...] = (
    "wca-bench/webgate",
    "wca-bench/clikit",
    "wca-bench/storelet",
    "wca-bench/authring",
    "wca-bench/jobline",
    "wca-bench/routecore",
    "wca-bench/cachemap",
    "wca-bench/tokenmint",
    "wca-bench/taskflow",
    "wca-bench/indexer",
)


def generate_corpus(
    names: tuple[str, ...] = SYNTHETIC_CORPUS, *, seed: int = 20260829
) -> list[RepoBundle]:
    """Ten repositories, each a different domain/layout/dependency shape."""
    return [generate_repo(n, seed=seed + i * 977) for i, n in enumerate(names)]
