"""Stage 3 -- Symbol graph.

This is what makes the cross-file claim defensible rather than a hope about long
context. From the ASTs already built in `parse.py`, extract per file:

    imports   -- module/header/package strings
    defs      -- functions, classes, methods, types (name + byte span + lines)
    calls     -- call sites, by callee name
    literals  -- string literals (secrets, URLs, SQL, paths)

Then link files:

    IMPORT  A -> B   A imports a module that resolves to B
    CALL    A -> B   A calls a symbol defined only in B
    LITERAL A -> B   A and B share a distinctive string literal
                     (this is the classic "secret defined here, used there" edge)

Implementation note: per-language tree-sitter *queries* would be more precise but
mean maintaining 7 query files. A node-type table plus a single generic walk
covers the four categories above across every grammar with far less surface area,
and any grammar we haven't tabulated degrades to "no symbols" rather than
crashing. Precision here is traded for breadth on purpose -- the packer only
needs relatedness ordering, not a sound call graph.
"""

from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass, field

from tree_sitter import Node

from wca.parse import ParsedFile

# --------------------------------------------------------------------------- #
# Per-language node-type tables
# --------------------------------------------------------------------------- #

DEF_NODES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "function",
    },
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "namespace",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "constructor_declaration": "method",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
    },
}

IMPORT_NODES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "go": {"import_spec"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
    "java": {"import_declaration"},
    "rust": {"use_declaration"},
}

CALL_NODES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "go": {"call_expression"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "rust": {"call_expression", "macro_invocation"},
}

STRING_NODES: dict[str, set[str]] = {
    "python": {"string"},
    "javascript": {"string", "template_string"},
    "go": {"interpreted_string_literal", "raw_string_literal"},
    "c": {"string_literal"},
    "cpp": {"string_literal", "raw_string_literal"},
    "java": {"string_literal"},
    "rust": {"string_literal", "raw_string_literal"},
}

NAME_NODE_TYPES = frozenset(
    {"identifier", "field_identifier", "type_identifier", "property_identifier", "word"}
)

# Sinks whose presence makes a file worth spending full-body budget on.
# Deliberately pattern-level, not exploit-level -- this project demonstrates
# systems engineering, not vulnerability research.
DANGEROUS_CALLS = frozenset(
    {
        "eval", "exec", "execfile", "compile", "system", "popen", "spawn",
        "execve", "execl", "execlp", "execvp", "call", "check_output", "run",
        "pickle", "loads", "load", "yaml_load", "unserialize", "deserialize",
        "Function", "setTimeout", "innerHTML", "dangerouslySetInnerHTML",
        "query", "execute", "executemany", "raw", "cursor", "format",
        "strcpy", "strcat", "sprintf", "gets", "memcpy", "alloca",
        "Exec", "Command", "Sprintf", "Query", "Unmarshal",
        "createElement", "require", "importlib", "__import__", "getattr",
        "sudo", "chmod", "chown", "urlopen", "get", "post", "request",
    }
)

SECRETISH = re.compile(
    r"(?i)(api[_-]?key|secret|passwd|password|token|credential|private[_-]?key|"
    r"aws_|bearer\s|authorization|-----BEGIN|mongodb(\+srv)?://|postgres(ql)?://|"
    r"mysql://|redis://|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,})"
)

# Literals too generic to be evidence of a cross-file link.
LITERAL_MIN_LEN = 8
LITERAL_MAX_FILES = 4  # a string in >4 files is boilerplate, not a link


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Definition:
    name: str
    kind: str
    path: str
    start_byte: int
    end_byte: int
    start_line: int  # 0-indexed
    end_line: int


@dataclass
class FileSymbols:
    path: str
    lang: str
    imports: list[str] = field(default_factory=list)
    defs: list[Definition] = field(default_factory=list)
    calls: set[str] = field(default_factory=set)
    literals: set[str] = field(default_factory=set)
    secretish: list[str] = field(default_factory=list)
    danger_hits: set[str] = field(default_factory=set)

    @property
    def risk_score(self) -> float:
        """Heuristic priority for the packer. Not a vulnerability score."""
        return 3.0 * len(self.secretish) + 1.0 * len(self.danger_hits)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str  # import | call | literal
    detail: str = ""


@dataclass
class SymbolGraph:
    symbols: dict[str, FileSymbols] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # def name -> paths defining it
    def_index: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def neighbors(self, path: str) -> set[str]:
        out = set()
        for e in self.edges:
            if e.src == path:
                out.add(e.dst)
            elif e.dst == path:
                out.add(e.src)
        return out

    def degree(self) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for e in self.edges:
            d[e.src] += 1
            d[e.dst] += 1
        return d

    def components(self) -> list[list[str]]:
        """Connected components over the undirected edge set, largest first."""
        adj: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            adj[e.src].add(e.dst)
            adj[e.dst].add(e.src)
        seen: set[str] = set()
        comps: list[list[str]] = []
        for start in self.symbols:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                n = stack.pop()
                comp.append(n)
                for m in adj[n]:
                    if m not in seen and m in self.symbols:
                        seen.add(m)
                        stack.append(m)
            comps.append(sorted(comp))
        comps.sort(key=len, reverse=True)
        return comps

    def summary(self) -> str:
        by_kind: dict[str, int] = defaultdict(int)
        for e in self.edges:
            by_kind[e.kind] += 1
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "none"
        n_defs = sum(len(s.defs) for s in self.symbols.values())
        return (
            f"{len(self.symbols)} files, {n_defs} defs, "
            f"{len(self.edges)} cross-file edges ({kinds}), "
            f"{len(self.components())} components"
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: Node, src: bytes) -> str | None:
    """Prefer the `name` field; otherwise the first identifier-ish descendant.

    C/C++ hide the name under nested declarators, so a bounded DFS is needed.
    """
    named = node.child_by_field_name("name")
    if named is not None:
        return _node_text(named, src)
    decl = node.child_by_field_name("declarator")
    target = decl if decl is not None else node
    stack, budget = [target], 64
    while stack and budget > 0:
        budget -= 1
        n = stack.pop(0)
        if n.type in NAME_NODE_TYPES:
            return _node_text(n, src)
        stack.extend(n.children)
    return None


def _callee_name(node: Node, src: bytes) -> str | None:
    """Last identifier of the callee expression: `a.b.c(x)` -> 'c'."""
    fn = node.child_by_field_name("function") or node.child_by_field_name("constructor")
    if fn is None:
        fn = node.child_by_field_name("name") or node.child_by_field_name("macro")
    if fn is None:
        return None
    if fn.type in NAME_NODE_TYPES:
        return _node_text(fn, src)
    last = None
    stack = [fn]
    while stack:
        n = stack.pop()
        if n.type in NAME_NODE_TYPES:
            last = n
        stack.extend(reversed(n.children))
    return _node_text(last, src) if last is not None else None


def _clean_literal(raw: str) -> str:
    return raw.strip().strip("\"'`").strip()


def _import_targets(node: Node, src: bytes, lang: str) -> list[str]:
    """Pull the module/header/package string out of an import node."""
    text = _node_text(node, src)
    if lang in ("c", "cpp"):
        m = re.search(r'[<"]([^>"]+)[>"]', text)
        return [m.group(1)] if m else []
    if lang == "javascript":
        m = re.search(r'from\s+[\'"]([^\'"]+)[\'"]', text) or re.search(
            r'[\'"]([^\'"]+)[\'"]', text
        )
        return [m.group(1)] if m else []
    if lang == "go":
        m = re.search(r'"([^"]+)"', text)
        return [m.group(1)] if m else []
    if lang == "java":
        m = re.search(r"import\s+(?:static\s+)?([\w.]+)", text)
        return [m.group(1)] if m else []
    if lang == "rust":
        m = re.search(r"use\s+([\w:]+)", text)
        return [m.group(1).replace("::", ".")] if m else []
    # python
    out: list[str] = []
    m = re.match(r"\s*from\s+([.\w]+)\s+import", text)
    if m:
        out.append(m.group(1))
    else:
        out.extend(re.findall(r"(?:import|,)\s+([.\w]+)", text))
    return out


def extract_symbols(pf: ParsedFile) -> FileSymbols:
    """Single DFS over the AST, dispatching on the per-language node tables."""
    lang = pf.lang
    src = pf.source
    defs_tbl = DEF_NODES.get(lang, {})
    imports_tbl = IMPORT_NODES.get(lang, set())
    calls_tbl = CALL_NODES.get(lang, set())
    strings_tbl = STRING_NODES.get(lang, set())

    fs = FileSymbols(path=pf.path, lang=lang)
    stack = [pf.root]
    while stack:
        node = stack.pop()
        t = node.type

        if t in defs_tbl:
            name = _find_name(node, src)
            if name:
                fs.defs.append(
                    Definition(
                        name=name,
                        kind=defs_tbl[t],
                        path=pf.path,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        start_line=node.start_point[0],
                        end_line=node.end_point[0],
                    )
                )
        elif t in imports_tbl:
            fs.imports.extend(_import_targets(node, src, lang))
        elif t in calls_tbl:
            name = _callee_name(node, src)
            if name:
                fs.calls.add(name)
                if name in DANGEROUS_CALLS:
                    fs.danger_hits.add(name)
        elif t in strings_tbl:
            raw = _node_text(node, src)
            if len(raw) <= 512:
                val = _clean_literal(raw)
                if len(val) >= LITERAL_MIN_LEN:
                    fs.literals.add(val)
                if SECRETISH.search(raw):
                    fs.secretish.append(val[:120])

        stack.extend(node.children)

    # Assignment-style secrets that aren't string nodes (env keys, headers).
    for line in pf.text.splitlines():
        if SECRETISH.search(line) and "=" in line and len(line) < 300:
            fs.secretish.append(line.strip()[:120])

    fs.secretish = sorted(set(fs.secretish))[:40]
    return fs


# --------------------------------------------------------------------------- #
# Linking
# --------------------------------------------------------------------------- #


def _module_keys(path: str) -> set[str]:
    """Names by which other files might refer to this one."""
    stem = posixpath.splitext(path)[0]
    parts = stem.split("/")
    keys = {stem, stem.replace("/", "."), parts[-1], posixpath.basename(path)}
    if parts[-1] in ("__init__", "index", "mod"):
        keys.add(parts[-2] if len(parts) > 1 else parts[-1])
        keys.add("/".join(parts[:-1]))
    # trailing suffixes so 'pkg.sub.mod' matches 'a/b/pkg/sub/mod.py'
    for i in range(1, min(len(parts), 4)):
        keys.add(".".join(parts[-i:]))
        keys.add("/".join(parts[-i:]))
    return {k for k in keys if k}


def _resolve_import(imp: str, importer: str, key_index: dict[str, list[str]]) -> list[str]:
    """Map one import string to candidate files.

    Handles three shapes: dotted module paths (`pkg.sub.mod`, `pkg::sub::mod`),
    filesystem-relative specifiers (`../lib/helper.js`), and bare header names
    (`config.h`). Relative specifiers are resolved against the importer's own
    directory first, which is the only form where position matters.
    """
    candidates: list[str] = []

    if imp.startswith((".", "/")) or "/" in imp:
        base = posixpath.dirname(importer)
        joined = posixpath.normpath(posixpath.join(base, imp)) if imp.startswith(".") else imp
        for form in (joined, joined.lstrip("./"), imp.lstrip("./")):
            candidates += [form, posixpath.splitext(form)[0]]
        candidates.append(posixpath.splitext(posixpath.basename(imp))[0])
    else:
        norm = imp.lstrip(".").replace("::", ".")
        candidates += [norm, norm.replace(".", "/"), norm.split(".")[-1]]

    out: list[str] = []
    for c in candidates:
        if not c:
            continue
        for t in key_index.get(c, []):
            if t != importer and t not in out:
                out.append(t)
        if out:
            break  # most specific candidate wins
    return out


def build_graph(parsed: list[ParsedFile], *, max_literal_edges: int = 2000) -> SymbolGraph:
    g = SymbolGraph()
    for pf in parsed:
        g.symbols[pf.path] = extract_symbols(pf)

    # module key -> owning file paths
    key_index: dict[str, list[str]] = defaultdict(list)
    for path in g.symbols:
        for k in _module_keys(path):
            key_index[k].append(path)

    for path, fs in g.symbols.items():
        for d in fs.defs:
            g.def_index[d.name].append(path)

    seen: set[tuple[str, str, str]] = set()

    def add(src: str, dst: str, kind: str, detail: str = "") -> None:
        if src == dst:
            return
        key = (src, dst, kind)
        if key in seen:
            return
        seen.add(key)
        g.edges.append(Edge(src=src, dst=dst, kind=kind, detail=detail))

    # IMPORT edges
    for path, fs in g.symbols.items():
        for imp in fs.imports:
            for t in _resolve_import(imp, path, key_index):
                add(path, t, "import", imp)

    # CALL edges -- only when the callee is defined in exactly one other file.
    # Ambiguous names (defined in many files) carry no information.
    for path, fs in g.symbols.items():
        own = {d.name for d in fs.defs}
        for c in fs.calls:
            if c in own:
                continue
            owners = g.def_index.get(c, [])
            if len(owners) == 1 and owners[0] != path:
                add(path, owners[0], "call", c)

    # LITERAL edges -- distinctive shared strings. Bounded so a repo with a huge
    # shared constants file doesn't produce a quadratic edge blowup.
    lit_owners: dict[str, list[str]] = defaultdict(list)
    for path, fs in g.symbols.items():
        for lit in fs.literals:
            lit_owners[lit].append(path)
    n_lit = 0
    for lit, owners in lit_owners.items():
        if not (2 <= len(owners) <= LITERAL_MAX_FILES):
            continue
        if not (SECRETISH.search(lit) or "/" in lit or "://" in lit or len(lit) >= 20):
            continue
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                add(owners[i], owners[j], "literal", lit[:60])
                n_lit += 1
        if n_lit >= max_literal_edges:
            break

    return g
