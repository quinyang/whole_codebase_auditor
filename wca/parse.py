"""Stage 2 -- Parse.

Tree-sitter AST per file. Unlike the previous version this *keeps* the tree --
`graph.py` consumes it. Storing only `root_type`/`child_count` was why the
"structured context streams" claim wasn't real.

Fixes carried over from the old `tree_parser.py` / `language_dispatcher.py`:
  1. `Parser(module.language())` no longer works on tree-sitter >= 0.25 --
     `language()` returns a PyCapsule and must be wrapped in `Language(...)`.
  2. The git tree and the parse tree were both bound to `tree`, shadowing.
  3. A parse failure `return`ed from inside the file loop, aborting the whole
     scan and passing None downstream. Now it records the error and continues.
  4. `.rs` / `.java` were in the language map but the grammars weren't declared
     as dependencies. Both are now in pyproject, and a missing grammar degrades
     to "unsupported extension" instead of raising.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from tree_sitter import Language, Node, Parser, Tree

from wca.ingest import SourceFile

# ext -> (pypi module, canonical language name)
LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    ".py": ("tree_sitter_python", "python"),
    ".pyi": ("tree_sitter_python", "python"),
    ".c": ("tree_sitter_c", "c"),
    ".h": ("tree_sitter_c", "c"),
    ".cpp": ("tree_sitter_cpp", "cpp"),
    ".cxx": ("tree_sitter_cpp", "cpp"),
    ".cc": ("tree_sitter_cpp", "cpp"),
    ".hpp": ("tree_sitter_cpp", "cpp"),
    ".hxx": ("tree_sitter_cpp", "cpp"),
    ".cu": ("tree_sitter_cpp", "cpp"),  # CUDA parses close enough as C++
    ".cuh": ("tree_sitter_cpp", "cpp"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".mjs": ("tree_sitter_javascript", "javascript"),
    ".cjs": ("tree_sitter_javascript", "javascript"),
    ".jsx": ("tree_sitter_javascript", "javascript"),
    ".go": ("tree_sitter_go", "go"),
    ".java": ("tree_sitter_java", "java"),
    ".rs": ("tree_sitter_rust", "rust"),
}


class LanguageDispatcher:
    """Lazily loads and caches one Parser per language."""

    def __init__(self, language_map: dict[str, tuple[str, str]] | None = None, quiet: bool = True):
        self.language_map = dict(language_map or LANGUAGE_MAP)
        self.quiet = quiet
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self._unavailable: set[str] = set()  # grammars that failed to import

    def language_for_file(self, filename: str) -> str | None:
        ext = os.path.splitext(filename)[1].lower()
        entry = self.language_map.get(ext)
        return entry[1] if entry else None

    def get_parser_for_file(self, filename: str) -> tuple[Parser | None, Language | None]:
        ext = os.path.splitext(filename)[1].lower()
        entry = self.language_map.get(ext)
        if entry is None:
            return None, None
        module_name, lang_name = entry
        if lang_name in self._parsers:
            return self._parsers[lang_name], self._languages[lang_name]
        if lang_name in self._unavailable:
            return None, None
        try:
            self._load(module_name, lang_name)
        except Exception as exc:  # grammar not installed -> degrade, don't crash
            self._unavailable.add(lang_name)
            if not self.quiet:
                print(f"[wca] grammar unavailable for {lang_name}: {exc}")
            return None, None
        return self._parsers[lang_name], self._languages[lang_name]

    def _load(self, module_name: str, lang_name: str) -> None:
        module = importlib.import_module(module_name)
        # tree-sitter >= 0.25: module.language() is a PyCapsule, not a Language.
        raw = module.language()
        language = raw if isinstance(raw, Language) else Language(raw)
        self._languages[lang_name] = language
        self._parsers[lang_name] = Parser(language)
        if not self.quiet:
            print(f"[wca] loaded grammar: {lang_name}")

    @property
    def loaded(self) -> list[str]:
        return sorted(self._parsers)


@dataclass
class ParsedFile:
    """A source file plus its retained AST."""

    path: str
    lang: str
    source: bytes
    text: str
    tree: Tree
    has_error: bool = False

    @property
    def root(self) -> Node:
        return self.tree.root_node

    @property
    def n_lines(self) -> int:
        return self.text.count("\n") + 1

    def slice(self, start_byte: int, end_byte: int) -> str:
        return self.source[start_byte:end_byte].decode("utf-8", errors="replace")


@dataclass
class ParseResult:
    files: list[ParsedFile] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (path, error)

    def summary(self) -> str:
        by_lang: dict[str, int] = {}
        for f in self.files:
            by_lang[f.lang] = by_lang.get(f.lang, 0) + 1
        langs = ", ".join(f"{k}={v}" for k, v in sorted(by_lang.items())) or "none"
        return (
            f"parsed {len(self.files)} files ({langs}); "
            f"{len(self.unsupported)} unsupported, {len(self.failed)} failed"
        )


def parse_files(
    files: Iterable[SourceFile],
    dispatcher: LanguageDispatcher | None = None,
) -> ParseResult:
    """Parse every supported file. One bad file never aborts the scan."""
    dispatcher = dispatcher or LanguageDispatcher()
    result = ParseResult()

    for sf in files:
        parser, _ = dispatcher.get_parser_for_file(sf.path)
        if parser is None:
            result.unsupported.append(sf.path)
            continue
        try:
            tree = parser.parse(sf.data)
        except Exception as exc:
            # Previously this `return`ed, killing the run. Keep going.
            result.failed.append((sf.path, repr(exc)))
            continue
        lang = dispatcher.language_for_file(sf.path) or "unknown"
        result.files.append(
            ParsedFile(
                path=sf.path,
                lang=lang,
                source=sf.data,
                text=sf.text,
                tree=tree,
                has_error=bool(tree.root_node.has_error),
            )
        )
    return result
