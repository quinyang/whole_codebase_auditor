"""Whole-Codebase Auditor (WCA).

Pipeline: ingest -> parse -> graph -> pack -> infer -> findings.

Each stage is importable and independently testable. Only `infer` requires a GPU
or the `gpu` extra; everything up to and including `pack` runs on CPU.
"""

__version__ = "0.10.1"


def version_tuple(v: str | None = None) -> tuple[int, ...]:
    """Parse a version into comparable integers.

    String comparison is lexicographic, so `"0.10.0" >= "0.9.0"` is False -- it
    compares "1" against "9" at the third character and stops. A notebook guard
    written that way rejects the exact build it was meant to require.
    """
    parts = (v or __version__).split(".")
    return tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in parts)


def version_at_least(*minimum: int) -> bool:
    """True if the installed wca is at least `minimum`, e.g. (0, 10, 0)."""
    return version_tuple() >= tuple(minimum)


from wca.graph import SymbolGraph, build_graph
from wca.ingest import SourceFile, ingest
from wca.pack import PackedContext, Segment, pack
from wca.parse import LanguageDispatcher, ParsedFile, parse_files

__all__ = [
    "LanguageDispatcher",
    "PackedContext",
    "ParsedFile",
    "Segment",
    "SourceFile",
    "SymbolGraph",
    "build_graph",
    "ingest",
    "pack",
    "parse_files",
    "version_at_least",
    "version_tuple",
]
