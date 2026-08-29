"""Whole-Codebase Auditor (WCA).

Pipeline: ingest -> parse -> graph -> pack -> infer -> findings.

Each stage is importable and independently testable. Only `infer` requires a GPU
or the `gpu` extra; everything up to and including `pack` runs on CPU.
"""

__version__ = "0.8.0"

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
]
