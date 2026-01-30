"""
Language Dispatcher Module
==========================
Handles Tree-sitter parser initialization for multiple programming languages.

Improvements over original:
1. Lazy loading with proper caching
2. Graceful error handling for missing language packages
3. Support for more file extensions
4. Thread-safe parser creation
"""

import os
import importlib
import threading
from typing import Optional, Tuple, Dict, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from tree_sitter import Parser, Language


@dataclass
class LanguageInfo:
    """Information about a supported programming language."""
    name: str
    package: str
    extensions: Tuple[str, ...]
    comment_patterns: Tuple[str, ...]  # For potential comment extraction


# Comprehensive language registry
LANGUAGE_REGISTRY: Dict[str, LanguageInfo] = {
    "python": LanguageInfo(
        name="python",
        package="tree_sitter_python",
        extensions=(".py", ".pyw", ".pyi"),
        comment_patterns=("#", '"""', "'''"),
    ),
    "c": LanguageInfo(
        name="c",
        package="tree_sitter_c",
        extensions=(".c", ".h"),
        comment_patterns=("//", "/*"),
    ),
    "cpp": LanguageInfo(
        name="cpp",
        package="tree_sitter_cpp",
        extensions=(".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".cu", ".cuh"),
        comment_patterns=("//", "/*"),
    ),
    "javascript": LanguageInfo(
        name="javascript",
        package="tree_sitter_javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        comment_patterns=("//", "/*"),
    ),
    "typescript": LanguageInfo(
        name="typescript",
        package="tree_sitter_typescript",
        extensions=(".ts", ".tsx"),
        comment_patterns=("//", "/*"),
    ),
    "go": LanguageInfo(
        name="go",
        package="tree_sitter_go",
        extensions=(".go",),
        comment_patterns=("//", "/*"),
    ),
    "rust": LanguageInfo(
        name="rust",
        package="tree_sitter_rust",
        extensions=(".rs",),
        comment_patterns=("//", "/*"),
    ),
    "java": LanguageInfo(
        name="java",
        package="tree_sitter_java",
        extensions=(".java",),
        comment_patterns=("//", "/*"),
    ),
    "ruby": LanguageInfo(
        name="ruby",
        package="tree_sitter_ruby",
        extensions=(".rb", ".rake", ".gemspec"),
        comment_patterns=("#",),
    ),
    "php": LanguageInfo(
        name="php",
        package="tree_sitter_php",
        extensions=(".php",),
        comment_patterns=("//", "/*", "#"),
    ),
    "csharp": LanguageInfo(
        name="c_sharp",
        package="tree_sitter_c_sharp",
        extensions=(".cs",),
        comment_patterns=("//", "/*"),
    ),
    "swift": LanguageInfo(
        name="swift",
        package="tree_sitter_swift",
        extensions=(".swift",),
        comment_patterns=("//", "/*"),
    ),
    "kotlin": LanguageInfo(
        name="kotlin",
        package="tree_sitter_kotlin",
        extensions=(".kt", ".kts"),
        comment_patterns=("//", "/*"),
    ),
    "scala": LanguageInfo(
        name="scala",
        package="tree_sitter_scala",
        extensions=(".scala", ".sc"),
        comment_patterns=("//", "/*"),
    ),
}


class LanguageDispatcher:
    """
    Thread-safe language dispatcher for Tree-sitter parsers.
    
    Usage:
        dispatcher = LanguageDispatcher()
        parser, language = dispatcher.get_parser_for_file("main.py")
        if parser:
            tree = parser.parse(code_bytes)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._parsers: Dict[str, "Parser"] = {}
        self._languages: Dict[str, "Language"] = {}
        self._failed_packages: Set[str] = set()  # Don't retry failed imports
        
        # Build extension -> language mapping
        self._ext_to_lang: Dict[str, str] = {}
        for lang_name, info in LANGUAGE_REGISTRY.items():
            for ext in info.extensions:
                self._ext_to_lang[ext.lower()] = lang_name
    
    def get_supported_extensions(self) -> Set[str]:
        """Return all supported file extensions."""
        return set(self._ext_to_lang.keys())
    
    def is_supported(self, filename: str) -> bool:
        """Check if a file type is supported."""
        _, ext = os.path.splitext(filename)
        return ext.lower() in self._ext_to_lang
    
    def get_parser_for_file(
        self, filename: str
    ) -> Tuple[Optional["Parser"], Optional["Language"]]:
        """
        Get a (parser, language) tuple for the given filename.
        
        Returns (None, None) if:
        - File has no extension
        - Extension is not supported
        - Language package failed to load
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # No extension (Dockerfile, Makefile, etc.)
        if not ext:
            return None, None
        
        # Unsupported extension
        if ext not in self._ext_to_lang:
            return None, None
        
        lang_name = self._ext_to_lang[ext]
        lang_info = LANGUAGE_REGISTRY[lang_name]
        
        # Check if we already failed to load this package
        if lang_info.package in self._failed_packages:
            return None, None
        
        # Thread-safe lazy loading
        with self._lock:
            if lang_name not in self._parsers:
                try:
                    self._load_language(lang_name, lang_info)
                except Exception as e:
                    self._failed_packages.add(lang_info.package)
                    if self.verbose:
                        print(f"⚠️  Failed to load {lang_name} parser: {e}")
                    return None, None
        
        return self._parsers.get(lang_name), self._languages.get(lang_name)
    
    def _load_language(self, lang_name: str, lang_info: LanguageInfo) -> None:
        """
        Internal: Load a language parser.
        
        This handles the Tree-sitter API differences between versions.
        """
        # Dynamic import
        try:
            module = importlib.import_module(lang_info.package)
        except ImportError as e:
            raise ImportError(
                f"Package '{lang_info.package}' not installed. "
                f"Install with: pip install {lang_info.package}"
            ) from e
        
        # Get language pointer (PyCapsule)
        language_func = getattr(module, "language", None)
        if language_func is None:
            raise AttributeError(
                f"Package '{lang_info.package}' has no 'language()' function. "
                f"It may be incompatible with tree-sitter >= 0.22"
            )
        
        language_ptr = language_func()
        
        # Import tree_sitter components
        # Handle both old and new API
        try:
            from tree_sitter import Language, Parser
        except ImportError:
            raise ImportError(
                "tree-sitter package not installed. "
                "Install with: pip install tree-sitter>=0.22"
            )
        
        # Wrap the pointer in a Language object
        try:
            language_obj = Language(language_ptr)
        except TypeError:
            # Older tree-sitter versions might work differently
            language_obj = language_ptr
        
        # Create parser and set language
        parser = Parser(language_obj)
        
        # Cache
        self._languages[lang_name] = language_obj
        self._parsers[lang_name] = parser
        
        if self.verbose:
            print(f"✅ Loaded grammar: {lang_name.upper()}")
    
    def get_language_info(self, filename: str) -> Optional[LanguageInfo]:
        """Get language metadata for a file."""
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext not in self._ext_to_lang:
            return None
        
        return LANGUAGE_REGISTRY[self._ext_to_lang[ext]]
    
    def parse_file(
        self, filename: str, content: bytes
    ) -> Optional[Tuple[object, bool]]:
        """
        Convenience method: Parse a file and return (tree, has_errors).
        
        Returns None if parsing is not supported.
        """
        parser, _ = self.get_parser_for_file(filename)
        if parser is None:
            return None
        
        tree = parser.parse(content)  # type: ignore
        return tree, tree.root_node.has_error  # type: ignore


# Singleton instance for convenience
_default_dispatcher: Optional[LanguageDispatcher] = None


def get_dispatcher(verbose: bool = True) -> LanguageDispatcher:
    """Get the default LanguageDispatcher singleton."""
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = LanguageDispatcher(verbose=verbose)
    return _default_dispatcher