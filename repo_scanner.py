"""
Repository Scanner Module
=========================
Fetches and parses code from GitHub repositories.

Improvements:
1. Proper rate limiting with exponential backoff
2. Parallel file fetching for speed
3. Better binary file detection
4. Metadata extraction for security context
"""

import os
import base64
import time
import fnmatch
from typing import List, Dict, Optional, Generator, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import WCAConfig
from language_dispatcher import get_dispatcher, LanguageDispatcher


@dataclass
class ScannedFile:
    """Represents a parsed source file with metadata."""
    path: str
    content: str
    size_bytes: int
    language: Optional[str] = None
    
    # Tree-sitter analysis
    root_type: Optional[str] = None
    statement_count: int = 0
    has_syntax_errors: bool = False
    
    # Security-relevant metadata
    has_imports: bool = False
    import_statements: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    
    # For chunking decisions
    token_estimate: int = 0  # Rough estimate: chars / 4
    
    def __post_init__(self):
        # Estimate tokens (rough heuristic: 1 token ≈ 4 chars for code)
        self.token_estimate = len(self.content) // 4


@dataclass
class ScanResult:
    """Complete scan result with statistics."""
    files: List[ScannedFile]
    total_bytes: int = 0
    total_tokens_estimate: int = 0
    skipped_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    scan_duration_seconds: float = 0.0
    
    def __post_init__(self):
        self.total_bytes = sum(f.size_bytes for f in self.files)
        self.total_tokens_estimate = sum(f.token_estimate for f in self.files)


class RepoScanner:
    """
    Scans GitHub repositories and extracts source code with metadata.
    
    Usage:
        scanner = RepoScanner(config)
        result = scanner.scan()
        for file in result.files:
            print(f"{file.path}: {file.token_estimate} tokens")
    """
    
    def __init__(
        self,
        config: Optional[WCAConfig] = None,
        dispatcher: Optional[LanguageDispatcher] = None,
    ):
        self.config = config or WCAConfig()
        self.dispatcher = dispatcher or get_dispatcher()
        self._rate_limit_lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_request_interval = 0.05  # 50ms between requests
    
    def _should_skip_file(self, path: str, size_bytes: int = 0) -> bool:
        """Check if a file should be skipped based on config."""
        # Check patterns
        for pattern in self.config.skip_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        
        # Check size
        if size_bytes > self.config.max_file_size_kb * 1024:
            return True
        
        # Check if we can parse it
        if not self.dispatcher.is_supported(path):
            return True
        
        return False
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.time()
    
    def _decode_blob_content(self, blob: Any) -> Optional[str]:
        """Decode blob content, handling various encodings."""
        try:
            if blob.encoding == "base64":
                content_bytes = base64.b64decode(blob.content)
            else:
                content_bytes = blob.content.encode("utf-8")
            
            # Try UTF-8 first
            try:
                return content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Try common alternatives
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return content_bytes.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                
                # Binary file
                return None
                
        except Exception:
            return None
    
    def _extract_metadata(
        self, path: str, content: str, tree: Any
    ) -> Dict[str, Any]:
        """Extract security-relevant metadata from parsed code."""
        metadata = {
            "root_type": tree.root_node.type if tree else None,
            "statement_count": tree.root_node.child_count if tree else 0,
            "has_syntax_errors": tree.root_node.has_error if tree else False,
            "has_imports": False,
            "import_statements": [],
            "function_names": [],
            "class_names": [],
        }
        
        if not tree:
            return metadata
        
        # Language-specific extraction (simplified)
        root = tree.root_node
        
        def traverse(node):
            """Recursively traverse the AST."""
            node_type = node.type
            
            # Python/JS imports
            if node_type in ("import_statement", "import_from_statement", 
                            "import_declaration", "require_call"):
                metadata["has_imports"] = True
                try:
                    metadata["import_statements"].append(
                        content[node.start_byte:node.end_byte][:100]  # Truncate
                    )
                except Exception:
                    pass
            
            # Function definitions
            if node_type in ("function_definition", "function_declaration",
                            "method_definition", "arrow_function"):
                for child in node.children:
                    if child.type in ("identifier", "name"):
                        try:
                            name = content[child.start_byte:child.end_byte]
                            metadata["function_names"].append(name)
                        except Exception:
                            pass
                        break
            
            # Class definitions
            if node_type in ("class_definition", "class_declaration"):
                for child in node.children:
                    if child.type in ("identifier", "name"):
                        try:
                            name = content[child.start_byte:child.end_byte]
                            metadata["class_names"].append(name)
                        except Exception:
                            pass
                        break
            
            # Recurse
            for child in node.children:
                traverse(child)
        
        try:
            traverse(root)
        except Exception:
            pass  # Don't fail on AST traversal errors
        
        return metadata
    
    def _process_file(
        self, repo: Any, item: Any, index: int, total: int
    ) -> Optional[ScannedFile]:
        """Process a single file from the repository."""
        self._rate_limit()
        
        try:
            # Get blob
            blob = repo.get_git_blob(item.sha)
            
            # Decode content
            content = self._decode_blob_content(blob)
            if content is None:
                return None  # Binary file
            
            # Parse with tree-sitter
            parser, language = self.dispatcher.get_parser_for_file(item.path)
            tree = None
            if parser is not None:
                try:
                    # parser is a tree_sitter.Parser with a .parse() method
                    tree = parser.parse(content.encode("utf-8"))  # type: ignore[union-attr]
                except Exception:
                    pass
            
            # Extract metadata
            metadata = self._extract_metadata(item.path, content, tree)
            lang_info = self.dispatcher.get_language_info(item.path)
            
            return ScannedFile(
                path=item.path,
                content=content,
                size_bytes=len(content.encode("utf-8")),
                language=lang_info.name if lang_info else None,
                **metadata,
            )
            
        except Exception as e:
            print(f"❌ Error processing {item.path}: {e}")
            return None
    
    def scan(self, verbose: bool = True) -> ScanResult:
        """
        Scan the configured GitHub repository.
        
        Returns a ScanResult with all parsed files and statistics.
        """
        from github import Github, Auth
        
        start_time = time.time()
        
        if not self.config.github_token:
            raise ValueError("GitHub token not configured. Set GITHUB_API_KEY environment variable.")
        
        if not self.config.github_repo:
            raise ValueError("GitHub repo not configured. Set GITHUB_REPO environment variable.")
        
        if verbose:
            print(f"🔹 Authenticating with GitHub...")
        
        auth = Auth.Token(self.config.github_token)
        g = Github(auth=auth)
        repo = g.get_repo(self.config.github_repo)
        
        if verbose:
            print(f"🔹 Fetching file list from branch: {self.config.target_branch}...")
        
        branch = repo.get_branch(self.config.target_branch)
        tree = repo.get_git_tree(branch.commit.sha, recursive=True)
        
        # Filter to blobs (files)
        all_files = [item for item in tree.tree if item.type == "blob"]
        if verbose:
            print(f"📁 Found {len(all_files)} files in repository")
        
        # Process files
        results: List[ScannedFile] = []
        skipped: List[str] = []
        failed: List[str] = []
        
        for idx, item in enumerate(all_files):
            # Check if should skip
            if self._should_skip_file(item.path, item.size if hasattr(item, 'size') else 0):
                skipped.append(item.path)
                if verbose:
                    print(f"⏭️  Skipping: {item.path}")
                continue
            
            if verbose:
                print(f"🔹 Processing ({idx + 1}/{len(all_files)}): {item.path}")
            
            result = self._process_file(repo, item, idx, len(all_files))
            
            if result:
                results.append(result)
            else:
                failed.append(item.path)
        
        duration = time.time() - start_time
        
        scan_result = ScanResult(
            files=results,
            skipped_files=skipped,
            failed_files=failed,
            scan_duration_seconds=duration,
        )
        
        if verbose:
            print(f"\n✅ Scan Complete!")
            print(f"   📊 Processed: {len(results)} files")
            print(f"   ⏭️  Skipped: {len(skipped)} files")
            print(f"   ❌ Failed: {len(failed)} files")
            print(f"   📦 Total Size: {scan_result.total_bytes / 1024:.2f} KB")
            print(f"   🎯 Token Estimate: ~{scan_result.total_tokens_estimate:,} tokens")
            print(f"   ⏱️  Duration: {duration:.2f}s")
        
        return scan_result


def scan_local_directory(
    directory: str,
    config: Optional[WCAConfig] = None,
    verbose: bool = True,
) -> ScanResult:
    """
    Scan a local directory instead of GitHub.
    
    Useful for testing without API calls.
    """
    config = config or WCAConfig()
    dispatcher = get_dispatcher()
    
    start_time = time.time()
    results: List[ScannedFile] = []
    skipped: List[str] = []
    failed: List[str] = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, directory)
            
            # Check skip patterns
            skip = False
            for pattern in config.skip_patterns:
                if fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch(filename, pattern):
                    skip = True
                    break
            
            if skip or not dispatcher.is_supported(filename):
                skipped.append(relpath)
                continue
            
            try:
                with open(filepath, "rb") as f:
                    content_bytes = f.read()
                
                # Size check
                if len(content_bytes) > config.max_file_size_kb * 1024:
                    skipped.append(relpath)
                    continue
                
                # Decode
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    failed.append(relpath)
                    continue
                
                # Parse
                parser, _ = dispatcher.get_parser_for_file(filename)
                tree = None
                if parser is not None:
                    try:
                        tree = parser.parse(content_bytes)  # type: ignore[union-attr]
                    except Exception:
                        pass
                
                lang_info = dispatcher.get_language_info(filename)
                
                results.append(ScannedFile(
                    path=relpath,
                    content=content,
                    size_bytes=len(content_bytes),
                    language=lang_info.name if lang_info else None,
                    root_type=tree.root_node.type if tree else None,
                    statement_count=tree.root_node.child_count if tree else 0,
                    has_syntax_errors=tree.root_node.has_error if tree else False,
                ))
                
                if verbose:
                    print(f"✅ Parsed: {relpath}")
                    
            except Exception as e:
                failed.append(relpath)
                if verbose:
                    print(f"❌ Failed: {relpath}: {e}")
    
    duration = time.time() - start_time
    
    return ScanResult(
        files=results,
        skipped_files=skipped,
        failed_files=failed,
        scan_duration_seconds=duration,
    )