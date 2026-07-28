"""Stage 1 -- Ingest.

Produces a list of `SourceFile` from either a GitHub repo or a local directory.

Why a tarball instead of the GitHub API: the previous implementation issued one
`get_git_blob` call per file plus a 50ms sleep, so a 500-file repo cost ~500
authenticated requests and ~25s of pure sleeping, and ran into secondary rate
limits. `codeload.github.com` serves the whole tree as one gzip stream with no
token required for public repos.
"""

from __future__ import annotations

import fnmatch
import io
import os
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CODELOAD = "https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"

# Directories that are never the user's own code. Matched on any path component.
SKIP_DIRS = frozenset(
    {
        ".git", ".github", ".idea", ".vscode", "__pycache__", "node_modules",
        "vendor", "third_party", "thirdparty", "external", "deps", "dist",
        "build", "out", "target", "site-packages", ".venv", "venv", "env",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "migrations",
    }
)

# Glob patterns for generated / vendored / lock files.
SKIP_GLOBS = (
    "*.lock", "*-lock.json", "*.min.js", "*.min.css", "*.map",
    "*json.hpp", "*.pb.go", "*.pb.py", "*_pb2.py", "*_pb2_grpc.py",
    "*.generated.*", "*.g.dart", "*.designer.cs",
)

DEFAULT_MAX_FILE_BYTES = 200_000  # a single 1MB amalgamated header eats the budget
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024


@dataclass
class SourceFile:
    """One text file from the target repository."""

    path: str  # repo-relative, posix separators
    data: bytes
    text: str

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower()


@dataclass
class RepoBundle:
    """Everything ingest knows about the target."""

    name: str
    ref: str
    files: list[SourceFile] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def summary(self) -> str:
        skips = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"
        return (
            f"{self.name}@{self.ref}: {len(self.files)} files, "
            f"{self.total_bytes / 1024:.1f} KB kept (skipped: {skips})"
        )


def _should_skip(path: str, size: int, max_file_bytes: int) -> str | None:
    """Return a skip reason, or None to keep the file."""
    parts = path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return "vendor_dir"
    name = parts[-1]
    if any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(path, g) for g in SKIP_GLOBS):
        return "generated"
    if size > max_file_bytes:
        return "too_large"
    if size == 0:
        return "empty"
    return None


def _decode(data: bytes) -> str | None:
    """Decode as UTF-8, or return None if the file is binary."""
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


def parse_repo_spec(spec: str) -> tuple[str, str]:
    """Accept 'owner/repo', a full GitHub URL, or a .git URL."""
    s = spec.strip().removesuffix(".git")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"cannot parse repo spec: {spec!r} (want 'owner/repo')")
    return parts[0], parts[1]


def from_github(
    spec: str,
    ref: str = "main",
    *,
    token: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    timeout: int = 120,
) -> RepoBundle:
    """Download the repo as a single tarball and extract text files from the stream."""
    owner, repo = parse_repo_spec(spec)
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY")

    last_err: Exception | None = None
    for candidate in _ref_candidates(ref):
        url = CODELOAD.format(owner=owner, repo=repo, ref=candidate)
        req = urllib.request.Request(url, headers={"User-Agent": "wca/0.2"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                blob = resp.read(max_total_bytes + 1)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (404, 422):  # wrong ref, try the next candidate
                continue
            raise
        if len(blob) > max_total_bytes:
            raise ValueError(f"tarball exceeds max_total_bytes ({max_total_bytes})")
        bundle = _bundle_from_tar(blob, f"{owner}/{repo}", candidate, max_file_bytes)
        return bundle

    raise RuntimeError(f"could not fetch {owner}/{repo} at ref {ref!r}: {last_err}")


def _ref_candidates(ref: str) -> list[str]:
    """codeload accepts refs/heads/X, refs/tags/X, or a raw SHA. Try in order."""
    if ref.startswith("refs/") or (len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)):
        return [ref]
    return [f"refs/heads/{ref}", f"refs/tags/{ref}", ref]


def _bundle_from_tar(blob: bytes, name: str, ref: str, max_file_bytes: int) -> RepoBundle:
    bundle = RepoBundle(name=name, ref=ref)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # GitHub wraps everything in '<repo>-<sha>/'; strip that component.
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not rel:
                continue
            reason = _should_skip(rel, member.size, max_file_bytes)
            if reason:
                bundle.skipped[reason] = bundle.skipped.get(reason, 0) + 1
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            data = fh.read()
            text = _decode(data)
            if text is None:
                bundle.skipped["binary"] = bundle.skipped.get("binary", 0) + 1
                continue
            bundle.files.append(SourceFile(path=rel, data=data, text=text))
    return bundle


def from_local(
    root: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> RepoBundle:
    """Walk a local directory. Same filtering rules as the tarball path."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    bundle = RepoBundle(name=root.name, ref="local")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            fp = Path(dirpath) / fn
            rel = fp.relative_to(root).as_posix()
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            reason = _should_skip(rel, size, max_file_bytes)
            if reason:
                bundle.skipped[reason] = bundle.skipped.get(reason, 0) + 1
                continue
            try:
                data = fp.read_bytes()
            except OSError:
                continue
            text = _decode(data)
            if text is None:
                bundle.skipped["binary"] = bundle.skipped.get("binary", 0) + 1
                continue
            bundle.files.append(SourceFile(path=rel, data=data, text=text))
    bundle.files.sort(key=lambda f: f.path)
    return bundle


def ingest(target: str, ref: str = "main", **kw) -> RepoBundle:
    """Dispatch on whether `target` is a local path or a GitHub spec."""
    p = Path(target).expanduser()
    if p.exists() and p.is_dir():
        return from_local(p, max_file_bytes=kw.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    return from_github(target, ref, **kw)
