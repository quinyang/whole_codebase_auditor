"""Session 3 -- benchmark construction by injection.

Public vulnerability benchmarks (OWASP Benchmark, NIST Juliet) are single-file
and single-function. They cannot test the cross-file claim, which is the entire
thesis, so scoring well on them would be evidence of nothing.

Instead: take real repositories, plant a cross-file vulnerability into two files
that already reference each other, and record exactly where it went. Two
properties make the resulting number defensible:

  SEEDED      the same seed reproduces the same corpus, so a reported score can
              be re-derived by someone else.
  DISTANCE    each pattern is planted both at short and long separation in the
              packer's emitted order. This is the variable that actually tests
              whether graph ordering earns its complexity -- an SSM carries a
              fixed-size state, so distance between the two halves should cost
              recall, and the packer exists to shorten it.

Injection edits files in memory. Nothing on disk is touched, and the clean
version of every repo is retained as its own negative control.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from wca.graph import SymbolGraph
from wca.ingest import RepoBundle, SourceFile

PATTERNS = ("credential_leak", "taint_to_sink")


@dataclass
class PlantedSpot:
    file: str
    line: int  # 1-indexed
    code: str


@dataclass
class Planted:
    id: str
    pattern: str
    severity: str
    category: str
    spots: dict[str, PlantedSpot] = field(default_factory=dict)
    requires_files: list[str] = field(default_factory=list)
    distance: str = "unknown"  # near | far -- separation in the packed stream

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spots"] = {k: asdict(v) for k, v in self.spots.items()}
        return d


@dataclass
class BenchmarkCase:
    repo: str
    seed: int
    files: list[SourceFile]
    planted: list[Planted] = field(default_factory=list)
    clean_files: set[str] = field(default_factory=set)

    @property
    def is_negative_control(self) -> bool:
        return not self.planted

    def ground_truth(self) -> dict:
        return {
            "repo": self.repo,
            "seed": self.seed,
            "negative_control": self.is_negative_control,
            "planted": [p.to_dict() for p in self.planted],
            "clean_files": sorted(self.clean_files),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _append(sf: SourceFile, block: str) -> SourceFile:
    """Append a block to a file."""
    text = sf.text if sf.text.endswith("\n") else sf.text + "\n"
    new_text = text + block
    return SourceFile(path=sf.path, data=new_text.encode("utf-8"), text=new_text)


def _locate(sf: SourceFile, code: str) -> int:
    """1-indexed line containing `code`, searched from the end.

    Computing line numbers by counting the blank lines in an injected block is
    arithmetic that silently drifts whenever the block is edited -- it was
    off by one in three of four spots on first run. Searching for the line that
    is actually there cannot drift.
    """
    needle = code.strip()
    lines = sf.text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if needle in lines[i]:
            return i + 1
    raise AssertionError(f"injected line not found in {sf.path}: {code!r}")


def _pick_pair(
    graph: SymbolGraph, by_path: dict[str, SourceFile], rng: random.Random, distance: str
) -> tuple[str, str] | None:
    """Choose (definition_file, use_file) that already reference each other.

    Planting into files with a real import edge means the injected vulnerability
    sits on a genuine dependency path, not an artificial one -- otherwise the
    benchmark would only measure whether the model can spot two adjacent blocks
    of suspicious code.
    """
    edges = [
        (e.dst, e.src)  # dst defines, src uses
        for e in graph.edges
        if e.kind == "import"
        and e.src in by_path
        and e.dst in by_path
        and e.src.endswith(".py")
        and e.dst.endswith(".py")
    ]
    if not edges:
        py = [p for p in by_path if p.endswith(".py")]
        if len(py) < 2:
            return None
        a, b = rng.sample(py, 2)
        return a, b

    order = list(graph.symbols)
    pos = {p: i for i, p in enumerate(order)}
    edges.sort(key=lambda ab: abs(pos.get(ab[0], 0) - pos.get(ab[1], 0)))
    # near = tightest coupling in stream order; far = widest
    return edges[0] if distance == "near" else edges[-1]


# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #


def _inject_credential_leak(
    by_path: dict[str, SourceFile], defn: str, use: str, rng: random.Random
) -> Planted:
    token = f"sk-live-{rng.randrange(16**16):016x}"
    var = "_WCA_DB_PASSWORD"
    getter = "_wca_get_credential"

    block = (
        f'\n\n{var} = "{token}"\n\n\n'
        f"def {getter}():\n"
        f"    return {var}\n"
    )
    defn_code = f'{var} = "{token}"'
    by_path[defn] = _append(by_path[defn], block)
    defn_line = _locate(by_path[defn], defn_code)

    block2 = (
        f"\n\nimport logging as _wca_logging\n"
        f"from {_module_name(defn)} import {var}\n\n\n"
        f"def _wca_debug_dump():\n"
        f'    _wca_logging.getLogger(__name__).info("db credential=%s", {var})\n'
    )
    use_code = f'_wca_logging.getLogger(__name__).info("db credential=%s", {var})'
    by_path[use] = _append(by_path[use], block2)
    use_line = _locate(by_path[use], use_code)

    return Planted(
        id=f"credential_leak::{defn}->{use}",
        pattern="credential_leak",
        severity="critical",
        category="hardcoded_secret",
        spots={
            "definition": PlantedSpot(defn, defn_line, defn_code),
            "use": PlantedSpot(use, use_line, use_code),
        },
        requires_files=sorted({defn, use}),
    )


def _inject_taint_to_sink(
    by_path: dict[str, SourceFile], sink_file: str, source_file: str, rng: random.Random
) -> Planted:
    sink = f"_wca_run_query_{rng.randrange(1000):03d}"

    block = (
        f"\n\nimport sqlite3 as _wca_sqlite\n\n\n"
        f"def {sink}(sql):\n"
        f'    conn = _wca_sqlite.connect(":memory:")\n'
        f"    return conn.cursor().execute(sql).fetchall()\n"
    )
    sink_code = "return conn.cursor().execute(sql).fetchall()"
    by_path[sink_file] = _append(by_path[sink_file], block)
    sink_line = _locate(by_path[sink_file], sink_code)

    block2 = (
        f"\n\nfrom {_module_name(sink_file)} import {sink}\n\n\n"
        f"def _wca_report_handler(request):\n"
        f'    table = request.args.get("table")\n'
        f'    return {sink}("SELECT * FROM " + table)\n'
    )
    source_code = f'return {sink}("SELECT * FROM " + table)'
    by_path[source_file] = _append(by_path[source_file], block2)
    source_line = _locate(by_path[source_file], source_code)

    return Planted(
        id=f"taint_to_sink::{source_file}->{sink_file}",
        pattern="taint_to_sink",
        severity="high",
        category="injection",
        spots={
            "sink": PlantedSpot(sink_file, sink_line, sink_code),
            "source": PlantedSpot(source_file, source_line, source_code),
        },
        requires_files=sorted({sink_file, source_file}),
    )


_INJECTORS = {
    "credential_leak": _inject_credential_leak,
    "taint_to_sink": _inject_taint_to_sink,
}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def inject(
    bundle: RepoBundle,
    graph: SymbolGraph,
    *,
    seed: int,
    patterns: tuple[str, ...] = PATTERNS,
    distance: str = "near",
) -> BenchmarkCase:
    """Plant `patterns` into `bundle`, returning modified files + ground truth.

    Pass `patterns=()` to produce a negative control: the repo untouched, with
    every file recorded as clean. Precision is unmeasurable without these, so
    they are a first-class output rather than an afterthought.
    """
    rng = random.Random(seed)
    by_path = {f.path: f for f in bundle.files}
    case = BenchmarkCase(repo=bundle.name, seed=seed, files=[], planted=[])

    if not patterns:
        case.files = list(bundle.files)
        case.clean_files = set(by_path)
        return case

    touched: set[str] = set()
    for pattern in patterns:
        pair = _pick_pair(graph, by_path, rng, distance)
        if pair is None:
            continue
        defn, use = pair
        if defn in touched or use in touched:
            # keep each planted vulnerability in its own pair of files, so a
            # finding can be attributed to exactly one of them
            remaining = [p for p in by_path if p.endswith(".py") and p not in touched]
            if len(remaining) < 2:
                continue
            defn, use = rng.sample(remaining, 2)
        planted = _INJECTORS[pattern](by_path, defn, use, rng)
        planted.distance = distance
        case.planted.append(planted)
        touched.update({defn, use})

    case.files = [by_path[p] for p in sorted(by_path)]
    case.clean_files = set(by_path) - touched
    return case


def verify(case: BenchmarkCase) -> list[str]:
    """Check every recorded line actually contains the recorded code.

    A wrong line number silently corrupts every score computed from this corpus,
    and the error is invisible in the final number. Cheap to check, so check.
    """
    by_path = {f.path: f.text.splitlines() for f in case.files}
    problems: list[str] = []
    for planted in case.planted:
        for key, spot in planted.spots.items():
            lines = by_path.get(spot.file)
            if lines is None:
                problems.append(f"{planted.id}: {spot.file} not in case")
                continue
            if not (1 <= spot.line <= len(lines)):
                problems.append(f"{planted.id}/{key}: line {spot.line} out of range")
                continue
            actual = lines[spot.line - 1].strip()
            if spot.code.strip() not in actual:
                problems.append(
                    f"{planted.id}/{key}: {spot.file}:{spot.line} is {actual!r}, "
                    f"expected {spot.code!r}"
                )
    return problems
