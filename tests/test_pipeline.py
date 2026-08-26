"""CPU-only tests for ingest -> parse -> graph -> pack.

The manifest tests are the important ones. If offset resolution is wrong, every
finding the model produces points at the wrong line, and the eval number is
meaningless. These assert it exhaustively rather than by spot check.

Run: pytest -q
"""

from __future__ import annotations

import io
import tarfile

import pytest

from wca.findings import extract_json_array, parse_findings
from wca.graph import build_graph
from wca.ingest import _bundle_from_tar, from_local, parse_repo_spec
from wca.pack import pack
from wca.parse import LanguageDispatcher, parse_files

CONFIG_PY = '''\
import os

DB_PASSWORD = "admin_password_123"
DB_URL = "postgresql://svc:admin_password_123@prod-db.internal:5432/app"


def get_conn_string():
    return DB_URL
'''

HANDLERS_PY = '''\
import logging
from lib.config import DB_PASSWORD, get_conn_string

from lib.db import execute_raw

logger = logging.getLogger(__name__)


def debug_dump(request):
    logger.info("connecting with %s / %s", get_conn_string(), DB_PASSWORD)
    return {"ok": True}


def run_report(request):
    return execute_raw("SELECT * FROM " + request.args.get("table"))
'''

DB_PY = '''\
import sqlite3
from lib.config import get_conn_string


def execute_raw(sql):
    conn = sqlite3.connect(get_conn_string())
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()
'''

UTIL_JS = '''\
import { helper } from "../lib/helper.js";
const TOKEN = "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
export function render(input) {
  document.body.innerHTML = helper(input);
}
'''

HELPER_JS = 'export function helper(x) { return "<div>" + x + "</div>"; }\n'


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "lib" / "config.py").write_text(CONFIG_PY)
    (tmp_path / "lib" / "db.py").write_text(DB_PY)
    (tmp_path / "lib" / "helper.js").write_text(HELPER_JS)
    (tmp_path / "app" / "handlers.py").write_text(HANDLERS_PY)
    (tmp_path / "app" / "util.js").write_text(UTIL_JS)
    # noise that must be filtered out
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("var x = 1;\n")
    (tmp_path / "package-lock.json").write_text('{"a": 1}\n')
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return tmp_path


@pytest.fixture
def built(repo):
    bundle = from_local(repo)
    parsed = parse_files(bundle.files, LanguageDispatcher())
    graph = build_graph(parsed.files)
    return bundle, parsed, graph


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec",
    [
        "pallets/flask",
        "https://github.com/pallets/flask",
        "https://github.com/pallets/flask.git",
        "git@github.com:pallets/flask.git",
    ],
)
def test_parse_repo_spec(spec):
    assert parse_repo_spec(spec) == ("pallets", "flask")


def test_local_ingest_filters_noise(built):
    bundle, _, _ = built
    paths = {f.path for f in bundle.files}
    assert paths == {
        "lib/config.py",
        "lib/db.py",
        "lib/helper.js",
        "app/handlers.py",
        "app/util.js",
    }


def test_tarball_strips_github_prefix_and_filters():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [
            ("repo-abc123/main.py", b"import os\n"),
            ("repo-abc123/node_modules/x.js", b"var a=1;\n"),
            ("repo-abc123/yarn.lock", b"lockfile\n"),
            ("repo-abc123/img.png", b"\x89PNG\r\n\x1a\n\x00\x00"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    bundle = _bundle_from_tar(buf.getvalue(), "o/r", "refs/heads/main", 200_000)
    assert [f.path for f in bundle.files] == ["main.py"]


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #


def test_all_files_parse(built):
    _, parsed, _ = built
    assert len(parsed.files) == 5
    assert not parsed.failed
    assert {f.lang for f in parsed.files} == {"python", "javascript"}


def test_bad_syntax_does_not_abort_scan(tmp_path):
    """Regression: the old loop `return`ed on error, killing the whole scan."""
    (tmp_path / "broken.py").write_text("def f(:\n  ???\n")
    (tmp_path / "fine.py").write_text("def g():\n    return 1\n")
    bundle = from_local(tmp_path)
    parsed = parse_files(bundle.files, LanguageDispatcher())
    assert {f.path for f in parsed.files} == {"broken.py", "fine.py"}
    assert any(f.has_error for f in parsed.files)


def test_dispatcher_loads_grammars_on_modern_tree_sitter():
    """Regression: Parser(module.language()) raises on tree-sitter >= 0.25."""
    d = LanguageDispatcher()
    for fn in ("a.py", "a.c", "a.cpp", "a.js", "a.go", "a.java", "a.rs"):
        parser, _lang = d.get_parser_for_file(fn)
        assert parser is not None, fn
        assert parser.parse(b"") is not None


def test_unsupported_extension_is_skipped_not_raised():
    d = LanguageDispatcher()
    assert d.get_parser_for_file("README.md") == (None, None)
    assert d.get_parser_for_file("Dockerfile") == (None, None)


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #


def test_graph_finds_the_cross_file_secret_link(built):
    _, _, g = built
    edges = {(e.src, e.dst, e.kind) for e in g.edges}
    # secret defined in lib/config.py, logged in app/handlers.py
    assert ("app/handlers.py", "lib/config.py", "import") in edges
    assert ("app/handlers.py", "lib/config.py", "call") in edges
    # tainted input reaches the sink in lib/db.py
    assert ("app/handlers.py", "lib/db.py", "call") in edges


def test_relative_js_import_resolves(built):
    _, _, g = built
    assert ("app/util.js", "lib/helper.js", "import") in {
        (e.src, e.dst, e.kind) for e in g.edges
    }


def test_secrets_and_sinks_detected(built):
    _, _, g = built
    assert g.symbols["lib/config.py"].secretish
    assert g.symbols["app/util.js"].secretish
    assert "execute" in g.symbols["lib/db.py"].danger_hits
    assert g.symbols["lib/config.py"].risk_score > g.symbols["lib/helper.js"].risk_score


def test_ambiguous_call_names_produce_no_edge(tmp_path):
    """A name defined in many files carries no linking information."""
    for i in range(3):
        (tmp_path / f"m{i}.py").write_text("def shared():\n    return 1\n")
    (tmp_path / "caller.py").write_text("def go():\n    return shared()\n")
    bundle = from_local(tmp_path)
    g = build_graph(parse_files(bundle.files, LanguageDispatcher()).files)
    assert not [e for e in g.edges if e.kind == "call" and e.detail == "shared"]


# --------------------------------------------------------------------------- #
# pack -- budget
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", [4_000, 8_000, 16_000, 32_000])
def test_never_exceeds_budget(built, budget):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=budget)
    assert p.used_tokens <= budget


def test_segments_are_ordered_and_non_overlapping(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=16_000)
    for a, b in zip(p.segments, p.segments[1:]):
        assert a.char_start <= b.char_start
        assert a.char_end <= b.char_start


def test_high_risk_files_survive_a_tight_budget(built):
    """The point of the priority pass: secrets must not be evicted first."""
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=3_500)
    full = {s.path for s in p.segments if s.mode == "full"}
    assert "lib/config.py" in full


def test_signature_tier_is_reached_on_a_large_repo(tmp_path):
    """Regression: greedy emission in stream order never demoted anything."""
    for i in range(200):
        (tmp_path / f"m{i}.py").write_text(
            f"import m{max(0, i - 1)}\n" + "".join(
                f"def f{i}_{j}(x):\n    return x + {j}\n\n" for j in range(8)
            )
        )
    (tmp_path / "secrets.py").write_text('API_KEY = "sk-live-deadbeefcafe0123456789"\n')
    bundle = from_local(tmp_path)
    parsed = parse_files(bundle.files, LanguageDispatcher())
    g = build_graph(parsed.files)
    p = pack(parsed.files, g, budget_tokens=16_000)
    assert p.n_signature > 0, "signature tier never used"
    assert "secrets.py" in {s.path for s in p.segments if s.mode == "full"}


# --------------------------------------------------------------------------- #
# pack -- manifest (the correctness property that everything downstream needs)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", [4_000, 16_000, 32_000])
def test_every_offset_resolves_to_the_correct_source_line(built, budget):
    bundle, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=budget)
    originals = {f.path: f.text.splitlines() for f in bundle.files}

    checked = 0
    for seg in p.segments:
        for off in range(seg.char_start, seg.char_end):
            resolved = p.resolve(off)
            assert resolved is not None, f"{seg.path}@{off} did not resolve"
            path, line = resolved
            assert path == seg.path
            ch = p.text[off]
            if ch == "\n" or "elided" in p.text[max(0, off - 30) : off + 30]:
                continue
            src = originals[path]
            assert 1 <= line <= len(src), f"{path}:{line} out of range"
            assert ch in src[line - 1], f"{path}:{line} char {ch!r} not in {src[line - 1]!r}"
            checked += 1
    assert checked > 100


def test_resolve_snippet_grounds_real_evidence_lines(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    assert p.resolve_snippet('DB_PASSWORD = "admin_password_123"') == ("lib/config.py", 3)
    assert p.resolve_snippet("cur.execute(sql)") == ("lib/db.py", 8)
    assert p.resolve_snippet("this text is not in the repo at all") is None


@pytest.mark.parametrize(
    "evidence",
    [
        'DB_PASSWORD = "admin_password_123"',  # exact
        "DB_PASSWORD = 'admin_password_123'",  # quote style swapped
        '`DB_PASSWORD = "admin_password_123"`',  # markdown backticks
        "DB_PASSWORD = “admin_password_123”",  # smart quotes
        'DB_PASSWORD   =   "admin_password_123"',  # whitespace reflowed
    ],
)
def test_grounding_survives_model_requoting(built, evidence):
    """Regression: a model re-quoting a line with different quote style is still
    quoting real code. Exact substring matching alone scored correct findings as
    hallucinations, which is the expensive direction to get wrong."""
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    assert p.resolve_snippet(evidence) == ("lib/config.py", 3)


@pytest.mark.parametrize(
    "evidence",
    [
        "os.system(user_input)  # nowhere in this repo",
        "subprocess.call(shell=True, cmd=user_data)",
        "return None",  # too generic to be evidence
        "   ",
    ],
)
def test_grounding_still_rejects_hallucinations(built, evidence):
    """Loosening the match must not make grounding meaningless."""
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    assert p.resolve_snippet(evidence) is None


def test_resolve_rejects_out_of_range_offsets(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=16_000)
    assert p.resolve(-1) is None
    assert p.resolve(len(p.text) + 10) is None


def test_manifest_round_trips_as_json(built, tmp_path):
    import json

    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=16_000)
    out = tmp_path / "m.json"
    p.write_manifest(str(out))
    m = json.loads(out.read_text())
    assert m["counts"]["full"] + m["counts"]["signature"] == len(p.segments)


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #


def test_extract_json_handles_fences_and_prose():
    assert extract_json_array('```json\n[{"title": "a"}]\n```')[0]["title"] == "a"
    assert extract_json_array('Sure! Here you go:\n[{"title": "b"}]\nHope that helps.')
    assert extract_json_array("no json here") == []


def test_extract_json_salvages_a_truncated_array():
    truncated = '[{"title": "a", "severity": "high"}, {"title": "b", "severity": "low"'
    assert len(extract_json_array(truncated)) == 1


def test_hallucinated_paths_are_stripped_and_flagged(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = """[{
      "title": "Fabricated", "severity": "critical", "category": "injection",
      "files": ["does/not/exist.py"], "evidence": "nothing like this exists",
      "why_cross_file": "made up", "confidence": 0.9
    }]"""
    f = parse_findings(raw, p)[0]
    assert f.files == []
    assert not f.grounded
    assert any("not in stream" in n for n in f.notes)


def test_real_evidence_is_grounded_to_file_and_line(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = """[{
      "title": "Credential logged across module boundary",
      "severity": "critical", "category": "hardcoded_secret",
      "files": ["lib/config.py", "app/handlers.py"],
      "evidence": "DB_PASSWORD = \\"admin_password_123\\"",
      "why_cross_file": "defined in config, logged in handlers",
      "confidence": 0.85
    }]"""
    f = parse_findings(raw, p)[0]
    assert f.grounded
    assert f.location == "lib/config.py:3"
    assert f.is_cross_file
