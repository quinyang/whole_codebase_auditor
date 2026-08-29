"""CPU-only tests for ingest -> parse -> graph -> pack.

The manifest tests are the important ones. If offset resolution is wrong, every
finding the model produces points at the wrong line, and the eval number is
meaningless. These assert it exhaustively rather than by spot check.

Run: pytest -q
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile

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
    """Both objects survive: the complete one, plus the cut-off one repaired.

    Previously only the complete object was kept. A generation stopped by
    max_new_tokens leaves real fields in the final object; dropping them loses
    recall for a formatting reason.
    """
    truncated = '[{"title": "a", "severity": "high"}, {"title": "b", "severity": "low"'
    titles = {o["title"] for o in extract_json_array(truncated)}
    assert titles == {"a", "b"}


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


# --------------------------------------------------------------------------- #
# fixture + sweep (CPU-testable parts)
# --------------------------------------------------------------------------- #


def test_embedded_fixture_matches_disk():
    """wca/fixtures.py is generated from eval/fixtures/toy_vuln; keep them in sync."""
    import pathlib

    from wca.fixtures import GROUND_TRUTH, TOY_VULN

    root = pathlib.Path(__file__).parent.parent / "eval" / "fixtures" / "toy_vuln"
    if not root.exists():  # installed as a wheel, no eval/ dir
        pytest.skip("source checkout only")
    on_disk = {
        p.relative_to(root).as_posix(): p.read_text()
        for p in root.rglob("*")
        if p.is_file() and p.name != "GROUND_TRUTH.json"
    }
    assert TOY_VULN == on_disk, "regenerate wca/fixtures.py from eval/fixtures/toy_vuln"
    assert json.loads((root / "GROUND_TRUTH.json").read_text()) == GROUND_TRUTH


def test_ground_truth_line_numbers_are_exact():
    """A wrong line number here silently corrupts every score."""
    from wca.fixtures import GROUND_TRUTH, materialise_toy_vuln

    root = materialise_toy_vuln(tempfile.mkdtemp())
    for planted in GROUND_TRUTH["planted"]:
        for key in ("definition", "use", "source", "sink"):
            if key not in planted:
                continue
            spot = planted[key]
            actual = (root / spot["file"]).read_text().splitlines()[spot["line"] - 1].strip()
            assert spot["code"].strip() in actual or actual in spot["code"].strip(), (
                f"{spot['file']}:{spot['line']} is {actual!r}, expected {spot['code']!r}"
            )


def test_toy_fixture_graph_finds_both_planted_chains():
    from wca.fixtures import materialise_toy_vuln

    root = materialise_toy_vuln(tempfile.mkdtemp())
    parsed = parse_files(from_local(root).files, LanguageDispatcher())
    g = build_graph(parsed.files)
    edges = {(e.src, e.dst) for e in g.edges}
    assert ("app/handlers.py", "lib/config.py") in edges  # credential leak
    assert ("app/handlers.py", "lib/db.py") in edges  # taint -> sink


def test_toy_fixture_evidence_lines_all_ground():
    """Every planted line must resolve through the manifest, or scoring is broken."""
    from wca.fixtures import GROUND_TRUTH, materialise_toy_vuln

    root = materialise_toy_vuln(tempfile.mkdtemp())
    parsed = parse_files(from_local(root).files, LanguageDispatcher())
    g = build_graph(parsed.files)
    p = pack(parsed.files, g, budget_tokens=8_000, repo_name="toy_vuln")
    assert "GROUND_TRUTH" not in p.text, "answer key must not reach the model"
    for planted in GROUND_TRUTH["planted"]:
        for key in ("definition", "use", "source", "sink"):
            if key not in planted:
                continue
            spot = planted[key]
            assert p.resolve_snippet(spot["code"]) == (spot["file"], spot["line"])


def test_audit_once_records_oom_instead_of_raising():
    """A failing budget must become a data point, not kill the sweep."""
    from wca.fixtures import materialise_toy_vuln
    from wca.sweep import audit_once

    root = materialise_toy_vuln(tempfile.mkdtemp())
    parsed = parse_files(from_local(root).files, LanguageDispatcher())
    g = build_graph(parsed.files)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * (len(text) // 4)}

    class Boom:
        tokenizer = FakeTokenizer()

        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory. Tried to allocate 10.92 GiB")

    row = audit_once(Boom(), parsed.files, g, 4_000, repo_name="toy")
    assert row.error.startswith("RuntimeError")
    assert "out of memory" in row.error
    assert row.n_findings == 0
    assert row.grounding_rate == 0.0


@pytest.mark.parametrize(
    "evidence",
    [
        "# In my_script.sh:\nexport SECRET_TOKEN='supersecrettoken1234567890'\n# then used",
        "# In core.py:\n# The following function signature shows how flags can be set",
        "For example, the password would look like this",
        "// example: pass the token here",
        "some_call(arg)...",
    ],
)
def test_authored_examples_are_flagged_as_fabricated(evidence):
    """Observed on pallets/click: the model wrote tutorial snippets, inventing a
    file and a secret, instead of quoting the stream. Distinguishing this from a
    near-miss matters -- one is a prompt bug, the other a matcher bug."""
    from wca.findings import looks_fabricated

    assert looks_fabricated(evidence)


@pytest.mark.parametrize(
    "evidence",
    [
        'DB_PASSWORD = "admin_password_123"',
        "cur.execute(sql)",
        "logger.info('connecting with %s', DB_PASSWORD)",
        "return execute_raw('SELECT * FROM ' + table)",
        "",
    ],
)
def test_real_code_lines_are_not_flagged_as_fabricated(evidence):
    from wca.findings import looks_fabricated

    assert not looks_fabricated(evidence)


def test_fabricated_evidence_gets_a_distinct_note(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = json.dumps([{
        "title": "Leaked token", "severity": "high", "category": "hardcoded_secret",
        "files": ["lib/config.py"],
        "evidence": "# In my_script.sh:\nexport SECRET_TOKEN='supersecret1234567890'",
        "why_cross_file": "n/a", "confidence": 1.0,
    }])
    f = parse_findings(raw, p)[0]
    assert not f.grounded
    assert any("AUTHORED" in n for n in f.notes)


# --------------------------------------------------------------------------- #
# lenient JSON parsing -- models do not reliably emit strict JSON
# --------------------------------------------------------------------------- #

FALCON_SINGLE_QUOTED = """```json
[
    {
        "title": "Hardcoded Credential Exposed via Log Message",
        "severity": "high",
        "category": "injection",
        "files": ["lib/db.py", "app/handlers.py"],
        "evidence": 'logger.info("connecting with %s", DB_PASSWORD)',
        "why_cross_file": "defined in config, logged in handlers",
        "confidence": 1.0
    },
    {
        "title": "Untrusted Input Reaches Database Sink",
        "severity": "high",
        "category": "injection",
        "files": ["lib/db.py", "app/handlers.py"],
        "evidence": 'execute_raw("SELECT * FROM " + table)',
        "why_cross_file": "user data reaches execute_raw",
"""


def test_python_style_quotes_are_parsed():
    """Observed from Falcon3-Mamba: correct findings emitted with Python string
    syntax. Strict json.loads discarded every one of them, scoring the model's
    best work as a total miss."""
    objs = extract_json_array(FALCON_SINGLE_QUOTED)
    assert len(objs) == 2
    assert objs[0]["evidence"] == 'logger.info("connecting with %s", DB_PASSWORD)'
    assert objs[1]["evidence"] == 'execute_raw("SELECT * FROM " + table)'


def test_truncated_final_object_is_recovered():
    """max_new_tokens cutting off the last object must not discard it."""
    objs = extract_json_array(FALCON_SINGLE_QUOTED)
    assert any("Untrusted Input" in o["title"] for o in objs)


def test_json_literals_map_to_python():
    raw = '[{"title": "a", "ok": true, "bad": false, "x": null, "evidence": \'y\'}]'
    objs = extract_json_array(raw)
    assert objs and objs[0]["ok"] is True and objs[0]["x"] is None


def test_trailing_commas_tolerated():
    raw = '[{"title": "a", "severity": "high",}, {"title": "b",},]'
    assert len(extract_json_array(raw)) == 2


def test_apostrophe_inside_double_quoted_value_survives():
    """The scanner tracks both quote chars; it must not treat ' as an opener
    when it appears inside a normal JSON string."""
    raw = '[{"title": "it\'s fine", "evidence": "x = 1"}]'
    objs = extract_json_array(raw)
    assert objs and objs[0]["title"] == "it's fine"


def test_strict_json_still_preferred():
    raw = '[{"title": "a", "evidence": "b"}]'
    assert extract_json_array(raw) == [{"title": "a", "evidence": "b"}]


def test_prose_without_json_yields_nothing():
    assert extract_json_array("I could not find any vulnerabilities.") == []


def test_lenient_parsing_grounds_real_evidence(built):
    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = (
        '[{"title": "leak", "severity": "critical", "category": "hardcoded_secret",'
        ' "files": ["lib/config.py"],'
        " \"evidence\": 'DB_PASSWORD = \"admin_password_123\"',"
        ' "why_cross_file": "x", "confidence": 1.0}]'
    )
    f = parse_findings(raw, p)[0]
    assert f.grounded
    assert f.location == "lib/config.py:3"


# --------------------------------------------------------------------------- #
# graph-based attribution repair
# --------------------------------------------------------------------------- #


def test_graph_repairs_wrong_file_attribution(built):
    """Observed: the model produced the correct evidence line but named the
    wrong counterpart file. It is reliable about where it is looking and
    unreliable about what it is looking at, so derive the counterpart from the
    symbol graph instead of trusting it."""
    from wca.findings import enrich_with_graph

    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = (
        '[{"title": "leak", "severity": "critical", "category": "hardcoded_secret",'
        ' "files": ["lib/db.py"],'  # wrong counterpart
        " \"evidence\": 'logger.info(\"connecting with %s / %s\", get_conn_string(), DB_PASSWORD)',"
        ' "why_cross_file": "x", "confidence": 1.0}]'
    )
    f = enrich_with_graph(parse_findings(raw, p), g)[0]
    assert f.grounded
    # get_conn_string is defined only in lib/config.py -> that is the counterpart
    assert "lib/config.py" in f.files
    assert any("symbol graph" in n for n in f.notes)


def test_graph_enrichment_ignores_ambiguous_symbols(tmp_path):
    """A name defined in several files identifies nothing; adding all of them
    would manufacture cross-file findings out of noise."""
    from wca.findings import enrich_with_graph

    for i in range(3):
        (tmp_path / f"m{i}.py").write_text("def shared():\n    return 1\n")
    (tmp_path / "caller.py").write_text("def go():\n    return shared()\n")
    parsed = parse_files(from_local(tmp_path).files, LanguageDispatcher())
    g = build_graph(parsed.files)
    p = pack(parsed.files, g, budget_tokens=16_000)
    raw = json.dumps([{
        "title": "t", "severity": "low", "category": "other",
        "files": ["caller.py"], "evidence": "return shared()",
        "why_cross_file": "x", "confidence": 0.5,
    }])
    f = enrich_with_graph(parse_findings(raw, p), g)[0]
    assert not any(p_ in f.files for p_ in ("m0.py", "m1.py", "m2.py"))


def test_grounded_only_scoring_discards_hallucinations(built):
    """The precision mechanism: ungrounded findings never reach the score."""
    from wca.sweep import score_against_ground_truth

    _, parsed, g = built
    p = pack(parsed.files, g, budget_tokens=32_000)
    raw = json.dumps([{
        "title": "invented", "severity": "high", "category": "injection",
        "files": ["lib/config.py", "lib/db.py"],
        "evidence": '["--help", "--show-vars="].append(arg)',  # not in this repo
        "why_cross_file": "x", "confidence": 0.95,
    }])
    findings = parse_findings(raw, p)
    assert findings and not findings[0].grounded
    scored = score_against_ground_truth(findings, grounded_only=True)
    assert scored["identified"] == 0
    assert scored["false_positives"] == 0


# --------------------------------------------------------------------------- #
# benchmark injection (session 3)
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_bundle():
    from wca.fixtures import materialise_toy_vuln

    root = materialise_toy_vuln(tempfile.mkdtemp())
    bundle = from_local(root)
    graph = build_graph(parse_files(bundle.files, LanguageDispatcher()).files)
    return bundle, graph


@pytest.mark.parametrize("distance", ["near", "far"])
def test_injected_lines_are_recorded_exactly(clean_bundle, distance):
    """A wrong line number silently corrupts every score derived from the corpus
    and is invisible in the final number. It was off by one on first run."""
    from wca.inject import inject, verify

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42, distance=distance)
    assert case.planted
    assert verify(case) == []


@pytest.mark.parametrize("distance", ["near", "far"])
def test_injected_lines_ground_through_the_manifest(clean_bundle, distance):
    """If a planted line cannot be resolved, recall is unmeasurable for it."""
    from wca.inject import inject

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42, distance=distance)
    parsed = parse_files(case.files, LanguageDispatcher())
    g2 = build_graph(parsed.files)
    p = pack(parsed.files, g2, budget_tokens=16_000, repo_name="injected")
    for planted in case.planted:
        for spot in planted.spots.values():
            assert p.resolve_snippet(spot.code) == (spot.file, spot.line)


def test_injection_creates_real_cross_file_edges(clean_bundle):
    """The planted halves must actually be linked, or the case tests nothing
    about cross-file reasoning."""
    from wca.inject import inject

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42)
    g2 = build_graph(parse_files(case.files, LanguageDispatcher()).files)
    edges = {(e.src, e.dst) for e in g2.edges}
    for planted in case.planted:
        a, b = planted.requires_files
        assert (a, b) in edges or (b, a) in edges


def test_negative_control_is_untouched(clean_bundle):
    """Precision is unmeasurable without repos that contain nothing."""
    from wca.inject import inject

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=1, patterns=())
    assert case.is_negative_control
    assert not case.planted
    assert case.clean_files == {f.path for f in bundle.files}
    assert [f.text for f in case.files] == [f.text for f in bundle.files]


def test_injection_is_reproducible_from_seed(clean_bundle):
    """A score nobody can re-derive is not a result."""
    from wca.inject import inject

    bundle, g = clean_bundle
    a = inject(bundle, g, seed=7)
    b = inject(bundle, g, seed=7)
    c = inject(bundle, g, seed=8)
    assert a.ground_truth() == b.ground_truth()
    assert [f.text for f in a.files] == [f.text for f in b.files]
    assert a.ground_truth() != c.ground_truth()


def test_verify_catches_a_corrupted_ground_truth(clean_bundle):
    from wca.inject import inject, verify

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42)
    case.planted[0].spots["definition"].line += 3
    assert verify(case)


# --------------------------------------------------------------------------- #
# benchmark scoring
# --------------------------------------------------------------------------- #


def _audit_fake(case, model_json):
    from wca.findings import enrich_with_graph

    parsed = parse_files(case.files, LanguageDispatcher())
    g = build_graph(parsed.files)
    p = pack(parsed.files, g, budget_tokens=16_000, repo_name="x")
    return enrich_with_graph(parse_findings(model_json, p), g)


def _quote(spot, i=0):
    return json.dumps({
        "title": f"f{i}", "severity": "high", "category": "injection",
        "files": [spot.file], "evidence": spot.code,
        "why_cross_file": "x", "confidence": 0.9,
    })


def test_scoring_credits_planted_vulnerabilities(clean_bundle):
    from wca.benchmark import score_case
    from wca.inject import inject

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42, distance="near")
    spots = [s for p in case.planted for s in p.spots.values()]
    raw = "[" + ",".join(_quote(s, i) for i, s in enumerate(spots)) + "]"
    o = score_case(_audit_fake(case, raw), case, "near", "toy")
    assert o.true_positives == len(case.planted)
    assert o.false_positives == 0
    assert not o.planted_missed


def test_ungrounded_hallucination_is_not_a_false_positive(clean_bundle):
    """The negative-control result depends on this: a finding the model could
    not evidence is filtered before scoring, so it costs nothing."""
    from wca.benchmark import score_case
    from wca.inject import inject

    bundle, g = clean_bundle
    clean = inject(bundle, g, seed=42, patterns=())
    raw = json.dumps([{
        "title": "ghost", "severity": "high", "category": "injection",
        "files": ["lib/db.py"], "evidence": "os.system(user_input)",
        "why_cross_file": "x", "confidence": 0.95,
    }])
    o = score_case(_audit_fake(clean, raw), clean, "clean", "toy")
    assert o.n_grounded == 0
    assert o.false_positives == 0


def test_grounded_but_unplanted_finding_is_a_false_positive(clean_bundle):
    """Quoting real code is not the same as being right. A grounded finding on a
    clean repo matches nothing planted and must be counted against precision."""
    from wca.benchmark import score_case
    from wca.inject import inject

    bundle, g = clean_bundle
    clean = inject(bundle, g, seed=42, patterns=())
    raw = json.dumps([{
        "title": "real line, nothing planted", "severity": "high", "category": "injection",
        "files": ["lib/db.py"], "evidence": "cur.execute(sql)",
        "why_cross_file": "x", "confidence": 0.9,
    }])
    o = score_case(_audit_fake(clean, raw), clean, "clean", "toy")
    assert o.n_grounded == 1
    assert o.false_positives == 1
    assert o.true_positives == 0


def test_missed_planted_vulnerabilities_are_recorded(clean_bundle):
    from wca.benchmark import score_case
    from wca.inject import inject

    bundle, g = clean_bundle
    case = inject(bundle, g, seed=42, distance="near")
    spot = next(iter(case.planted[0].spots.values()))
    o = score_case(_audit_fake(case, "[" + _quote(spot) + "]"), case, "near", "toy")
    assert o.true_positives == 1
    assert len(o.planted_missed) == len(case.planted) - 1


def test_summary_computes_precision_recall_and_ablation():
    from wca.benchmark import AuditOutcome, summarise

    outcomes = [
        AuditOutcome(repo="a", variant="near", n_planted=2, true_positives=2),
        AuditOutcome(repo="a", variant="far", n_planted=2, true_positives=1),
        AuditOutcome(repo="a", variant="clean", n_planted=0, false_positives=1),
    ]
    s = summarise(outcomes)
    assert s["true_positives"] == 3
    assert s["false_positives"] == 1
    assert s["recall"] == 0.75  # 3 of 4 planted
    assert s["precision"] == 0.75  # 3 of 4 reported
    assert s["by_variant"]["near"]["recall"] == 1.0
    assert s["by_variant"]["far"]["recall"] == 0.5


def test_failed_audits_do_not_corrupt_the_score():
    from wca.benchmark import AuditOutcome, summarise

    outcomes = [
        AuditOutcome(repo="a", variant="near", n_planted=2, true_positives=2),
        AuditOutcome(repo="b", variant="near", n_planted=2, error="OOM"),
    ]
    s = summarise(outcomes)
    assert s["failed_audits"] == 1
    assert s["planted_total"] == 2  # the failed repo's planted vulns are excluded
    assert s["recall"] == 1.0
