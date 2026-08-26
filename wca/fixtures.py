"""Embedded test fixture.

The toy repo also lives at `eval/fixtures/toy_vuln/` as browsable files; this
module is the packaged copy, so `pip install wca` carries it and the demo runs
without cloning anything. `test_embedded_fixture_matches_disk` keeps them in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

GROUND_TRUTH: dict = json.loads(
    "{\"repo\": \"toy_vuln\", \"purpose\": \"Smallest repo that exercises the cross-file claim. Used for the first end-to-end audit, where the point is proving the pipeline runs -- not measuring detection rate.\", \"n_files\": 4, \"planted\": [{\"id\": \"cred_leak\", \"category\": \"hardcoded_secret\", \"severity\": \"critical\", \"definition\": {\"file\": \"lib/config.py\", \"line\": 6, \"code\": \"DB_PASSWORD = \\\"admin_password_123\\\"\"}, \"use\": {\"file\": \"app/handlers.py\", \"line\": 17, \"code\": \"logger.info(\\\"connecting with %s / %s\\\", get_conn_string(), DB_PASSWORD)\"}, \"why_cross_file\": \"The credential is defined in lib/config.py and written to logs in app/handlers.py. Neither file alone shows the leak.\", \"requires_files\": [\"lib/config.py\", \"app/handlers.py\"]}, {\"id\": \"sql_injection\", \"category\": \"injection\", \"severity\": \"high\", \"source\": {\"file\": \"app/handlers.py\", \"line\": 23, \"code\": \"table = request.args.get(\\\"table\\\")\"}, \"sink\": {\"file\": \"lib/db.py\", \"line\": 12, \"code\": \"cur.execute(sql)\"}, \"why_cross_file\": \"Untrusted input enters in app/handlers.py and reaches an unparameterised execute() in lib/db.py via execute_raw().\", \"requires_files\": [\"app/handlers.py\", \"lib/db.py\"]}], \"not_planted\": [\"app/util.py contains no vulnerability. A finding against it is a false positive.\", \"lib/db.py execute_safe() is correctly parameterised. Flagging it is a false positive.\"], \"notes\": \"Line numbers are 1-indexed and refer to the files as committed. If you edit the fixture, update them -- the grounding check compares against these.\"}"
)

TOY_VULN: dict[str, str] = {
    'app/handlers.py': '"""HTTP request handlers."""\n\nimport logging\n\nfrom lib.config import DB_PASSWORD, get_conn_string\nfrom lib.db import execute_raw\n\nlogger = logging.getLogger(__name__)\n\n\ndef health(request):\n    return {"status": "ok"}\n\n\ndef debug_dump(request):\n    """GROUND TRUTH #1: credential defined in lib/config.py is logged here."""\n    logger.info("connecting with %s / %s", get_conn_string(), DB_PASSWORD)\n    return {"ok": True}\n\n\ndef run_report(request):\n    """GROUND TRUTH #2: untrusted input reaches the sink in lib/db.py."""\n    table = request.args.get("table")\n    return execute_raw("SELECT * FROM " + table)\n',
    'app/util.py': '"""Assorted helpers. Contains no planted vulnerability -- noise for the packer."""\n\n\ndef slugify(text):\n    return "-".join(text.lower().split())\n\n\ndef truncate(text, n=80):\n    return text if len(text) <= n else text[: n - 1] + "…"\n\n\ndef chunk(items, size):\n    for i in range(0, len(items), size):\n        yield items[i : i + size]\n',
    'lib/config.py': '"""Application configuration."""\n\nimport os\n\n# GROUND TRUTH: hardcoded credential, consumed in app/handlers.py\nDB_PASSWORD = "admin_password_123"\nDB_URL = "postgresql://svc:admin_password_123@prod-db.internal:5432/app"\n\nDEBUG = os.getenv("APP_DEBUG", "0") == "1"\n\n\ndef get_conn_string():\n    return DB_URL\n\n\ndef get_password():\n    return DB_PASSWORD\n',
    'lib/db.py': '"""Database access layer."""\n\nimport sqlite3\n\nfrom lib.config import get_conn_string\n\n\ndef execute_raw(sql):\n    """GROUND TRUTH: unparameterised sink. Caller is app/handlers.py."""\n    conn = sqlite3.connect(get_conn_string())\n    cur = conn.cursor()\n    cur.execute(sql)\n    return cur.fetchall()\n\n\ndef execute_safe(sql, params):\n    conn = sqlite3.connect(get_conn_string())\n    cur = conn.cursor()\n    cur.execute(sql, params)\n    return cur.fetchall()\n',
}


def materialise_toy_vuln(dest: str | Path) -> Path:
    """Write the toy repo to `dest` and return its path."""
    dest = Path(dest)
    for rel, content in TOY_VULN.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return dest
