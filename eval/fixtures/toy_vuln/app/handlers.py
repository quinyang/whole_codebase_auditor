"""HTTP request handlers."""

import logging

from lib.config import DB_PASSWORD, get_conn_string
from lib.db import execute_raw

logger = logging.getLogger(__name__)


def health(request):
    return {"status": "ok"}


def debug_dump(request):
    """GROUND TRUTH #1: credential defined in lib/config.py is logged here."""
    logger.info("connecting with %s / %s", get_conn_string(), DB_PASSWORD)
    return {"ok": True}


def run_report(request):
    """GROUND TRUTH #2: untrusted input reaches the sink in lib/db.py."""
    table = request.args.get("table")
    return execute_raw("SELECT * FROM " + table)
