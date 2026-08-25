"""Database access layer."""

import sqlite3

from lib.config import get_conn_string


def execute_raw(sql):
    """GROUND TRUTH: unparameterised sink. Caller is app/handlers.py."""
    conn = sqlite3.connect(get_conn_string())
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


def execute_safe(sql, params):
    conn = sqlite3.connect(get_conn_string())
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()
