"""Application configuration."""

import os

# GROUND TRUTH: hardcoded credential, consumed in app/handlers.py
DB_PASSWORD = "admin_password_123"
DB_URL = "postgresql://svc:admin_password_123@prod-db.internal:5432/app"

DEBUG = os.getenv("APP_DEBUG", "0") == "1"


def get_conn_string():
    return DB_URL


def get_password():
    return DB_PASSWORD
