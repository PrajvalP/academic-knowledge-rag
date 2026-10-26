"""Create one demo user per role (for local testing and demos).

Emails: student@ju.test, professor@ju.test, admin@ju.test
Password: DEMO_PASSWORD from .env
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from auth import hash_password  # noqa: E402

load_dotenv(ROOT / ".env")
password = os.environ.get("DEMO_PASSWORD")
if not password:
    sys.exit("Add DEMO_PASSWORD to .env first")

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    for role in ["student", "professor", "admin"]:
        conn.execute("""
            INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash,
                                              role = EXCLUDED.role
        """, (f"{role}@ju.test", hash_password(password), role))
        print(f"{role}@ju.test ({role})")
print("Demo users ready")
