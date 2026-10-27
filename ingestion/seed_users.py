"""Create one demo user per role (for local testing and the public demo).

Emails: student@ju.test, professor@ju.test, admin@ju.test
Passwords from the environment / .env:
  DEMO_PASSWORD   student and professor (fine to publish in the README)
  ADMIN_PASSWORD  admin (keep private); falls back to DEMO_PASSWORD if unset

Runs against DATABASE_URL, so to update Neon:
  DATABASE_URL="$NEON" DEMO_PASSWORD=... ADMIN_PASSWORD=... python ingestion/seed_users.py
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from auth import hash_password  # noqa: E402

load_dotenv(ROOT / ".env")   # values already set on the command line win
demo = os.environ.get("DEMO_PASSWORD")
if not demo:
    sys.exit("Set DEMO_PASSWORD first")
passwords = {
    "student": demo,
    "professor": demo,
    "admin": os.environ.get("ADMIN_PASSWORD") or demo,
}

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    host = conn.info.host
    for role, password in passwords.items():
        conn.execute("""
            INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash,
                                              role = EXCLUDED.role
        """, (f"{role}@ju.test", hash_password(password), role))
        print(f"{role}@ju.test ({role})")
print(f"Demo users ready on {host}"
      + ("" if os.environ.get("ADMIN_PASSWORD") else "  (admin uses DEMO_PASSWORD)"))
