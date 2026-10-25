"""Stage 7: mark outdated document versions.

Documents in the same "family" (same course code, or same title for
policies) are versions of each other. The current version is the newest
one whose valid_from date has already arrived. Every older version gets
superseded_by = id of the current one. Versions valid in the future are
left alone: they become current when their date comes.

Run after load.py:  python ingestion/versions.py
"""
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def effective_date(valid_from, year) -> date:
    """Use valid_from; fall back to 1 January of the year if it's missing."""
    return valid_from or date(year or 1900, 1, 1)


def main():
    load_dotenv(ROOT / ".env")
    today = date.today()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        docs = conn.execute(
            "SELECT id, file, title, course_code, valid_from, year FROM documents"
        ).fetchall()

        families = defaultdict(list)
        for doc in docs:
            doc_id, file, title, code, valid_from, year = doc
            families[code or title].append(doc)

        conn.execute("UPDATE documents SET superseded_by = NULL")   # start clean
        marked = 0
        for key, members in sorted(families.items()):
            if len(members) < 2:
                continue
            in_force = [d for d in members if effective_date(d[4], d[5]) <= today]
            if not in_force:
                continue
            current = max(in_force, key=lambda d: effective_date(d[4], d[5]))
            for d in in_force:
                if d[0] != current[0]:
                    conn.execute("UPDATE documents SET superseded_by = %s WHERE id = %s",
                                 (current[0], d[0]))
                    marked += 1
                    print(f"{d[1]:32} -> superseded by {current[1]}")

        print(f"\nMarked {marked} documents as superseded (today = {today})")


if __name__ == "__main__":
    main()
