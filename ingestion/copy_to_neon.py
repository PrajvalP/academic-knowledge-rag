"""Copy the local database (Docker) to Neon, vectors included.

Creates the schema on Neon, then copies documents, chunks (with their
embeddings) and users, so nothing has to be re-extracted or re-embedded.
Safe to re-run: it empties the Neon tables first.

Needs in .env: DATABASE_URL (local) and NEON_DATABASE_URL (target).
Usage: python ingestion/copy_to_neon.py
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
BATCH = 100

DOC_COLS = ["id", "file", "title", "doc_type", "course_code", "school", "year",
            "valid_from", "access_level", "source_url"]          # superseded_by later
CHUNK_COLS = ["id", "document_id", "chunk_id", "section", "page_start", "page_end",
              "text", "embed_text", "num_words", "embedding"]    # tsv is generated
USER_COLS = ["id", "email", "password_hash", "role"]


def direct(url: str) -> str:
    """Neon's pooled host has '-pooler' in it; bulk copying works best direct."""
    return url.replace("-pooler", "", 1)


def copy_table(src, dst, table, cols):
    rows = src.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY id").fetchall()
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) "
           f"VALUES ({', '.join(['%s'] * len(cols))})")
    with dst.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            cur.executemany(sql, rows[i:i + BATCH])
    # rows were inserted with their ids, so move each id counter past the
    # highest id, otherwise the next normal INSERT would reuse id 1
    dst.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"(SELECT COALESCE(MAX(id), 1) FROM {table}))")
    print(f"  {table}: {len(rows)} rows")


def main():
    load_dotenv(ROOT / ".env")
    local_url = os.environ["DATABASE_URL"]
    neon_url = os.environ.get("NEON_DATABASE_URL")
    if not neon_url:
        sys.exit("NEON_DATABASE_URL is missing from .env")

    # safety: this script empties the target, so never point it at localhost
    host = urlparse(neon_url).hostname or ""
    if host in ("localhost", "127.0.0.1") or neon_url == local_url:
        sys.exit("Refusing: the target looks like the local database")

    with psycopg.connect(local_url) as src, psycopg.connect(direct(neon_url)) as dst:
        print(f"Copying local -> {host}")

        # 1. schema (safe to run again: IF NOT EXISTS everywhere)
        dst.execute((ROOT / "ingestion" / "schema.sql").read_text())
        register_vector(src)
        register_vector(dst)   # needs the vector extension, created by the schema

        # 2. empty the target, then copy (documents first: chunks point to them)
        dst.execute("TRUNCATE users, chunks, documents RESTART IDENTITY CASCADE")
        copy_table(src, dst, "documents", DOC_COLS)
        copy_table(src, dst, "chunks", CHUNK_COLS)
        copy_table(src, dst, "users", USER_COLS)

        # 3. superseded_by points from one document to another, so it is
        #    copied after all documents exist
        links = src.execute("SELECT id, superseded_by FROM documents "
                            "WHERE superseded_by IS NOT NULL").fetchall()
        with dst.cursor() as cur:
            cur.executemany("UPDATE documents SET superseded_by = %s WHERE id = %s",
                            [(sup, doc) for doc, sup in links])
        print(f"  superseded links: {len(links)}")

        # 4. the vector index (created by embed.py locally, so add it here too)
        dst.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)")
        dst.commit()

        # 5. check: same counts on both sides
        print("\nCheck (local / Neon):")
        for q in ["SELECT count(*) FROM documents",
                  "SELECT count(*) FROM documents WHERE superseded_by IS NOT NULL",
                  "SELECT count(*) FROM chunks",
                  "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL",
                  "SELECT count(*) FROM users"]:
            a, b = src.execute(q).fetchone()[0], dst.execute(q).fetchone()[0]
            print(f"  {'OK ' if a == b else 'DIFF'} {a:4} / {b:4}  {q[21:]}")


if __name__ == "__main__":
    main()
