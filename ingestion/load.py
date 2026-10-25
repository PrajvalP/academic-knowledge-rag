"""Stage 1c: load chunks into Postgres.

Reads data/processed/chunks.jsonl and writes the documents and chunks
tables. Re-running it replaces each document's chunks, so it never
creates duplicates.
"""
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"

DOC_FIELDS = ["file", "title", "doc_type", "course_code", "school", "year",
              "valid_from", "access_level", "source_url"]


def main():
    load_dotenv(ROOT / ".env")
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]

    # group chunks by their document
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(c["file"], []).append(c)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            for file, doc_chunks in by_file.items():
                d = doc_chunks[0]
                values = [d[k] or None for k in DOC_FIELDS]   # "" becomes NULL

                # insert the document, or update it if it already exists
                cur.execute(f"""
                    INSERT INTO documents ({", ".join(DOC_FIELDS)})
                    VALUES ({", ".join(["%s"] * len(DOC_FIELDS))})
                    ON CONFLICT (file) DO UPDATE SET
                    {", ".join(f"{k} = EXCLUDED.{k}" for k in DOC_FIELDS[1:])}
                    RETURNING id
                """, values)
                doc_id = cur.fetchone()[0]

                # replace this document's chunks
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                cur.executemany("""
                    INSERT INTO chunks (document_id, chunk_id, section, page_start,
                                        page_end, text, embed_text, num_words)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, [(doc_id, c["chunk_id"], c["section"], c["page_start"],
                       c["page_end"], c["text"], c["embed_text"], c["num_words"])
                      for c in doc_chunks])
        # leaving the 'with' block commits everything at once

        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"Loaded {n} chunks from {docs} documents into Postgres")


if __name__ == "__main__":
    main()
