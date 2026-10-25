"""Try semantic search from the terminal.

Usage: python ingestion/try_search.py "can I take deep learning without machine learning?"
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import ROOT, load_model  # noqa: E402

TOP_K = 5


def main():
    question = " ".join(sys.argv[1:]) or "What are the entry requirements for Deep Learning?"
    load_dotenv(ROOT / ".env")
    model = load_model()
    q_vec = model.encode(question, normalize_embeddings=True)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        rows = conn.execute("""
            SELECT c.chunk_id, c.section, c.page_start, d.valid_from,
                   1 - (c.embedding <=> %s) AS similarity, c.text
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """, (q_vec, q_vec, TOP_K)).fetchall()

    print(f"\nQ: {question}\n")
    for chunk_id, section, page, valid_from, sim, text in rows:
        print(f"{sim:.3f}  {chunk_id}  [{section}]  p.{page}  valid from {valid_from}")
        print(f"       {text[:150].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
