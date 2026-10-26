"""Compare local models with the API versions before switching.

Checks three things:
  1. Do API embeddings of stored chunks match the vectors in the database?
  2. Do API embeddings of questions match local ones?
  3. Do API rerank scores match local ones (same scale, same order)?

Needs: local models (.venv), Postgres running, MODEL_API_KEY in .env.
Usage: python eval/compare_backends.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import models  # noqa: E402

QUESTIONS = [json.loads(l)["question"] for l in (ROOT / "eval" / "testset.jsonl").open()][::5]


def with_backend(mode: str, func, *args):
    """Run one function with MODEL_BACKEND temporarily set to mode."""
    old = os.environ.get("MODEL_BACKEND")
    os.environ["MODEL_BACKEND"] = mode
    try:
        return func(*args)
    finally:
        if old is None:
            os.environ.pop("MODEL_BACKEND")
        else:
            os.environ["MODEL_BACKEND"] = old


def main():
    print(f"API: {models.API_BASE}\n")

    # 1. stored chunk vectors (made locally) vs. API embeddings of the same text
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        rows = conn.execute("SELECT chunk_id, embed_text, embedding FROM chunks "
                            "ORDER BY random() LIMIT 10").fetchall()
    sims = [float(np.dot(stored, with_backend("api", models.embed_query, text)))
            for _, text, stored in rows]
    print("1. Stored chunk vectors vs API embeddings (cosine; 1.000 = identical)")
    print(f"   min {min(sims):.4f}  mean {np.mean(sims):.4f}\n")

    # 2. question embeddings, local vs API
    sims = [float(np.dot(with_backend("local", models.embed_query, q),
                         with_backend("api", models.embed_query, q))) for q in QUESTIONS]
    print(f"2. Question embeddings, local vs API ({len(QUESTIONS)} questions)")
    print(f"   min {min(sims):.4f}  mean {np.mean(sims):.4f}\n")

    # 3. rerank scores, local vs API, on real search candidates
    from rerank import passage
    from retrieval import search
    print("3. Rerank scores, local vs API")
    for q in QUESTIONS[:4]:
        docs = [passage(r) for r in with_backend("local", search, q)]
        local = with_backend("local", models.rerank_scores, q, docs)
        api = with_backend("api", models.rerank_scores, q, docs)
        diff = max(abs(a - b) for a, b in zip(local, api))
        same_top = int(np.argmax(local)) == int(np.argmax(api))
        print(f"   {q[:55]:55}  top local {max(local):.3f} / api {max(api):.3f}"
              f"  max diff {diff:.3f}  same #1: {same_top}")


if __name__ == "__main__":
    main()
