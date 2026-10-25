"""Stage 3: hybrid retrieval.

search() runs three searches and merges them with Reciprocal Rank Fusion:
  1. semantic: nearest embeddings (meaning)
  2. keyword:  Postgres full-text search (exact words, course codes)
  3. course:   if the question names a course code, that course's chunks

Usage: python app/retrieval.py "can I take TDIS22 without machine learning?"
"""
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-m3"   # must match ingestion/embed.py
CANDIDATES = 20              # results taken from each search
RRF_K = 60                   # standard RRF constant: dampens rank differences
COURSE_CODE = re.compile(r"\b([A-Za-z]{4}\d{2})\b")   # e.g. TARI29, tmls22

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes seconds)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 512
    return model


def semantic(cur, q_vec, levels) -> list[int]:
    cur.execute("""
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.access_level = ANY(%s)
        ORDER BY c.embedding <=> %s LIMIT %s
    """, (levels, q_vec, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def keyword(cur, question, levels) -> list[int]:
    # plainto_tsquery joins words with AND (&). A natural question rarely
    # contains every word of a passage, so switch to OR (|) and let ts_rank
    # put passages matching more words first.
    cur.execute("""
        WITH q AS (
            SELECT to_tsquery('english',
                   replace(plainto_tsquery('english', %s)::text, '&', '|')) AS query
        )
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id, q
        WHERE d.access_level = ANY(%s) AND c.tsv @@ q.query
        ORDER BY ts_rank(c.tsv, q.query) DESC LIMIT %s
    """, (question, levels, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def course(cur, codes, q_vec, levels) -> list[int]:
    if not codes:
        return []
    cur.execute("""
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.access_level = ANY(%s) AND d.course_code = ANY(%s)
        ORDER BY c.embedding <=> %s LIMIT %s
    """, (levels, codes, q_vec, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def rrf(rankings: dict[str, list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (K + rank)."""
    scores: dict[int, float] = {}
    for ids in rankings.values():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return scores


def search(question: str, levels=("public", "student"), top_n: int = 20) -> list[dict]:
    """Return the top_n chunks for a question, best first."""
    levels = list(levels)
    codes = sorted({c.upper() for c in COURSE_CODE.findall(question)})
    q_vec = get_model().encode(question, normalize_embeddings=True)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rankings = {
                "semantic": semantic(cur, q_vec, levels),
                "keyword": keyword(cur, question, levels),
                "course": course(cur, codes, q_vec, levels),
            }
            scores = rrf(rankings)
            best = sorted(scores, key=scores.get, reverse=True)[:top_n]
            if not best:
                return []
            cur.execute("""
                SELECT c.id, c.chunk_id, c.section, c.page_start, c.page_end, c.text,
                       d.title, d.course_code, d.year, d.valid_from, d.source_url
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id = ANY(%s)
            """, (best,))
            cols = [col.name for col in cur.description]
            rows = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

    results = []
    for chunk_id in best:
        row = rows[chunk_id]
        row["rrf_score"] = scores[chunk_id]
        # where this chunk ranked in each list (None = not found there)
        row["ranks"] = {name: (ids.index(chunk_id) + 1 if chunk_id in ids else None)
                        for name, ids in rankings.items()}
        results.append(row)
    return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I take TDIS22 without machine learning?"
    print(f"Q: {question}\n")
    for i, r in enumerate(search(question, top_n=8), start=1):
        ranks = "  ".join(f"{k[:3]}={v or '-':<2}" for k, v in r["ranks"].items())
        print(f"{i}. {r['rrf_score']:.4f}  {r['chunk_id']:34} [{r['section'][:28]}]  {ranks}")
        print(f"   {r['text'][:140].replace(chr(10), ' ')}...\n")
