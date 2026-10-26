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
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import embed_query  # noqa: E402
from tracing import get_client, observe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = 20              # results taken from each search
RRF_K = 60                   # standard RRF constant: dampens rank differences
COURSE_CODE = re.compile(r"\b([A-Za-z]{4}\d{2})\b")   # e.g. TARI29, tmls22
YEAR = re.compile(r"\b(20\d{2})\b")                     # e.g. "the 2024 rules"
# keep only current versions, unless include_old is true
CURRENT = "(d.superseded_by IS NULL OR %s)"

load_dotenv(ROOT / ".env")


def semantic(cur, q_vec, levels, include_old) -> list[int]:
    cur.execute(f"""
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.access_level = ANY(%s) AND {CURRENT}
        ORDER BY c.embedding <=> %s LIMIT %s
    """, (levels, include_old, q_vec, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def keyword(cur, question, levels, include_old) -> list[int]:
    # plainto_tsquery joins words with AND (&). A natural question rarely
    # contains every word of a passage, so switch to OR (|) and let ts_rank
    # put passages matching more words first.
    cur.execute(f"""
        WITH q AS (
            SELECT to_tsquery('english',
                   replace(plainto_tsquery('english', %s)::text, '&', '|')) AS query
        )
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id, q
        WHERE d.access_level = ANY(%s) AND {CURRENT} AND c.tsv @@ q.query
        ORDER BY ts_rank(c.tsv, q.query) DESC LIMIT %s
    """, (question, levels, include_old, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def course(cur, codes, q_vec, levels, include_old) -> list[int]:
    if not codes:
        return []
    cur.execute(f"""
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.access_level = ANY(%s) AND d.course_code = ANY(%s) AND {CURRENT}
        ORDER BY c.embedding <=> %s LIMIT %s
    """, (levels, codes, include_old, q_vec, CANDIDATES))
    return [r[0] for r in cur.fetchall()]


def rrf(rankings: dict[str, list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (K + rank)."""
    scores: dict[int, float] = {}
    for ids in rankings.values():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return scores


@observe(name="hybrid_search", as_type="retriever", capture_output=False)
def search(question: str, levels=("public", "student"), top_n: int = 20) -> list[dict]:
    """Return the top_n chunks for a question, best first."""
    levels = list(levels)
    codes = sorted({c.upper() for c in COURSE_CODE.findall(question)})
    # a question naming a year ("the 2024 syllabus") may need old versions
    include_old = bool(YEAR.search(question))
    q_vec = embed_query(question)   # local model or API, see models.py

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rankings = {
                "semantic": semantic(cur, q_vec, levels, include_old),
                "keyword": keyword(cur, question, levels, include_old),
                "course": course(cur, codes, q_vec, levels, include_old),
            }
            scores = rrf(rankings)
            best = sorted(scores, key=scores.get, reverse=True)[:top_n]
            if not best:
                return []
            cur.execute("""
                SELECT c.id, c.chunk_id, c.section, c.page_start, c.page_end, c.text,
                       d.title, d.course_code, d.year, d.valid_from, d.source_url,
                       d.superseded_by IS NOT NULL AS superseded
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

    get_client().update_current_span(
        output=[r["chunk_id"] for r in results],
        metadata={"found": {k: len(v) for k, v in rankings.items()},
                  "course_codes": codes, "include_old": include_old},
    )
    return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I take TDIS22 without machine learning?"
    print(f"Q: {question}\n")
    for i, r in enumerate(search(question, top_n=8), start=1):
        ranks = "  ".join(f"{k[:3]}={v or '-':<2}" for k, v in r["ranks"].items())
        print(f"{i}. {r['rrf_score']:.4f}  {r['chunk_id']:34} [{r['section'][:28]}]  {ranks}")
        print(f"   {r['text'][:140].replace(chr(10), ' ')}...\n")
