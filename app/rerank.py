"""Stage 4: rerank hybrid search results with a cross-encoder.

The embedding model compares question and chunk separately (fast, rough).
A cross-encoder reads them together as one input (slow, accurate), so it
is only run on the ~20 candidates hybrid search returns.

Usage: python app/rerank.py "Can I take TDIS22 without machine learning?"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import rerank_scores, warm_up  # noqa: E402
from retrieval import search  # noqa: E402
from tracing import get_client, observe  # noqa: E402

TOP_K = 5


def passage(r: dict) -> str:
    """What the reranker reads: where the text comes from, plus the text."""
    code = f" ({r['course_code']})" if r.get("course_code") else ""
    return f"{r['title']}{code} | {r['section']}\n{r['text']}"


@observe(name="rerank", capture_input=False, capture_output=False)
def rerank(question: str, results: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Score each (question, passage) pair; return the top_k, best first.

    rerank_score is between 0 and 1 (the model applies a sigmoid).
    Stage 5 uses the best score to decide when to answer "I don't know".
    """
    if not results:
        return []
    scores = rerank_scores(question, [passage(r) for r in results])   # local or API
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    top = sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_k]
    get_client().update_current_span(
        input={"question": question, "candidates": len(results)},
        output=[{"chunk_id": r["chunk_id"], "score": round(r["rerank_score"], 3)} for r in top],
    )
    return top


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I take TDIS22 without machine learning?"
    warm_up()                            # load local models first, so timing excludes loading

    t0 = time.perf_counter()
    candidates = search(question, top_n=20)
    t1 = time.perf_counter()
    before = {r["chunk_id"]: i for i, r in enumerate(candidates, start=1)}
    top = rerank(question, candidates)
    t2 = time.perf_counter()

    print(f"Q: {question}")
    print(f"search {1000 * (t1 - t0):.0f} ms | rerank {1000 * (t2 - t1):.0f} ms "
          f"for {len(candidates)} candidates\n")
    for i, r in enumerate(top, start=1):
        print(f"{i}. {r['rerank_score']:.3f}  (was #{before[r['chunk_id']]:<2}) "
              f"{r['chunk_id']:34} [{r['section'][:30]}]")
        print(f"   {r['text'][:140].replace(chr(10), ' ')}...\n")
