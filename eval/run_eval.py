"""Stage 8: run the test set and score the system.

Two modes:
  python eval/run_eval.py --retrieval   search + rerank only (free, no LLM)
  python eval/run_eval.py               full pipeline with the LLM

Results are saved to eval/results/<date>_<mode>.json so runs can be compared.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from generation import MIN_RERANK_SCORE, answer  # noqa: E402
from rerank import rerank                         # noqa: E402
from retrieval import search                      # noqa: E402
from tracing import get_client, propagate_attributes  # noqa: E402

from auth import ROLE_LEVELS                     # noqa: E402


def chunk_file(chunk_id: str) -> str:
    return chunk_id.split("#")[0] + ".pdf"


def run_one(t: dict, use_llm: bool) -> dict:
    levels = ROLE_LEVELS[t["role"]]
    expected = set(t["expected_files"])
    outdated = set(t["outdated_files"])
    t0 = time.perf_counter()

    if use_llm:
        with propagate_attributes(tags=["eval", t["category"], t["role"]],
                                  metadata={"test_id": t["id"]}, trace_name="eval"):
            res = answer(t["question"], levels=levels)
        passages = res["passages"]
        answered = res["status"] == "answered"
        cited = [c for c in res["citations"]]
        cited_files = {
            chunk_file(passages[c["n"] - 1]["chunk_id"]) for c in cited
        }
        output = res["answer"]
        reason = res["reason"]
        # for general questions: does the answer say which course its rule is from?
        names_course = None
        if t.get("must_name_course") and answered:
            text = output.lower()
            names_course = any(
                (c.get("course_code") or "").lower() in text and c.get("course_code")
                or c["title"].lower() in text for c in res["citations"])
    else:
        passages = rerank(t["question"], search(t["question"], levels=levels))
        answered = bool(passages) and passages[0]["rerank_score"] >= MIN_RERANK_SCORE
        cited_files, output, reason, names_course = set(), None, None, None

    top_files = [chunk_file(p["chunk_id"]) for p in passages]
    return {
        "id": t["id"], "category": t["category"], "role": t["role"],
        "question": t["question"], "should_answer": t["should_answer"],
        "expected_answer": t["expected_answer"],
        "seconds": round(time.perf_counter() - t0, 2),
        "top_score": round(passages[0]["rerank_score"], 3) if passages else 0.0,
        "top_files": top_files,
        # retrieval: did an expected document reach the top 5?
        "hit_at_5": bool(expected & set(top_files)) if expected else None,
        # decision: answered when it should, refused when it should
        "decision_ok": answered == t["should_answer"],
        "answered": answered,
        "answer": output, "idk_reason": reason,
        "cited_files": sorted(cited_files),
        "citation_ok": (bool(expected & cited_files)
                        if use_llm and answered and expected else None),
        "cited_outdated": bool(outdated & cited_files) if use_llm else None,
        "names_course": names_course,
        # did a restricted document leak to a role that may not see it?
        "leaked": any("synthetic" in f for f in top_files) and not t["should_answer"]
                  and t["category"] == "permission",
    }


def rate(values) -> str:
    vals = [v for v in values if v is not None]
    return f"{100 * sum(vals) / len(vals):5.1f}% ({sum(vals)}/{len(vals)})" if vals else "    -"


def main():
    use_llm = "--retrieval" not in sys.argv
    mode = "full" if use_llm else "retrieval"
    tests = [json.loads(l) for l in (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8")]

    results = []
    for t in tests:
        r = run_one(t, use_llm)
        results.append(r)
        mark = "ok  " if r["decision_ok"] and r["hit_at_5"] is not False else "FAIL"
        print(f"{mark} {r['id']:4} score={r['top_score']:.3f} {r['question'][:60]}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n=== Summary ({mode} mode, {len(results)} questions) ===")
    print(f"{'category':14} {'hit@5':>16} {'decision':>16} {'citation':>16}")
    for cat, rs in list(by_cat.items()) + [("ALL", results)]:
        print(f"{cat:14} {rate(r['hit_at_5'] for r in rs):>16} "
              f"{rate(r['decision_ok'] for r in rs):>16} "
              f"{rate(r['citation_ok'] for r in rs):>16}")
    if use_llm:
        print(f"cited an outdated version: {sum(bool(r['cited_outdated']) for r in results)}")
    print(f"restricted documents leaked: {sum(r['leaked'] for r in results)}")
    named = [r["names_course"] for r in results if r["names_course"] is not None]
    if named:
        print(f"ambiguous answers naming their course: {sum(named)}/{len(named)}")
    print(f"average seconds per question: {sum(r['seconds'] for r in results) / len(results):.1f}")

    failures = [r for r in results if not r["decision_ok"] or r["hit_at_5"] is False
                or r["citation_ok"] is False or r["cited_outdated"]
                or r["names_course"] is False]
    print(f"\n=== {len(failures)} failures ===")
    for r in failures:
        print(f"{r['id']}: {r['question']}")
        print(f"   should_answer={r['should_answer']} answered={r['answered']} "
              f"score={r['top_score']} reason={r['idk_reason']}")
        print(f"   top files: {r['top_files'][:3]}")
        if r["answer"]:
            print(f"   answer: {r['answer'][:200]}")

    out = ROOT / "eval" / "results" / f"{datetime.now():%Y-%m-%d_%H%M}_{mode}.json"
    out.write_text(json.dumps({
        "mode": mode, "min_rerank_score": MIN_RERANK_SCORE, "results": results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")
    get_client().flush()   # send traces before the script exits


if __name__ == "__main__":
    main()
