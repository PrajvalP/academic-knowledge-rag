"""Stage 5: generate a grounded answer with citations.

Pipeline: hybrid search -> rerank -> "I don't know" gate -> LLM -> citation check.

Settings in .env:
  LLM_PROVIDER = anthropic | openai
  LLM_MODEL    = model name (optional for anthropic)
  ANTHROPIC_API_KEY or OPENAI_API_KEY

Usage: python app/generation.py "Can I take TDIS22 without machine learning?"
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank import rerank      # noqa: E402
from retrieval import search   # noqa: E402

IDK = "I do not know based on the available documents."
MIN_RERANK_SCORE = 0.3   # below this, no passage really answers: skip the LLM
TOP_K = 5

SYSTEM_PROMPT = f"""You answer questions from students at Jönköping University.
Use ONLY the numbered passages provided. Never use outside knowledge.

Rules:
- Cite the passage numbers that support each sentence, like [1] or [2][3].
- If the passages do not contain the answer, set "answer" to exactly:
  "{IDK}" and "citations" to [].
- If passages from different years disagree, prefer the most recent one and say so.
- Some passages are in Swedish; answer in English.
- Be brief: at most 4 sentences.

Reply with JSON only, no other text:
{{"answer": "...", "citations": [1, 2]}}"""


def call_llm(system: str, user: str) -> str:
    """Send one prompt to the configured provider and return its text."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    model = os.environ.get("LLM_MODEL")

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        msg = client.messages.create(
            model=model or "claude-haiku-5-5", max_tokens=2000,  # room for thinking + answer
            system=system, messages=[{"role": "user", "content": user}],
        )
        # the reply is a list of blocks; some models add a "thinking" block
        # before the answer, so keep only the text blocks
        return "".join(b.text for b in msg.content if b.type == "text")

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()                # reads OPENAI_API_KEY
        if not model:
            raise ValueError("Set LLM_MODEL in .env for OpenAI")
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for n, p in enumerate(passages, start=1):
        code = f" ({p['course_code']})" if p.get("course_code") else ""
        valid = f", valid from {p['valid_from']}" if p.get("valid_from") else ""
        blocks.append(f"[{n}] {p['title']}{code}, {p['year']}{valid}, "
                      f"section \"{p['section']}\", page {p['page_start']}\n{p['text']}")
    return "\n\n".join(blocks)


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add words; find the {...}."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def idk(reason: str, passages=None) -> dict:
    return {"answer": IDK, "citations": [], "status": "i_dont_know",
            "reason": reason, "passages": passages or []}


def answer(question: str, levels=("public", "student")) -> dict:
    passages = rerank(question, search(question, levels=levels), top_k=TOP_K)

    # Gate 1: nothing relevant enough, so don't even ask the LLM (saves cost)
    if not passages or passages[0]["rerank_score"] < MIN_RERANK_SCORE:
        return idk("low_retrieval_score", passages)

    raw = call_llm(SYSTEM_PROMPT, f"Passages:\n\n{format_passages(passages)}"
                                  f"\n\nQuestion: {question}")
    try:
        data = parse_json(raw)
    except json.JSONDecodeError:
        return idk("unparseable_llm_output", passages)

    text = (data.get("answer") or "").strip()
    if not text or text == IDK:
        return idk("model_said_unknown", passages)

    # Gate 2: check citations in code, never just trust the prompt
    cited = {n for n in data.get("citations", []) if isinstance(n, int)}
    cited |= {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    valid = sorted(n for n in cited if 1 <= n <= len(passages))
    if not valid:
        return idk("no_valid_citations", passages)

    citations = [{
        "n": n,
        "title": passages[n - 1]["title"],
        "course_code": passages[n - 1]["course_code"],
        "year": passages[n - 1]["year"],
        "section": passages[n - 1]["section"],
        "page": passages[n - 1]["page_start"],
        "url": passages[n - 1]["source_url"],
    } for n in valid]
    return {"answer": text, "citations": citations, "status": "answered",
            "reason": None, "passages": passages}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    q = " ".join(sys.argv[1:]) or "Can I take TDIS22 without machine learning?"
    result = answer(q)
    print(f"Q: {q}\n\nA: {result['answer']}\n")
    print(f"status: {result['status']}" + (f" ({result['reason']})" if result["reason"] else ""))
    for c in result["citations"]:
        code = f" {c['course_code']}" if c["course_code"] else ""
        print(f"  [{c['n']}] {c['title']}{code} ({c['year']}), {c['section']}, p.{c['page']}")
    if result["passages"]:
        print(f"top rerank score: {result['passages'][0]['rerank_score']:.3f}")
