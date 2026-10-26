"""Embedding and reranking, run locally or through an API.

MODEL_BACKEND in .env chooses:
  local  the models run on this machine (needs torch, ~4-5 GB of memory)
  api    the same models are called through Novita's API (tiny memory)

Both modes use the SAME models (bge-m3 and bge-reranker-v2-m3), so the
vectors already stored in the database work in either mode.

The rest of the app only calls embed_query() and rerank_scores(); it never
needs to know which mode is active.
"""
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LOCAL_EMBED_MODEL = "BAAI/bge-m3"
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
API_EMBED_MODEL = "baai/bge-m3"
API_RERANK_MODEL = "baai/bge-reranker-v2-m3"
API_BASE = os.environ.get("MODEL_API_BASE", "https://api.novita.ai/v3/openai")
API_TIMEOUT = 30   # seconds


def backend() -> str:
    mode = os.environ.get("MODEL_BACKEND", "local").lower()
    if mode not in ("local", "api"):
        raise ValueError(f"MODEL_BACKEND must be 'local' or 'api', not {mode!r}")
    return mode


# ---------- local mode: models loaded on this machine ----------
# torch and sentence_transformers are imported inside these functions, so
# api mode never loads them and can run without them installed.

def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _local_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_EMBED_MODEL, device=_device())
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def _local_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(LOCAL_RERANK_MODEL, device=_device(), max_length=512)


# ---------- api mode: same models, called over HTTP ----------

def _api_post(path: str, payload: dict) -> dict:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is missing from .env (needed for MODEL_BACKEND=api)")
    r = requests.post(f"{API_BASE}/{path}", json=payload, timeout=API_TIMEOUT,
                      headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()   # turn HTTP errors (401, 429, 500...) into exceptions
    return r.json()


# ---------- the two functions the app uses ----------

def embed_query(text: str) -> np.ndarray:
    """One text -> a normalized 1024-number vector."""
    if backend() == "api":
        data = _api_post("embeddings", {"model": API_EMBED_MODEL, "input": [text]})
        vec = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
        return vec / np.linalg.norm(vec)   # normalize, exactly like local mode
    return _local_embedder().encode(text, normalize_embeddings=True)


def rerank_scores(query: str, documents: list[str]) -> list[float]:
    """Relevance of each document to the query, between 0 and 1, in input order."""
    if not documents:
        return []
    if backend() == "api":
        data = _api_post("rerank", {"model": API_RERANK_MODEL, "query": query,
                                    "documents": documents, "top_n": len(documents)})
        scores = [0.0] * len(documents)
        for item in data.get("results", []):
            # results come back sorted by score; 'index' points to the input order
            scores[item["index"]] = float(item.get("relevance_score", item.get("score", 0.0)))
        # local scores are probabilities (sigmoid applied); some APIs return raw
        # logits instead. Values outside 0..1 can only be logits, so convert them.
        if any(s < 0 or s > 1 for s in scores):
            scores = [1 / (1 + math.exp(-s)) for s in scores]
        return scores
    return [float(s) for s in _local_reranker().predict([(query, d) for d in documents])]


def warm_up() -> None:
    """Load local models at startup so the first question isn't slow (no-op for api)."""
    if backend() == "local":
        _local_embedder()
        _local_reranker()
