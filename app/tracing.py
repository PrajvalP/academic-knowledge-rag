"""Stage 9: tracing helpers.

If Langfuse keys are in .env, these are the real Langfuse tools.
If not, they become harmless no-ops, so the app still runs without
tracing (useful for tests, or a machine without keys).
"""
import os
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe, propagate_attributes  # noqa: F401
else:
    def observe(*args, **kwargs):
        if args and callable(args[0]):      # used as @observe
            return args[0]
        return lambda func: func            # used as @observe(...)

    def propagate_attributes(**kwargs):
        return nullcontext()

    class _NoOpClient:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def get_client():
        return _NoOpClient()


def llm_cost(input_tokens: int, output_tokens: int) -> dict | None:
    """Cost in USD from per-million-token prices in .env (if set)."""
    p_in = os.environ.get("LLM_PRICE_INPUT_PER_MTOK")
    p_out = os.environ.get("LLM_PRICE_OUTPUT_PER_MTOK")
    if not (p_in and p_out):
        return None
    cost_in = input_tokens * float(p_in) / 1_000_000
    cost_out = output_tokens * float(p_out) / 1_000_000
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}
