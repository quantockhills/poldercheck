"""OpenRouter embeddings client — Qwen3-Embedding-0.6B.

Replaces local sentence-transformers for better multilingual retrieval quality.
Drop-in: embed_texts(list[str]) -> list[list[float]], embed_query(str) -> list[float].

Requires OPENROUTER_API_KEY env var (falls back to LLM_API_KEY).
"""
import os
import time

from openai import OpenAI

EMBED_MODEL_ID = "qwen/qwen3-embedding-8b"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "Set OPENROUTER_API_KEY in .env to use embeddings (or LLM_API_KEY pointing to OpenRouter)."
            )
        _client = OpenAI(base_url=_OPENROUTER_BASE, api_key=api_key)
    return _client


def embed_texts(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed a list of texts via OpenRouter; returns one vector per input, in order."""
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL_ID, input=batch)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend(d.embedding for d in sorted_data)
        if i + batch_size < len(texts):
            time.sleep(0.05)
    return all_embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
