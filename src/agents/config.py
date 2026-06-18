"""Agent configuration — provider-agnostic.

Everything is driven by env vars; the code has no provider preference.

Shared (all agents fall back to these):
  LLM_BASE_URL         OpenAI-compatible base URL (required)
  LLM_API_KEY          API key (falls back to OPENROUTER_API_KEY, then ANTHROPIC_AUTH_TOKEN)

Per-agent model / overrides (optional):
  POLDERCHECK_POLITICAL_MODEL   (falls back to POLDERCHECK_MODEL, then empty — must be set)
  POLDERCHECK_DATA_MODEL
  POLDERCHECK_SYNTHESIS_MODEL
  POLDERCHECK_POLITICAL_BASE_URL / _API_KEY   (override per agent if needed)

MVP (.env):
  LLM_BASE_URL=https://api.deepseek.com
  LLM_API_KEY=sk-...
  POLDERCHECK_MODEL=deepseek-v4-pro          # all three agents use this

Later (.env):
  LLM_BASE_URL=https://openrouter.ai/api/v1
  LLM_API_KEY=sk-or-...
  POLDERCHECK_POLITICAL_MODEL=anthropic/claude-sonnet-4-6
  POLDERCHECK_DATA_MODEL=qwen/qwen3-30b-a3b
  POLDERCHECK_SYNTHESIS_MODEL=anthropic/claude-sonnet-4-6
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared credentials
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("LLM_BASE_URL", "")


def _resolve_api_key() -> str:
    for var in ("LLM_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        val = os.environ.get(var, "")
        if val:
            return val
    return ""


API_KEY = _resolve_api_key()

# ---------------------------------------------------------------------------
# Per-agent configs
# ---------------------------------------------------------------------------


def _agent_cfg(name: str, max_tokens: int) -> dict:
    prefix = f"POLDERCHECK_{name.upper()}"
    model = os.environ.get(f"{prefix}_MODEL") or os.environ.get("POLDERCHECK_MODEL") or ""
    return {
        "base_url": os.environ.get(f"{prefix}_BASE_URL", BASE_URL),
        "api_key": os.environ.get(f"{prefix}_API_KEY", API_KEY),
        "model": model,
        "max_tokens": max_tokens,
    }


AGENT_CONFIGS = {
    "political_analyst": _agent_cfg("political", 800),
    "opentk_agent": _agent_cfg("opentk", 800),  # tool-calling loop — use a fast model
    "data_analyst": _agent_cfg("data", 600),
    "synthesis": _agent_cfg("synthesis", 2000),
}
