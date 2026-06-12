import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

AGENT_CONFIGS = {
    "political_analyst": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 800,
    },
    "data_analyst": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "qwen/qwen3-30b-a3b",
        "max_tokens": 600,
    },
    "synthesis": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 500,
    },
}
