"""Conversation history persistence.

Each conversation is stored as a JSON file in data/history/.
When a conversation is deleted, its JSON and any associated dataset cache
(data/history/{conv_id}/datasets/) are removed.
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).parent.parent / "data" / "history"


def save_conversation(query: str, result: dict, settings: dict) -> str:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    conv_id = uuid.uuid4().hex[:10]
    timestamp = datetime.now().isoformat()
    slug = timestamp[:19].replace(":", "-")
    data = {
        "id": conv_id,
        "timestamp": timestamp,
        "query": query,
        "settings": settings,
        "final_response": result.get("final_response", ""),
        "political_response": result.get("political_response", ""),
        "data_response": result.get("data_response", ""),
        "political_passages": result.get("political_passages", []),
    }
    (HISTORY_DIR / f"{slug}_{conv_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return conv_id


def load_history() -> list[dict]:
    if not HISTORY_DIR.exists():
        return []
    convos = []
    for f in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            convos.append(json.loads(f.read_text()))
        except Exception:
            pass
    return convos


def delete_conversation(conv_id: str) -> None:
    if not HISTORY_DIR.exists():
        return
    for f in HISTORY_DIR.glob(f"*{conv_id}*.json"):
        f.unlink(missing_ok=True)
    dataset_cache = HISTORY_DIR / conv_id
    if dataset_cache.exists():
        shutil.rmtree(dataset_cache)
