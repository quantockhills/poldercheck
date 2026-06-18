"""Drop both ChromaDB collections and rebuild them with Qwen3-Embedding-0.6B.

Run once after switching from sentence-transformers to OpenRouter embeddings.
Requires OPENROUTER_API_KEY in .env.

    python -m scripts.rebuild_embeddings
"""

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTIONS = ("poldercheck_static", "cbs_catalog")


def _drop_collections():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for name in COLLECTIONS:
        try:
            client.delete_collection(name)
            print(f"Dropped: {name}")
        except Exception:
            print(f"Not found (skip): {name}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=== Step 1: drop stale collections ===")
    _drop_collections()

    print("\n=== Step 2: rebuild poldercheck_static ===")
    from src.ingest.chunk import build_store

    build_store()

    print("\n=== Step 3: rebuild cbs_catalog from JSONL ===")
    import chromadb as _chroma

    from src.ingest.retrieve import _build_cbs_collection_from_jsonl

    client = _chroma.PersistentClient(path=CHROMA_PATH)
    _build_cbs_collection_from_jsonl(client)

    print("\nDone. Both collections rebuilt with Qwen3-Embedding-0.6B.")
