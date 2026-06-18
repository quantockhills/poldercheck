"""Semantic search over the static ChromaDB corpus and the CBS catalog."""

import json
from pathlib import Path

import chromadb

from src.ingest.embed import embed_query, embed_texts

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"
CBS_CATALOG_COLLECTION = "cbs_catalog"
CBS_CATALOG_JSONL = Path("data/catalog/cbs_catalog.jsonl")

_collection = None
_cbs_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _build_cbs_collection_from_jsonl(client: chromadb.PersistentClient):
    """Build the CBS catalog ChromaDB collection from the committed JSONL."""
    if not CBS_CATALOG_JSONL.exists():
        raise FileNotFoundError(
            f"CBS catalog not found at {CBS_CATALOG_JSONL}. Run scripts/build_cbs_catalog.py to fetch and refresh it."
        )
    print("DEBUG_LOG: building CBS catalog index from JSONL (one-time ~5 min via API)...")
    datasets = [json.loads(line) for line in CBS_CATALOG_JSONL.read_text().splitlines() if line.strip()]

    collection = client.create_collection(CBS_CATALOG_COLLECTION, metadata={"hnsw:space": "cosine"})
    texts, metadatas, ids = [], [], []
    for ds in datasets:
        summary = ds.get("summary", "")[:200]
        embed_text = f"{ds['title']}. {summary}" if summary else ds["title"]
        texts.append(embed_text)
        metadatas.append(
            {
                "identifier": ds["identifier"],
                "title": ds["title"],
                "period": ds.get("modified", ds.get("period", "")),  # v4 uses modified date
                "language": ds.get("language", "nl"),
                "api_url": ds.get("api_url", ""),
            }
        )
        ids.append(ds["identifier"])

    print(f"DEBUG_LOG: embedding {len(texts)} datasets via OpenRouter...")
    embeddings = embed_texts(texts, batch_size=128)
    for i in range(0, len(texts), 1000):
        collection.add(
            documents=texts[i : i + 1000],
            embeddings=embeddings[i : i + 1000],
            metadatas=metadatas[i : i + 1000],
            ids=ids[i : i + 1000],
        )
    print(f"DEBUG_LOG: CBS catalog indexed: {len(texts)} datasets.")
    return collection


def _get_cbs_collection():
    global _cbs_collection
    if _cbs_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _cbs_collection = client.get_collection(CBS_CATALOG_COLLECTION)
        except Exception:
            _cbs_collection = _build_cbs_collection_from_jsonl(client)
    return _cbs_collection


def retrieve_static(
    query: str,
    n_results: int = 10,
    include_manifestos: bool = True,
) -> list[dict]:
    """
    Retrieve n_results most relevant chunks from the static corpus.
    Returns list of dicts with 'text', 'metadata' and 'relevance_score' keys.
    Always excludes raw Manifesto Project CSV quasi-sentences (type='manifesto').
    When include_manifestos=False, also excludes PDF party manifesto chunks
    (type='manifesto_pdf'), returning only CPB and PBL report chunks.
    Short junk chunks (< 30 chars) from PDF extraction artifacts are filtered out.
    """
    collection = _get_collection()
    embedding = [embed_query(query)]

    if include_manifestos:
        where = {"type": {"$ne": "manifesto"}}
    else:
        where = {"type": {"$nin": ["manifesto", "manifesto_pdf"]}}

    # Fetch extra to compensate for junk chunks filtered below
    fetch_n = n_results + 5
    results = collection.query(
        query_embeddings=embedding,
        n_results=fetch_n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if len(doc.strip()) < 30:
            continue
        passages.append(
            {
                "text": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 3),
            }
        )
    return passages[:n_results]


def format_for_prompt(passages: list[dict]) -> str:
    """Format retrieved passages for inclusion in an LLM prompt."""
    parts = []
    for i, p in enumerate(passages):
        meta = p["metadata"]
        citation = f"[{meta.get('source', '?')}, {meta.get('year', '?')}]"
        parts.append(f"Passage {i + 1} {citation}:\n{p['text']}")
    return "\n\n---\n\n".join(parts)


def _end_year(period: str) -> int:
    """Extract the last 4-digit year from a CBS period string."""
    import re

    years = re.findall(r"\b(19|20)\d{2}\b", period)
    return int(years[-1]) if years else 0


def retrieve_cbs_datasets(queries: list[str] | str, n_results: int = 5) -> list[dict]:
    """
    Semantic search over the CBS catalog.

    Accepts a list of query variants (from the LLM query planner) or a single string.
    Embeds all variants in one batch, merges results, and returns deduplicated
    candidates ranked by best relevance score across all variants.
    Auto-builds the ChromaDB collection from data/catalog/cbs_catalog.jsonl on
    first call if it is not already present (one-time ~2 min, then cached).
    Raises if the JSONL is missing (repo checkout problem, not a runtime fallback).
    """
    collection = _get_cbs_collection()

    variants = [queries] if isinstance(queries, str) else list(dict.fromkeys(q for q in queries if q))
    embeddings = embed_texts(variants)

    seen: dict[str, dict] = {}
    for emb in embeddings:
        results = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            identifier = meta["identifier"]
            score = round(1 - dist, 3)
            if identifier not in seen or score > seen[identifier]["relevance_score"]:
                seen[identifier] = {
                    "identifier": identifier,
                    "title": meta["title"],
                    "period": meta.get("period", ""),
                    "api_url": meta.get("api_url", ""),
                    "relevance_score": score,
                }

    all_candidates = sorted(seen.values(), key=lambda x: x["relevance_score"], reverse=True)

    # Prefer datasets with data from 2015 onwards; fall back to full list if none pass.
    recent = [c for c in all_candidates if _end_year(c["period"]) >= 2015]
    return (recent if recent else all_candidates)[:n_results]


if __name__ == "__main__":
    # Quick test
    results = retrieve_static("woningmarkt betaalbaarheid huurprijzen", n_results=3)
    for r in results:
        print(r["metadata"]["source"], r["relevance_score"])
        print(r["text"][:200])
        print()
