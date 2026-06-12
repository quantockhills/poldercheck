"""Semantic search over the static ChromaDB corpus."""
from sentence_transformers import SentenceTransformer
import chromadb

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"

_model = None
_collection = None


def _get_collection():
    global _model, _collection
    if _collection is None:
        _model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection, _model


def retrieve_static(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve n_results most relevant chunks from the static corpus.
    Returns list of dicts with 'text', 'metadata' and 'relevance_score' keys.
    """
    collection, model = _get_collection()
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text": doc,
            "metadata": meta,
            "relevance_score": round(1 - dist, 3),  # cosine distance to similarity
        })
    return passages


def format_for_prompt(passages: list[dict]) -> str:
    """Format retrieved passages for inclusion in an LLM prompt."""
    parts = []
    for i, p in enumerate(passages):
        meta = p["metadata"]
        citation = f"[{meta.get('source', '?')}, {meta.get('year', '?')}]"
        parts.append(
            f"Passage {i+1} {citation}:\n{p['text']}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick test
    results = retrieve_static("woningmarkt betaalbaarheid huurprijzen", n_results=3)
    for r in results:
        print(r["metadata"]["source"], r["relevance_score"])
        print(r["text"][:200])
        print()
