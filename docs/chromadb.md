# ChromaDB

Open-source AI data infrastructure / vector database for storing, querying, and retrieving embeddings.

## Core Capabilities

1. **Document storage** — documents + metadata stored together
2. **Embedding integration** — OpenAI, Cohere, HuggingFace, sentence-transformers, or bring-your-own
3. **Vector search** — dense, sparse, and hybrid similarity search
4. **Keyword/regex search** — pattern matching without embeddings
5. **Metadata filtering** — filter results by metadata conditions at query time
6. **Multi-modal** — images, audio, text

## Key API

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("name", metadata={"hnsw:space": "cosine"})

# Add
collection.add(documents=[...], embeddings=[...], metadatas=[...], ids=[...])

# Query
results = collection.query(query_embeddings=[...], n_results=5,
                           include=["documents", "metadatas", "distances"])
```

## Distance → Similarity

With `hnsw:space=cosine`, Chroma returns cosine *distance* (0=identical, 2=opposite). Convert to similarity: `score = 1 - distance`.

## Usage in poldercheck

Two collections in `./chroma_db/`:
- `poldercheck_static` — manifesto quasi-sentences + CPB/PBL/manifesto PDF chunks (2017–2025)
- `cbs_catalog` — 1,297 CBS StatLine dataset titles/descriptions for semantic catalog search

Both built via `src/ingest/chunk.py` and `scripts/build_cbs_catalog.py`. Queried via `src/ingest/retrieve.py`.

## Links
- Docs: https://docs.trychroma.com
- GitHub: https://github.com/chroma-core/chroma
