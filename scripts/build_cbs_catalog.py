"""
Refresh the CBS StatLine catalog from the OData v4 API.

Fetches from datasets.cbs.nl/odata/v1/CBS/Datasets — same source the MCP
server uses, so identifiers are guaranteed to resolve at query time.

data/catalog/cbs_catalog.jsonl is committed to the repo and auto-indexed into
ChromaDB on first startup. Run this script only to refresh (e.g. quarterly,
or after elections add new datasets).
"""
import json
import re
from pathlib import Path

import chromadb
import requests

from src.ingest.embed import embed_texts

CBS_V4_DATASETS_URL = "https://datasets.cbs.nl/odata/v1/CBS/Datasets"
CHROMA_PATH = "./chroma_db"
CATALOG_COLLECTION = "cbs_catalog"
OUTPUT_PATH = Path("data/catalog/cbs_catalog.jsonl")
PAGE_SIZE = 1000


def _extract_period(description: str) -> str:
    """Extract 'Beschikbaar vanaf: YYYY' from Dutch CBS description text."""
    m = re.search(r'[Bb]eschikbaar vanaf[:\s]+(\d{4})', description or "")
    return m.group(1) if m else ""


def _odata_url(distributions: list) -> str:
    for d in distributions or []:
        if d.get("Format") == "odata":
            return d.get("DownloadUrl", "")
    return ""


def fetch_full_catalog() -> list[dict]:
    """Fetch all active datasets from CBS OData v4."""
    datasets = []
    url = (
        f"{CBS_V4_DATASETS_URL}"
        f"?$top={PAGE_SIZE}&$filter=Status ne 'Gediscontinueerd'"
    )
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        datasets.extend(body.get("value", []))
        print(f"  Fetched {len(datasets)} datasets...")
        url = body.get("@odata.nextLink")
    return datasets


def build_cbs_catalog():
    print("Fetching CBS StatLine catalog from OData v4...")
    datasets = fetch_full_catalog()
    print(f"Total active datasets: {len(datasets)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for ds in datasets:
        description = (ds.get("Description") or "").strip()
        entry = {
            "identifier": ds["Identifier"],
            "title": ds["Title"].strip(),
            "summary": description[:300],
            "period": _extract_period(description),
            "modified": (ds.get("Modified") or "")[:10],  # YYYY-MM-DD
            "language": ds.get("Language", "nl"),
            "status": ds.get("Status", ""),
            "api_url": _odata_url(ds.get("Distributions", [])),
        }
        entries.append(entry)

    with OUTPUT_PATH.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved catalog to {OUTPUT_PATH}")

    print("Building ChromaDB embeddings via OpenRouter...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(CATALOG_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        CATALOG_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    texts, metadatas, ids = [], [], []
    for entry in entries:
        summary = entry["summary"][:200]
        embed_text = f"{entry['title']}. {summary}" if summary else entry["title"]
        texts.append(embed_text)
        metadatas.append({
            "identifier": entry["identifier"],
            "title": entry["title"],
            "period": entry["modified"],  # use modified date for recency filter
            "language": entry["language"],
            "api_url": entry["api_url"],
        })
        ids.append(entry["identifier"])

    print(f"Embedding {len(texts)} entries...")
    embeddings = embed_texts(texts, batch_size=128)

    for i in range(0, len(texts), 1000):
        collection.add(
            documents=texts[i:i + 1000],
            embeddings=embeddings[i:i + 1000],
            metadatas=metadatas[i:i + 1000],
            ids=ids[i:i + 1000],
        )
    print(f"Done. {len(texts)} CBS datasets indexed in '{CATALOG_COLLECTION}'.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    build_cbs_catalog()
