"""Chunk and embed the static corpus into ChromaDB.

Two ingestion paths: the Manifesto Project CSV (already chunked at
quasi-sentence level) and CPB/PBL PDFs (split into chunks first).

Resumable: uses get_or_create_collection and skips already-indexed IDs.
Low memory: embeds and stores in batches of EMBED_BATCH, never holding
all embeddings in RAM at once.
"""
from pathlib import Path

import chromadb
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.ingest.embed import embed_texts

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"
EMBED_BATCH = 32  # small batches = low peak memory; stored immediately on each batch

PDF_SOURCES = {
    "data/static/cpb_2025.pdf": {
        "source": "CPB Charted Choices 2025-2028",
        "type": "cpb_analysis",
        "year": "2025",
        "language": "nl",
    },
    "data/static/cpb_2027.pdf": {
        "source": "CPB Charted Choices 2027-2030",
        "type": "cpb_analysis",
        "year": "2027",
        "language": "nl",
    },
    "data/static/pbl_climate.pdf": {
        "source": "PBL Climate and Energy Analysis",
        "type": "pbl_analysis",
        "year": "2023",
        "language": "nl",
    },
}


def build_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # get_or_create preserves any already-indexed chunks on restart
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # Load existing IDs so we can skip them
    existing_ids = set(collection.get(include=[])["ids"])
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} chunks already indexed, skipping those.")

    all_texts, all_metadata, all_ids = [], [], []

    # Part A: Manifesto Project CSV (quasi-sentences, already chunked)
    manifesto_csv = Path("data/processed/manifesto_corpus.csv")
    if manifesto_csv.exists():
        df = pd.read_csv(manifesto_csv)
        df = df[df["cmp_code"] != "H"]  # drop section headers — no policy content
        print(f"Loading {len(df)} manifesto quasi-sentences...")
        for i, row in df.iterrows():
            all_texts.append(row["text"])
            all_metadata.append({
                "source": row["source"],
                "type": "manifesto",
                "party_name": row["party_name"],
                "election": str(row["election"]),
                "cmp_code": str(row["cmp_code"]),
                "year": str(row["election"])[:4],
                "language": "nl",
            })
            all_ids.append(f"manifesto_{row['party_id']}_{row['election']}_{i}")
    else:
        print("No manifesto CSV found : run fetch_manifestos.py first")

    # Part B: CPB/PBL PDFs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    for pdf_path, meta in PDF_SOURCES.items():
        if not Path(pdf_path).exists():
            print(f"Missing: {pdf_path} : skipping")
            continue
        print(f"Processing {pdf_path}...")
        docs = splitter.split_documents(PyPDFLoader(pdf_path).load())
        for j, doc in enumerate(docs):
            all_texts.append(doc.page_content)
            all_metadata.append({**meta, "chunk_index": j})
            all_ids.append(f"{meta['type']}_{meta['year']}_{j}")

    # Part C: party manifesto PDFs (all election years)
    election_map = {
        "2017": "201703",
        "2021": "202103",
        "2023": "202311",
        "2025": "202511",
    }
    for year, election_code in election_map.items():
        manifesto_dir = Path(f"data/static/manifestos_{year}")
        if not manifesto_dir.exists():
            continue
        for pdf_file in sorted(manifesto_dir.glob("*.pdf")):
            party = pdf_file.stem.replace(f"_{year}", "").upper()
            print(f"Processing manifesto: {pdf_file.name}...")
            try:
                docs = splitter.split_documents(PyPDFLoader(str(pdf_file)).load())
            except Exception as e:
                print(f"  SKIPPED (corrupt PDF): {e}")
                continue
            for j, doc in enumerate(docs):
                all_texts.append(doc.page_content)
                all_metadata.append({
                    "source": f"{party} Verkiezingsprogramma {year}",
                    "type": "manifesto_pdf",
                    "party_name": party,
                    "election": election_code,
                    "year": year,
                    "language": "nl",
                    "chunk_index": j,
                })
                all_ids.append(f"manifesto_pdf_{party}_{year}_{j}")

    if not all_texts:
        print("Nothing to embed - no manifesto CSV and no PDFs found.")
        return

    # Filter to only new chunks
    new_texts, new_metadata, new_ids = [], [], []
    for text, meta, cid in zip(all_texts, all_metadata, all_ids):
        if cid not in existing_ids:
            new_texts.append(text)
            new_metadata.append(meta)
            new_ids.append(cid)

    if not new_texts:
        print(f"All {len(all_ids)} chunks already indexed. Nothing to do.")
        return

    print(f"Embedding {len(new_texts)} new chunks ({len(existing_ids)} already done)...")

    # Embed + store in small batches immediately — low peak memory, crash-safe
    for i in tqdm(range(0, len(new_texts), EMBED_BATCH), desc="Embedding", unit="batch"):
        batch_texts = new_texts[i:i + EMBED_BATCH]
        batch_meta = new_metadata[i:i + EMBED_BATCH]
        batch_ids = new_ids[i:i + EMBED_BATCH]
        embeddings = embed_texts(batch_texts, batch_size=EMBED_BATCH)
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )

    print(f"Done. Total indexed: {collection.count()} chunks.")


if __name__ == "__main__":
    build_store()
