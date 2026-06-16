"""Chunk and embed the static corpus into ChromaDB.

Two ingestion paths: the Manifesto Project CSV (already chunked at
quasi-sentence level) and CPB/PBL PDFs (split into chunks first).
"""
from pathlib import Path

import chromadb
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest.embed import embed_texts

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"

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
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine space: retrieve.py converts distance to a similarity score,
    # which only makes sense for cosine (Chroma defaults to squared L2).
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

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

    # Part B: CPB/PBL PDFs + 2025 party manifestos (split into chunks)
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

    # Part C: 2025 party manifesto PDFs
    manifesto_dir = Path("data/static/manifestos_2025")
    if manifesto_dir.exists():
        for pdf_file in sorted(manifesto_dir.glob("*.pdf")):
            party = pdf_file.stem.replace("_2025", "").upper()
            print(f"Processing manifesto: {pdf_file.name}...")
            docs = splitter.split_documents(PyPDFLoader(str(pdf_file)).load())
            for j, doc in enumerate(docs):
                all_texts.append(doc.page_content)
                all_metadata.append({
                    "source": f"{party} Verkiezingsprogramma 2025",
                    "type": "manifesto_pdf",
                    "party_name": party,
                    "election": "202511",
                    "year": "2025",
                    "language": "nl",
                    "chunk_index": j,
                })
                all_ids.append(f"manifesto_pdf_{party}_2025_{j}")

    if not all_texts:
        print("Nothing to embed - no manifesto CSV and no PDFs found.")
        return

    print(f"Embedding {len(all_texts)} total chunks via OpenRouter...")
    embeddings = embed_texts(all_texts, batch_size=128)

    batch_size = 500
    for i in range(0, len(all_texts), batch_size):
        collection.add(
            documents=all_texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
        )
    print(f"Stored {len(all_texts)} chunks in ChromaDB.")


if __name__ == "__main__":
    build_store()
