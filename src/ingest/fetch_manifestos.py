"""Fetch coded party-manifesto quasi-sentences from the Manifesto Project API.

Output: data/processed/manifesto_corpus.csv, one row per quasi-sentence with
its policy category code (cmp_code), party, and election.
"""
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY = os.environ["MANIFESTO_API_KEY"]
BASE_URL = "https://manifesto-project.wzb.eu/api/v1"

# Dutch party codes in the Manifesto Project.
# IMPORTANT: verify against the Manifesto Project codebook party list - a
# wrong code silently returns no data for that party. Consider adding the
# parties that emerged around the 2023 election (BBB, NSC) since ELECTIONS
# includes 202311.
DUTCH_PARTIES = {
    22110: "VVD",
    22320: "PvdA",
    22526: "D66",
    22410: "CDA",
    22720: "GroenLinks",
    22951: "PVV",
    22220: "SP",
    22521: "ChristenUnie",
}

assert len(DUTCH_PARTIES) == 8, "duplicate party code dropped an entry"

# Elections to include (format: YYYYMM)
ELECTIONS = ["202311", "202103", "201703"]  # 2023, 2021, 2017


def fetch_manifesto_corpus(party_id: int, election_date: str) -> list[dict]:
    """Fetch coded quasi-sentences for a party/election from the Manifesto API."""
    params = {
        "api_key": API_KEY,
        "keys[]": f"{party_id}_{election_date}",
    }
    resp = requests.get(f"{BASE_URL}/texts_and_annotations", params=params, timeout=60)
    if resp.status_code != 200:
        print(f"No data for party {party_id}, election {election_date} (HTTP {resp.status_code})")
        return []

    data = resp.json()
    documents = data.get("items") or []
    if not documents:
        print(f"Empty response for party {party_id}, election {election_date}")
        return []

    # The API wraps each requested manifesto in a document object whose
    # "items" key holds the quasi-sentences; older API versions returned the
    # sentence list directly.
    first = documents[0]
    quasi_sentences = first.get("items", []) if isinstance(first, dict) else first

    sentences = []
    for item in quasi_sentences:
        if isinstance(item, dict) and item.get("text") and item.get("cmp_code"):
            sentences.append({
                "text": item["text"],
                "cmp_code": item["cmp_code"],      # policy category code
                "party_id": party_id,
                "party_name": DUTCH_PARTIES.get(party_id, str(party_id)),
                "election": election_date,
                "source": f"Manifesto Project: {DUTCH_PARTIES.get(party_id)} {election_date[:4]}",
                "type": "manifesto",
                "language": "nl",
            })
    return sentences


def fetch_all() -> pd.DataFrame:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    all_sentences = []
    for party_id, party_name in DUTCH_PARTIES.items():
        for election in ELECTIONS:
            print(f"Fetching {party_name} {election}...")
            sentences = fetch_manifesto_corpus(party_id, election)
            all_sentences.extend(sentences)
            print(f"  Got {len(sentences)} quasi-sentences")

    df = pd.DataFrame(all_sentences)
    df.to_csv("data/processed/manifesto_corpus.csv", index=False)
    print(f"Saved {len(df)} quasi-sentences to data/processed/manifesto_corpus.csv")
    return df


if __name__ == "__main__":
    fetch_all()
