"""Fetch coded party-manifesto quasi-sentences from the Manifesto Project API.

Party codes and election dates are derived from the core dataset rather than
hardcoded: a wrong hardcoded code silently mislabels a party (an early draft
of this file had GroenLinks's code labelled "VVD"), and omitting a party is
exactly what this project promises never to do. Every Dutch party coded by
the Manifesto Project for the selected elections is included.

Output: data/processed/manifesto_corpus.csv, one row per quasi-sentence with
its policy category code (cmp_code), party, and election.

Note: as of corpus version 2025-1, Dutch coverage ends at the 2021 election;
2023 and 2025 manifestos are not yet coded upstream.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["MANIFESTO_API_KEY"]
BASE_URL = "https://manifesto-project.wzb.eu/api/v1"

CORE_VERSION = "MPDS2025a"
CORPUS_VERSION = "2025-1"

# Include elections from this date onward (YYYYMM).
ELECTIONS_FROM = "201703"


def fetch_core_nl() -> list[dict]:
    """All Dutch party/election rows from the core dataset."""
    resp = requests.get(
        f"{BASE_URL}/get_core",
        params={"api_key": API_KEY, "key": CORE_VERSION},
        timeout=120,
    )
    resp.raise_for_status()
    rows = resp.json()
    header = rows[0]
    idx = {name: header.index(name) for name in ("countryname", "party", "partyabbrev", "partyname", "date")}
    nl = []
    for r in rows[1:]:
        if r[idx["countryname"]] == "Netherlands" and str(r[idx["date"]]) >= ELECTIONS_FROM:
            nl.append(
                {
                    "party_id": int(r[idx["party"]]),
                    "abbrev": r[idx["partyabbrev"]] or r[idx["partyname"]],
                    "election": str(r[idx["date"]]),
                }
            )
    return nl


def fetch_texts(party_id: int, election: str, abbrev: str) -> list[dict]:
    """Fetch coded quasi-sentences for one party/election."""
    resp = requests.get(
        f"{BASE_URL}/texts_and_annotations",
        params={
            "api_key": API_KEY,
            "keys[]": f"{party_id}_{election}",
            "version": CORPUS_VERSION,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} for {abbrev} {election}")
        return []

    data = resp.json()
    if data.get("missing_items"):
        return []  # no annotated text in the corpus for this manifesto

    documents = data.get("items") or []
    if not documents:
        return []
    first = documents[0]
    quasi_sentences = first.get("items", []) if isinstance(first, dict) else first

    sentences = []
    for item in quasi_sentences:
        if isinstance(item, dict) and item.get("text") and item.get("cmp_code"):
            sentences.append(
                {
                    "text": item["text"],
                    "cmp_code": item["cmp_code"],  # policy category code
                    "party_id": party_id,
                    "party_name": abbrev,
                    "election": election,
                    "source": f"Manifesto Project: {abbrev} {election[:4]}",
                    "type": "manifesto",
                    "language": "nl",
                }
            )
    return sentences


def fetch_all() -> pd.DataFrame:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    manifestos = fetch_core_nl()
    print(f"{len(manifestos)} Dutch party/election combinations since {ELECTIONS_FROM}")

    all_sentences = []
    for m in manifestos:
        sentences = fetch_texts(m["party_id"], m["election"], m["abbrev"])
        status = f"{len(sentences)} quasi-sentences" if sentences else "no coded text in corpus"
        print(f"  {m['abbrev']:>10} {m['election']}: {status}")
        all_sentences.extend(sentences)

    df = pd.DataFrame(all_sentences)
    df.to_csv("data/processed/manifesto_corpus.csv", index=False)
    print(f"Saved {len(df)} quasi-sentences to data/processed/manifesto_corpus.csv")
    return df


if __name__ == "__main__":
    fetch_all()
