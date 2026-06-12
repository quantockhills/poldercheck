import os

from src.ingest.retrieve import retrieve_static, format_for_prompt


def test_retrieve_returns_list():
    # This test only works after build_store() has been run.
    # Skip gracefully if chroma_db does not exist yet.
    if not os.path.exists("./chroma_db"):
        return
    results = retrieve_static("housing affordability", n_results=2)
    assert isinstance(results, list)


def test_format_for_prompt():
    passages = [
        {
            "text": "De woningmarkt staat onder druk.",
            "metadata": {"source": "VVD Manifesto 2023", "year": "2023"},
            "relevance_score": 0.85,
        }
    ]
    formatted = format_for_prompt(passages)
    assert "VVD Manifesto 2023" in formatted
    assert "woningmarkt" in formatted
