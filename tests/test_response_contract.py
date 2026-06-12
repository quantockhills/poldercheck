"""Deterministic response-contract checks (Step 14, layer 1).

These don't judge response *quality* (that's RAGAS's job); they enforce the
non-negotiables from the system prompts: every response either cites inline
or explicitly says nothing was found, and stays within the word budget.
"""
import re

from src.eval.contract import check_response_contract


def test_cited_response_passes():
    response = (
        "VVD has argued that building regulations should be relaxed "
        "[Manifesto Project: VVD 2023, 2023]. Sources consulted: VVD 2023."
    )
    assert check_response_contract(response) == []


def test_not_found_response_passes():
    response = (
        "I did not find relevant information on this topic in the current "
        "corpus. Other sources may exist that I do not have access to."
    )
    assert check_response_contract(response) == []


def test_uncited_response_fails():
    response = "Housing prices have risen sharply and parties disagree on why."
    violations = check_response_contract(response)
    assert any("citation" in v for v in violations)


def test_overlong_response_fails():
    response = "word " * 400 + "[Some Source, 2023]"
    violations = check_response_contract(response)
    assert any("word budget" in v for v in violations)
