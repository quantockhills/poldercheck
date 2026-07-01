"""Response-contract checks shared by tests and the eval runner."""

import re

CITATION_PATTERN = re.compile(r"\^\d+")
NOT_FOUND_SENTENCE = "I did not find relevant information"
CBS_NOT_FOUND_SENTENCE = "I could not find a CBS dataset"
CBS_FAILED_SENTENCE = "The CBS data retrieval process failed"
MAX_WORDS = 350  # synthesis budget is 300 words + citations slack


def check_response_contract(response: str) -> list[str]:
    """Returns a list of violations; empty list = pass."""
    violations = []
    has_not_found = (
        NOT_FOUND_SENTENCE in response
        or CBS_NOT_FOUND_SENTENCE in response
        or CBS_FAILED_SENTENCE in response
    )
    if not has_not_found and not CITATION_PATTERN.search(response):
        violations.append("no inline [Source, Year] citation and no explicit not-found")
    if len(response.split()) > MAX_WORDS:
        violations.append("response exceeds word budget")
    return violations
