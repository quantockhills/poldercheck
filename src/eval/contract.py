"""Response-contract checks shared by tests and the eval runner."""

import re

# Accept ASCII footnote citations (^1) and Unicode superscript digits (¹ ² ³),
# which models emit interchangeably for the same footnote instruction.
CITATION_PATTERN = re.compile(r"\^\d+|[⁰¹²³⁴-⁹]")
NOT_FOUND_SENTENCE = "I did not find relevant information"
CBS_NOT_FOUND_SENTENCE = "I could not find a CBS dataset"
CBS_FAILED_SENTENCE = "The CBS data retrieval process failed"
# The synthesis prompt caps prose at 350 words *excluding* the sources
# section, so the contract measures the same thing: the "## Sources" list
# grows with citation count, and a well-cited answer must not breach the
# budget for it. 50 words of slack absorbs superscripts and headings.
MAX_WORDS = 400
SOURCES_MARKER = "## Sources"


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
    prose = response.split(SOURCES_MARKER, 1)[0]
    if len(prose.split()) > MAX_WORDS:
        violations.append("response exceeds word budget")
    return violations
