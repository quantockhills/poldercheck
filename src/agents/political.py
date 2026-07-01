"""Political analyst agent: live Tweede Kamer search via OpenTK MCP + static ChromaDB retrieval."""

import asyncio
from pathlib import Path

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import format_for_prompt, retrieve_static

_BASE_PROMPT = (Path(__file__).parent.parent / "prompts" / "political_analyst.txt").read_text()

_LANG_EN = """
LANGUAGE: Respond entirely in English.
- Translate all Dutch terms, legislation names, and document titles to English
- When quoting Dutch source material directly, give the English translation first, then the Dutch original in square brackets: "far too little money freed up" [veel te weinig geld vrijgemaakt]
- Dutch legislation: English name with Dutch in brackets on first mention: Affordable Housing Act [Wet betaalbare huur]
- Sources section: translate document titles to English with Dutch original in brackets: "Two-minute Debate on the State of Housing [Tweeminutendebat Staat van de Volkshuisvesting], 26 March 2026"
"""

_LANG_NL = """
LANGUAGE: Respond entirely in Dutch. Source titles and document names stay in Dutch as they appear in the original documents. No translation needed.
"""


def _system_prompt(language: str) -> str:
    from datetime import date

    today = date.today().strftime("%-d %B %Y")
    date_preamble = (
        f"Today's date is {today}. Always include the year of any source you cite. "
        f"Then use judgment: if the question is about *current* party positions or present-day policy, "
        f"flag sources older than 12 months as potentially outdated (party positions may have evolved). "
        f"If the question asks about how views *evolved*, *changed*, or *developed over time*, "
        f"older sources are evidence — cite their year but do not treat age as a limitation.\n\n"
    )
    return date_preamble + _BASE_PROMPT + (_LANG_EN if language == "en" else _LANG_NL)


OPENTK_NOT_FOUND = "No relevant recent parliamentary debates found via OpenTK for this query."

OPENTK_TIMEOUT_S = 300

_ODATA_BASE = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0"
_MCP_BIN = str(Path(__file__).parent.parent.parent / "docs" / "opentk-mcp" / "dist" / "index.js")


async def run_political_analyst_v2(
    query: str,
    prior_context: str | None = None,
    language: str = "nl",
    mode: str = "deep",
    include_manifestos: bool = True,
    include_tk: bool = True,
    callbacks: list | None = None,
) -> dict:
    """
    Political analyst with live OpenTK MCP search + static ChromaDB retrieval.

    Runs the iterative LangGraph discover subgraph (political_discover.py):
    plan → search → synthesize. Falls back to static-only corpus on any failure.

    include_manifestos: when False, static search covers only CPB/PBL (no party PDF manifesto chunks)
    include_tk: when False, skips OpenTK live parliamentary search and uses only the static corpus
    """

    async def _run_iterative() -> dict:
        from src.agents.political_discover import build_political_discover_graph

        graph = build_political_discover_graph()
        initial_state = {
            "query": query,
            "language": language,
            "include_manifestos": include_manifestos,
            "include_tk": include_tk,
            "search_terms": [],
            "date_from": "",
            "date_to": "",
            "static_passages": [],
            "year_buckets": [],
            "odata_keywords": [],
            "odata_results": [],
            "opentk_docs": "",
            "final_response": "",
            "error": None,
        }
        config: dict = {}
        if callbacks:
            config["callbacks"] = callbacks
        result = await graph.ainvoke(initial_state, config=config)
        return {
            "response": result.get("final_response", "No response generated."),
            "passages": result.get("static_passages", []),
        }

    try:
        return await asyncio.wait_for(_run_iterative(), timeout=OPENTK_TIMEOUT_S)
    except Exception as exc:
        print(f"DEBUG_LOG: OpenTK MCP unavailable, falling back to static-only: {exc}")
        from langchain_openai import ChatOpenAI

        static_passages = retrieve_static(query, include_manifestos=include_manifestos)
        static_context = format_for_prompt(static_passages)

        opentk_cfg = AGENT_CONFIGS["opentk_agent"]
        if not opentk_cfg["model"]:
            opentk_cfg = AGENT_CONFIGS["political_analyst"]

        llm = ChatOpenAI(
            base_url=opentk_cfg["base_url"],
            api_key=opentk_cfg["api_key"],
            model=opentk_cfg["model"],
            max_tokens=opentk_cfg["max_tokens"],
            timeout=45,
            max_retries=1,
        ).bind(parallel_tool_calls=True)
        if callbacks:
            llm = llm.with_config({"callbacks": callbacks})

        response = await llm.ainvoke([
            {"role": "system", "content": _system_prompt(language)},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nStatic corpus:\n\n{static_context}\n\n"
                    f"Note: live parliamentary search is unavailable. {OPENTK_NOT_FOUND}"
                ),
            },
        ])
        return {"response": response.content, "passages": static_passages}
