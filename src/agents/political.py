"""Political analyst agent: live Tweede Kamer search via the OData API + static ChromaDB retrieval."""

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


TK_SEARCH_FAILED_NOTICE = (
    "⚠️ Live parliamentary search failed due to a technical error — "
    "Tweede Kamer debates were NOT searched for this answer."
)


def format_political_trace(trace: dict) -> str:
    """Render a pipeline trace dict as a human-readable debug report."""
    p = trace.get("plan", {})
    s = trace.get("search", {})
    y = trace.get("synthesis", {})
    lines = [
        "=== POLDERCHECK PIPELINE TRACE ===",
        "",
        f"PLAN  ({p.get('duration_s', '?')}s)",
        f"  OData keywords : {p.get('odata_keywords', [])}",
        f"  Search terms   : {p.get('search_terms_count', '?')} generated",
        f"  Date range     : {p.get('date_from', '')} → {p.get('date_to', '')}",
        f"  Year buckets   : {p.get('year_buckets', [])}",
        f"  Static passages: {p.get('static_passages_count', '?')}",
        "",
        f"SEARCH  ({s.get('duration_s', '?')}s)",
        f"  Keywords ({s.get('keywords_source', '?')}): {s.get('keywords', [])}",
        "  OData hits per bucket:",
    ]
    buckets = s.get("buckets", {})
    if buckets:
        for label in sorted(buckets):
            lines.append(f"    {label}: {buckets[label]} docs")
    else:
        lines.append("    (no year buckets)")
    lines += [
        f"  Total OData docs   : {s.get('total_odata_docs', '?')}",
        "",
        f"SYNTHESIS  ({y.get('duration_s', '?')}s)",
        f"  OData docs in context  : {y.get('odata_docs_in_context', '?')}",
        f"  Static passages        : {y.get('static_passages_in_context', '?')}",
        f"  Total context          : ~{y.get('context_chars', '?')} chars",
    ]
    durations = [x.get("duration_s", 0) for x in [p, s, y] if isinstance(x.get("duration_s"), (int, float))]
    lines.append(f"\nTOTAL: {sum(durations):.1f}s")
    return "\n".join(lines)

# Generous outer net around the whole discover subgraph (plan + search + synthesize).
# Not a working budget — only catches totally hung runs.
DISCOVER_TIMEOUT_S = 1800


async def run_political_analyst_v2(
    query: str,
    prior_context: str | None = None,
    language: str = "nl",
    mode: str = "deep",
    include_manifestos: bool = True,
    include_tk: bool = True,
    on_status=None,
    callbacks: list | None = None,
    debug: bool = False,
) -> dict:
    """
    Political analyst with live Tweede Kamer OData search + static ChromaDB retrieval.

    Runs the LangGraph discover subgraph (political_discover.py):
    plan → search (OData) → synthesize. Falls back to static-only corpus on any failure.

    include_manifestos: when False, static search covers only CPB/PBL (no party PDF manifesto chunks)
    include_tk: when False, skips the live TK search entirely and answers from the static corpus
    """

    async def _static_only(note: str, response_prefix: str = "") -> dict:
        from langchain_openai import ChatOpenAI

        static_passages = retrieve_static(query, include_manifestos=include_manifestos)
        static_context = format_for_prompt(static_passages)

        cfg = AGENT_CONFIGS["opentk_agent"]
        if not cfg["model"]:
            cfg = AGENT_CONFIGS["political_analyst"]

        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            timeout=600,
            max_retries=1,
        )
        if callbacks:
            llm = llm.with_config({"callbacks": callbacks})

        response = await llm.ainvoke([
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": f"Query: {query}\n\nStatic corpus:\n\n{static_context}\n\n{note}"},
        ])
        text = response.content if not response_prefix else f"{response_prefix}\n\n{response.content}"
        return {"response": text, "passages": static_passages, "trace": {}}

    if not include_tk:
        return await _static_only(
            "Note: live parliamentary search is disabled for this query, so parliamentary "
            "debates were not searched. Answer only from the static corpus above and mention "
            "that parliamentary debates were not part of this search."
        )

    async def _run_iterative() -> dict:
        from src.agents.political_discover import build_political_discover_graph

        graph = build_political_discover_graph()
        initial_state = {
            "query": query,
            "language": language,
            "include_manifestos": include_manifestos,
            "search_terms": [],
            "date_from": "",
            "date_to": "",
            "static_passages": [],
            "year_buckets": [],
            "odata_keywords": [],
            "odata_results": [],
            "final_response": "",
            "coverage_note": "",
            "error": None,
            "debug": debug,
            "plan_trace": {},
            "search_trace": {},
            "synthesis_trace": {},
        }
        config: dict = {}
        if callbacks:
            config["callbacks"] = callbacks
        if on_status:
            config.setdefault("configurable", {})["on_status"] = on_status
        result = await graph.ainvoke(initial_state, config=config)
        return {
            "response": result.get("final_response", "No response generated."),
            "passages": result.get("static_passages", []),
            "trace": {
                "plan": result.get("plan_trace", {}),
                "search": result.get("search_trace", {}),
                "synthesis": result.get("synthesis_trace", {}),
                # Raw retrieved TK evidence, carried along for eval/debugging only —
                # nothing in the pipeline reads this key.
                "odata_docs": [
                    {
                        "doc_id": d.get("doc_id", ""),
                        "datum": d.get("datum", ""),
                        "onderwerp": d.get("onderwerp", ""),
                        "champion": d.get("champion", ""),
                        "party_excerpts": d.get("party_excerpts", {}),
                    }
                    for d in result.get("odata_results", [])
                ],
            },
        }

    try:
        return await asyncio.wait_for(_run_iterative(), timeout=DISCOVER_TIMEOUT_S)
    except Exception as exc:
        print(f"DEBUG_LOG: TK discover subgraph failed ({type(exc).__name__}: {exc}), falling back to static-only")
        return await _static_only(
            "Note: the live Tweede Kamer search failed due to a technical error, so NO "
            "parliamentary debates could be searched. Answer only from the static corpus "
            "above. State clearly that parliamentary debates were not searched — do NOT "
            "claim or imply that no relevant debates exist.",
            response_prefix=TK_SEARCH_FAILED_NOTICE,
        )
