"""Political analyst agent.

v1 (run_political_analyst): static ChromaDB corpus only - used by the PoC.
v2 (run_political_analyst_v2): adds live Tweede Kamer search via the OpenTK
MCP server (Step 11).
"""

import asyncio
from pathlib import Path

from openai import OpenAI

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


def run_political_analyst(query: str, prior_context: str | None = None, language: str = "nl") -> dict:
    """
    Run the political analyst agent over the static corpus.
    Returns dict with 'response' and 'passages' keys.
    """
    cfg = AGENT_CONFIGS["political_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)

    # Retrieve relevant passages from static corpus
    passages = retrieve_static(query)
    context = format_for_prompt(passages)

    user_content = f"Query: {query}\n\nRetrieved passages from static corpus:\n\n{context}"

    if prior_context:
        user_content += (
            f"\n\nAdditional context from data analyst:\n{prior_context}\n\nIncorporate this data where relevant."
        )

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": user_content},
        ],
        max_tokens=cfg["max_tokens"],
    )

    return {
        "response": response.choices[0].message.content,
        "passages": passages,
    }


OPENTK_NOT_FOUND = "No relevant recent parliamentary debates found via OpenTK for this query."

OPENTK_TIMEOUT_S = 300

from pathlib import Path

_MCP_CONFIG = {
    "opentk": {
        "command": "node",
        "args": [str(Path(__file__).parent.parent.parent / "docs" / "opentk-mcp" / "dist" / "index.js")],
        "transport": "stdio",
    }
}

_ODATA_BASE = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0"
_MCP_BIN = str(Path(__file__).parent.parent.parent / "docs" / "opentk-mcp" / "dist" / "index.js")


async def discover_odata(
    keywords: list[str] | None = None,
    date_from: str = "2020-01-01",
    date_to: str = "2026-01-01",
    doc_type: str = "Stenogram",
    max_docs: int = 50,
    search_terms: list[str] | None = None,
    parallel: int = 8,
    top_n_for_excerpts: int = 10,
) -> list[dict]:
    """Query the official TK OData API for documents, then analyze relevance via OpenTK MCP.

    Steps:
    1. OData Document search: filter by date range + keyword (Onderwerp) + type (Soort)
    2. For each result, run OpenTK MCP's analyze_document_relevance
    3. For top N results, run find_party_in_document for each party to get excerpts
    4. Return ranked list with scores, parties, excerpts, and metadata.

    Args:
        keywords: Dutch terms to search for in document subject (Onderwerp).
        date_from: Start date (YYYY-MM-DD).
        date_to: End date (YYYY-MM-DD).
        doc_type: Document type filter (e.g. 'Stenogram', 'Motie'). None = all types.
        max_docs: Maximum documents to fetch from OData.
        search_terms: Terms passed to OpenTK for relevance scoring.
        parallel: Max concurrent MCP processes.
        top_n_for_excerpts: How many top docs to extract party snippets from.

    Returns:
        List of dicts, ranked by relevance score descending.
        Each dict: {doc_id, datum, onderwerp, score, parties, n_parties, chars, party_excerpts}
        party_excerpts: {party_name: [snippet_str, ...]}
    """
    import httpx, json, subprocess
    from urllib.parse import quote

    keywords = keywords or ["migratie", "immigratie", "asiel"]
    _search_terms = search_terms or keywords

    # Step 1: OData Document query
    kw_filter = " or ".join(f"contains(Onderwerp,'{k}')" for k in keywords)
    parts = [f"Verwijderd eq false", f"Datum ge {date_from}", f"Datum le {date_to}", f"({kw_filter})"]
    if doc_type:
        parts.append(f"Soort eq '{doc_type}'")
    parts.append("not contains(tolower(Onderwerp),'stemming')")
    odata_filter = " and ".join(parts)

    async with httpx.AsyncClient(timeout=15) as c:
        url = f"{_ODATA_BASE}/Document?$filter={quote(odata_filter)}&$select=DocumentNummer,Onderwerp,Datum&$orderby=Datum asc&$top={max_docs}"
        resp = await c.get(url, headers={"Accept": "application/json"})
        docs = resp.json().get("value", [])

    if not docs:
        return []

    sem = asyncio.Semaphore(parallel)

    async def _mcp_tool(tool: str, args: dict) -> dict | None:
        """Call an OpenTK MCP tool and return the parsed result."""
        proc = await asyncio.create_subprocess_exec(
            "node", _MCP_BIN,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": tool, "arguments": args,
        }})
        try:
            out, _ = await asyncio.wait_for(proc.communicate(input=(req + "\n").encode()), timeout=20)
            data = json.loads(out.decode())
            text = data["result"]["content"][0]["text"]
            return json.loads(text)
        except Exception:
            return None
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    # Step 2: analyze_document_relevance (parallel)
    async def _analyze(doc_id: str):
        async with sem:
            data = await _mcp_tool("analyze_document_relevance", {
                "docId": doc_id, "searchTerms": _search_terms,
            })
            if data:
                return {
                    "score": data.get("relevanceScore", 0),
                    "parties": data.get("entities", {}).get("parties", []),
                    "chars": data.get("statistics", {}).get("characterCount", 0) or 0,
                }
            return {"score": -1, "parties": [], "chars": 0}

    results = []
    tasks = [_analyze(d["DocumentNummer"]) for d in docs]
    for d, analysis in zip(docs, await asyncio.gather(*tasks)):
        if analysis["score"] < 0:
            continue
        results.append({
            "doc_id": d["DocumentNummer"],
            "datum": str(d.get("Datum", ""))[:10],
            "onderwerp": d.get("Onderwerp", ""),
            "score": analysis["score"],
            "parties": analysis["parties"],
            "n_parties": len(analysis["parties"]),
            "chars": analysis["chars"],
            "party_excerpts": {},
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    # Step 3: find_party_in_document for top N docs (parallel per party per doc)
    for rank, doc in enumerate(results[:top_n_for_excerpts]):
        doc["rank"] = rank + 1
        parties = doc["parties"]
        if not parties:
            continue
        party_tasks = [
            _mcp_tool("find_party_in_document", {"docId": doc["doc_id"], "partyName": p})
            for p in parties[:8]  # max 8 parties per doc
        ]
        party_results = await asyncio.gather(*party_tasks)
        excerpts: dict[str, list[str]] = {}
        for party, pdata in zip(parties[:8], party_results):
            if not pdata:
                continue
            occs = pdata.get("occurrences", [])
            snips = [occ.get("snippet", "")[:200] for occ in occs[:3] if occ.get("snippet")]
            if snips:
                excerpts[party] = snips
        doc["party_excerpts"] = excerpts

    return results


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

    Iterative discover: generates multiple Dutch search terms, searches both
    manifestos and Tweede Kamer with each batch, judges relevance across all
    collected sources, fetches top docs, and synthesizes in a single LLM call.
    No workers needed — text sources are small enough to handle directly.

    include_manifestos: when False, static search covers only CPB/PBL (no party PDF manifesto chunks)
    include_tk: when False, skips OpenTK live parliamentary search and uses only the static corpus
    """
    import json as _json, re

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI

    DEBUG_LOG = print

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

    async def _run_iterative() -> dict:
        """Run the political discover via LangGraph subgraph for tracing + state management."""
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

    async def _run_with_opentk() -> dict:
        return await _run_iterative()

    try:
        return await asyncio.wait_for(_run_with_opentk(), timeout=OPENTK_TIMEOUT_S)
    except Exception as exc:
        DEBUG_LOG(f"DEBUG_LOG: OpenTK MCP unavailable, falling back to static-only: {exc}")
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
