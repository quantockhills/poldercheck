"""LangGraph subgraph for political discover: term generation → OData+OpenTK search → synthesis.

Replaces the hand-rolled _run_discover loop in political.py with a proper LangGraph
subgraph for automatic tracing, state management, and future Send-based fan-out.
"""

import asyncio
import json
import re
import subprocess
import time
from typing import TypedDict
from urllib.parse import quote

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import format_for_prompt, retrieve_static

_ODATA_BASE = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0"
_MCP_BIN: str = ""

ODATA_PARTY_EXCERPTS = 10
MAX_ODATA_DOCS_PER_YEAR = 12
MCP_PARALLEL = 8
ODATA_EARLIEST_YEAR = 2018


def _init_mcp_bin():
    from pathlib import Path

    global _MCP_BIN
    _MCP_BIN = str(Path(__file__).parent.parent.parent / "docs" / "opentk-mcp" / "dist" / "index.js")


_init_mcp_bin()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PoliticalDiscoverState(TypedDict):
    query: str
    language: str
    include_manifestos: bool
    include_tk: bool
    # Plan outputs
    search_terms: list[str]
    odata_keywords: list[str]  # short Dutch root words for OData Onderwerp substring search
    date_from: str
    date_to: str
    static_passages: list[dict]
    year_buckets: list[dict]  # [{date_from, date_to, year_label}, ...] — created by plan
    # Search outputs
    odata_results: list[dict]  # ranked docs with party_excerpts
    opentk_docs: str
    # Synthesis
    final_response: str
    coverage_note: str  # non-empty when query predates OData coverage
    error: str | None
    # Debug / observability
    debug: bool
    plan_trace: dict    # timing + params from plan node
    search_trace: dict  # per-bucket OData counts + MCP stats from search node
    synthesis_trace: dict  # context size + timing from synthesis node


# ---------------------------------------------------------------------------
# 1. Plan node — generate terms, detect date range, search static corpus
# ---------------------------------------------------------------------------

async def _plan_node(state: PoliticalDiscoverState, config: RunnableConfig | None = None) -> dict:
    """Generate Dutch search terms from the query, extract date range, search static corpus."""
    from langchain_openai import ChatOpenAI

    t0 = time.perf_counter()
    debug = state.get("debug", False)
    on_status = None
    if config:
        on_status = (config.get("configurable") or {}).get("on_status")
    query = state["query"]
    include_manifestos = state.get("include_manifestos", False)

    # Extract date range from query
    from datetime import date as _date
    today = _date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_year = today.year

    years = sorted(set(int(y) for y in re.findall(r"\b(20[0-9]{2})\b", query) if 2000 <= int(y) <= 2030))

    # Detect open-ended "since X" / "vanaf X" / "sindsX" anchors — date_to = today
    _since_pat = re.compile(
        r"\b(?:since|vanaf|sinds|na|after|from)\s+(20[0-9]{2})\b", re.IGNORECASE
    )
    _since_match = _since_pat.search(query)

    if _since_match:
        anchor = int(_since_match.group(1))
        date_from = f"{anchor}-01-01"
        date_to = today_str
    elif len(years) >= 2:
        date_from = f"{years[0]}-01-01"
        date_to = f"{years[-1]}-12-31"
    elif years:
        date_from = f"{years[0]}-01-01"
        date_to = f"{years[0] + 1}-01-01"
    else:
        date_from = f"{today_year - 4}-01-01"
        date_to = today_str

    # Clamp to OData coverage — records before ODATA_EARLIEST_YEAR are unavailable
    requested_date_from = date_from
    if date_from < f"{ODATA_EARLIEST_YEAR}-01-01":
        date_from = f"{ODATA_EARLIEST_YEAR}-01-01"
    coverage_note = (
        f"Note: live parliamentary search (Tweede Kamer OData API) covers {ODATA_EARLIEST_YEAR} onwards. "
        f"Records before {ODATA_EARLIEST_YEAR} are not available in this system. "
        f"The query requested data from {requested_date_from[:4]}, but search was limited to {ODATA_EARLIEST_YEAR}–present."
        if requested_date_from < f"{ODATA_EARLIEST_YEAR}-01-01" else ""
    )

    # Create year buckets for parallel OData search — always bucket when range > 1 year
    year_buckets: list[dict] = []
    if _since_match:
        bucket_start = max(anchor, ODATA_EARLIEST_YEAR)
    elif len(years) >= 2:
        bucket_start = max(years[0], ODATA_EARLIEST_YEAR)
    elif years:
        bucket_start = None  # single-year query — no buckets needed
    else:
        bucket_start = today_year - 4  # no date anchor — default to last 5 years

    if bucket_start is not None:
        for y in range(bucket_start, today_year + 1):
            year_buckets.append({
                "date_from": f"{y}-01-01",
                "date_to": f"{y}-12-31",
                "year_label": str(y),
            })

    # LLM setup
    cfg = AGENT_CONFIGS.get("opentk_agent") or AGENT_CONFIGS["political_analyst"]
    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        max_tokens=300,
        timeout=30,
        max_retries=1,
    )

    # Generate search terms + OData root keywords in one call
    term_prompt = (
        f"Query: {query}\n\n"
        "Part 1: Generate 15 diverse Dutch search terms for finding parliamentary debates "
        "relevant to this query. Cover different angles and phrasings. "
        "Return one term per line.\n\n"
        "Then output exactly: ---\n\n"
        "Part 2: List 3-5 short Dutch root words (4-9 characters) that would appear in "
        "the TITLE of a Tweede Kamer debate about this topic. These are used for substring "
        "title search — so each word must be specific enough to isolate this topic, not so "
        "broad that it matches unrelated debates. "
        "Good: words that almost only appear in debates about THIS topic. "
        "Bad: words like 'macht', 'bedrijf', 'veilig', 'beleid' that appear in hundreds of unrelated debates. "
        "Example: for a women's rights query: emancip, quotum, gender, vrouwen. NOT: macht, bedrijf. "
        "For a migration query: migratie, asiel, vreemd. NOT: veilig, beleid. "
        "Return one word per line, nothing else."
    )
    resp = await llm.ainvoke([{"role": "user", "content": term_prompt}])
    raw = resp.content.strip()

    # Split on the --- separator
    if "---" in raw:
        terms_block, kw_block = raw.split("---", 1)
    else:
        terms_block, kw_block = raw, ""

    seen_terms = {t.strip() for t in terms_block.strip().split("\n") if t.strip()}
    odata_keywords = [
        w.strip().lower() for w in kw_block.strip().split("\n")
        if w.strip() and w.strip().isalpha() and 4 <= len(w.strip()) <= 9
    ][:5]

    # Search static corpus
    static_passages: list = []
    if include_manifestos:
        try:
            static_passages = retrieve_static(query, n_results=15, include_manifestos=include_manifestos)
        except Exception:
            pass

    plan_trace = {
        "odata_keywords": odata_keywords,
        "search_terms_count": len(seen_terms),
        "date_from": date_from,
        "date_to": date_to,
        "year_buckets": [b["year_label"] for b in year_buckets],
        "static_passages_count": len(static_passages),
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    if on_status and odata_keywords:
        on_status(f"TK search terms: *{', '.join(odata_keywords)}*")
    print(f"DEBUG_LOG: plan odata_keywords={odata_keywords!r} dates={date_from}→{date_to} buckets={plan_trace['year_buckets']}")
    if debug:
        print(f"[TRACE] PLAN: {plan_trace}")

    return {
        "search_terms": sorted(seen_terms),
        "odata_keywords": odata_keywords,
        "date_from": date_from,
        "date_to": date_to,
        "year_buckets": year_buckets,
        "static_passages": static_passages,
        "coverage_note": coverage_note,
        "error": None,
        "plan_trace": plan_trace,
    }


# ---------------------------------------------------------------------------
# 2. Search node — OData by year (parallel via asyncio.gather) + OpenTK content
# ---------------------------------------------------------------------------

async def _mcp_tool(tool: str, args: dict) -> dict | None:
    """Call an OpenTK MCP tool via subprocess."""
    proc = await asyncio.create_subprocess_exec(
        "node",
        _MCP_BIN,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}})
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


async def _discover_odata_inner(
    keywords: list[str],
    date_from: str,
    date_to: str,
    search_terms: list[str],
    max_docs: int = 15,
    excerpt_docs: int = 10,
) -> list[dict]:
    """Run the full OData → analyze → party excerpts pipeline."""
    kw_filter = " or ".join(f"contains(tolower(Onderwerp),'{k}')" for k in keywords[:5])
    parts = [
        "Verwijderd eq false",
        f"Datum ge {date_from}",
        f"Datum le {date_to}",
        f"({kw_filter})",
        "Soort eq 'Stenogram'",
        "not contains(tolower(Onderwerp),'stemming')",
    ]
    odata_filter = " and ".join(parts)

    async with httpx.AsyncClient(timeout=15) as c:
        url = f"{_ODATA_BASE}/Document?$filter={quote(odata_filter)}&$select=DocumentNummer,Onderwerp,Datum&$orderby=Datum desc&$top={max_docs}"
        resp = await c.get(url, headers={"Accept": "application/json"})
        docs = resp.json().get("value", [])

    if not docs:
        return []

    # analyze_document_relevance (parallel)
    sem = asyncio.Semaphore(MCP_PARALLEL)

    async def _analyze(doc_id: str):
        async with sem:
            data = await _mcp_tool("analyze_document_relevance", {"docId": doc_id, "searchTerms": search_terms or keywords})
            if data:
                return {
                    "score": data.get("relevanceScore", 0),
                    "parties": data.get("entities", {}).get("parties", []),
                    "chars": data.get("statistics", {}).get("characterCount", 0) or 0,
                }
            return {"score": 0, "parties": [], "chars": 0}

    results: list[dict] = []
    tasks = [_analyze(d["DocumentNummer"]) for d in docs]
    for d, analysis in zip(docs, await asyncio.gather(*tasks)):
        results.append(
            {
                "doc_id": d["DocumentNummer"],
                "datum": str(d.get("Datum", ""))[:10],
                "onderwerp": d.get("Onderwerp", ""),
                "score": analysis["score"],
                "parties": analysis["parties"],
                "n_parties": len(analysis["parties"]),
                "chars": analysis["chars"],
                "party_excerpts": {},
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)

    # find_party_in_document for top N
    for doc in results[:excerpt_docs]:
        parties = doc["parties"]
        if not parties:
            continue
        party_tasks = [_mcp_tool("find_party_in_document", {"docId": doc["doc_id"], "partyName": p}) for p in parties[:8]]
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


async def _search_node(state: PoliticalDiscoverState, config: RunnableConfig | None = None) -> dict:
    """OData discovery (parallel per year via asyncio.gather) + OpenTK content search."""
    t0 = time.perf_counter()
    debug = state.get("debug", False)
    on_status = None
    if config:
        on_status = (config.get("configurable") or {}).get("on_status")
    search_terms = state.get("search_terms", [])
    date_from_full = state.get("date_from", "2020-01-01")
    date_to_full = state.get("date_to", "2026-01-01")
    include_tk = state.get("include_tk", False)
    year_buckets = state.get("year_buckets", [])

    # Use explicit OData root keywords from plan node if available;
    # fall back to word-splitting heuristic on search terms.
    odata_keywords = state.get("odata_keywords", [])
    if odata_keywords:
        keywords = odata_keywords[:8]
        keywords_source = "plan"
    else:
        _STOP = {
            "tweede", "kamer", "debat", "politieke", "standpunten", "verandering",
            "naar", "voor", "over", "van", "met", "een", "het", "der", "dat",
            "die", "dit", "wat", "wie", "hoe", "als", "ook", "zijn", "heeft",
            "worden", "wordt", "maar", "door", "bij", "aan", "uit", "niet",
            "meer", "dan", "nog", "wel", "zonder", "tussen", "jaren", "jaar",
            "partij", "partijen", "parlement", "parlementair", "standpunt",
            "beleid", "politiek", "nederland", "nederlands", "nederlandse",
            "dutch", "since", "changed", "change", "view", "views", "how",
        }
        keywords = list(dict.fromkeys(
            w for t in search_terms[:10] for w in t.lower().split()
            if len(w) >= 5 and w.isalpha() and w not in _STOP
        ))[:8]
        keywords_source = "fallback"
    print(f"DEBUG_LOG: search keywords ({keywords_source}): {keywords!r}")

    # OData discovery — parallel per year if buckets exist
    bucket_counts: dict[str, int] = {}

    if on_status:
        on_status(f"Searching Tweede Kamer records: *{', '.join(keywords[:5])}*")

    if year_buckets and len(year_buckets) > 1:
        async def _bucket_search(b: dict) -> list[dict]:
            try:
                res = await _discover_odata_inner(
                    keywords=keywords, date_from=b["date_from"], date_to=b["date_to"],
                    search_terms=search_terms, max_docs=MAX_ODATA_DOCS_PER_YEAR, excerpt_docs=ODATA_PARTY_EXCERPTS,
                )
                for r in res:
                    r["year_bucket"] = b["year_label"]
                bucket_counts[b["year_label"]] = len(res)
                if on_status:
                    on_status(f"TK {b['year_label']}: {len(res)} documents")
                return res
            except Exception:
                bucket_counts[b["year_label"]] = 0
                return []

        per_year = await asyncio.gather(*[_bucket_search(b) for b in year_buckets])
        odata_results = [doc for batch in per_year for doc in batch]
    else:
        odata_results = await _discover_odata_inner(
            keywords=keywords, date_from=date_from_full, date_to=date_to_full,
            search_terms=search_terms,
        )
        bucket_counts["all"] = len(odata_results)

    odata_results.sort(key=lambda r: r["score"], reverse=True)
    print(f"DEBUG_LOG: OData total {len(odata_results)} docs across {len(bucket_counts)} bucket(s)")

    # OpenTK content search (only if include_tk)
    opentk_docs = ""
    if include_tk and search_terms:
        sem = asyncio.Semaphore(MCP_PARALLEL)
        candidate_docs: dict[str, dict] = {}

        for term in search_terms[:10]:
            if on_status:
                on_status(f"Searching debates: *{term[:60]}*")
            try:
                result = await _mcp_tool(
                    "search_tk_filtered",
                    {"query": term, "type": "Document", "limit": 5, "format": "full"},
                )
                if not result:
                    continue
                result_text = result.get("text", "") if isinstance(result, dict) else ""
                if not result_text:
                    continue
                doc_ids = list(dict.fromkeys(re.findall(r"\b\d{4}D\d+\b", str(result_text))))
                for did in doc_ids:
                    candidate_docs.setdefault(did, {"terms": set()})["terms"].add(term)
            except Exception:
                continue

        if candidate_docs:
            if on_status:
                on_status(f"Checking relevance: {len(candidate_docs)} documents")
            async def _analyze_tk(did: str):
                async with sem:
                    data = await _mcp_tool("analyze_document_relevance", {"docId": did, "searchTerms": search_terms[:5]})
                    if data:
                        return did, data.get("relevanceScore", 0), data.get("entities", {}).get("parties", [])
                    return did, -1, []

            tasks = [_analyze_tk(did) for did in list(candidate_docs.keys())[:15]]
            scored = await asyncio.gather(*tasks)
            for did, score, parties in scored:
                candidate_docs[did]["score"] = score
                candidate_docs[did]["parties"] = parties

            sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1].get("score", -1), reverse=True)
            top_ids = [did for did, _ in sorted_docs[:3]]
            if top_ids:
                async def _fetch(did: str):
                    async with sem:
                        d = await _mcp_tool("get_document_content", {"docId": did, "maxLength": 5000})
                        return did, d.get("text", "") if d else ""
                    return did, ""

                contents = await asyncio.gather(*[_fetch(did) for did in top_ids])
                blocks = [f"Document {did}:\n{text}" for did, text in contents if text]
                opentk_docs = "\n\n---\n\n".join(blocks)

    search_trace = {
        "keywords_source": keywords_source,
        "keywords": keywords,
        "buckets": bucket_counts,
        "total_odata_docs": len(odata_results),
        "opentk_docs_chars": len(opentk_docs),
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    if debug:
        print(f"[TRACE] SEARCH: {search_trace}")

    return {"odata_results": odata_results, "opentk_docs": opentk_docs, "error": None, "search_trace": search_trace}


# ---------------------------------------------------------------------------
# 3. Synthesize node — merge findings, get excerpts, LLM synthesis
# ---------------------------------------------------------------------------

async def _synthesize_node(state: PoliticalDiscoverState, config: RunnableConfig | None = None) -> dict:
    """Merge OData and OpenTK results, format for synthesis, call LLM."""

    from langchain_openai import ChatOpenAI

    t0 = time.perf_counter()
    debug = state.get("debug", False)
    query = state["query"]
    language = state.get("language", "nl")
    odata_results = state.get("odata_results", [])
    opentk_docs = state.get("opentk_docs", "")
    static_passages = state.get("static_passages", [])

    cfg = AGENT_CONFIGS.get("political_analyst")
    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        timeout=120,
        max_retries=1,
    )

    # Build synthesis prompt
    parts: list[str] = [f"Query: {query}\n\n"]

    # Static corpus
    if static_passages:
        static_ctx = format_for_prompt(static_passages)
        parts.append(f"Retrieved passages from static corpus (manifestos, CPB, PBL):\n\n{static_ctx}\n\n")
    else:
        parts.append("No relevant static corpus passages found.\n\n")

    # OData results (structured: per doc with scores, parties, excerpts)
    if odata_results:
        parts.append("Parliamentary debates found via official TK database:\n\n")
        for doc in odata_results[:10]:
            parts.append(
                f"[{doc['doc_id']}] {doc['datum']} — {doc['onderwerp'][:100]} "
                f"(relevance: {doc['score']}/100, {doc['n_parties']} parties)\n"
            )
            for party, snips in doc.get("party_excerpts", {}).items():
                for snippet in snips[:2]:
                    parts.append(f"  [{party}]: {snippet}\n")
            parts.append("\n")
    else:
        parts.append("No relevant parliamentary debates found in the official TK database.\n\n")

    # OpenTK content
    if opentk_docs:
        parts.append(f"Parliamentary documents from Tweede Kamer:\n\n{opentk_docs}\n\n")

    # Date range hint
    date_from = state.get("date_from", "")
    date_to = state.get("date_to", "")
    if date_from and date_to:
        parts.append(f"Date range of search: {date_from} to {date_to}.\n")
    coverage_note = state.get("coverage_note", "")
    if coverage_note:
        parts.append(f"{coverage_note}\n")

    parts.append("Cite each parliamentary document by its ID and date.\n")
    parts.append(
        "If the query asks about changes over time, explicitly mention which documents come from which years "
        "and how positions differ across the periods found.\n"
    )

    # System prompt
    from datetime import date as date_cls

    today = date_cls.today().strftime("%-d %B %Y")
    sys_prompt = (
        f"Today's date is {today}. Always include the year of any source you cite. "
        f"Then use judgment: if the question is about *current* party positions or present-day policy, "
        f"flag sources older than 12 months as potentially outdated. "
        f"If the question asks about how views *evolved* over time, "
        f"older sources are evidence — cite their year but do not treat age as a limitation.\n\n"
    )
    if language == "en":
        sys_prompt += (
            "LANGUAGE: Respond entirely in English.\n"
            "- Translate all Dutch terms, legislation names, and document titles to English\n"
            "- When quoting Dutch source material, give the English translation first, "
            "then the Dutch original in square brackets\n"
            "- Sources section: translate document titles to English with Dutch original in brackets\n"
        )
    else:
        sys_prompt += "LANGUAGE: Respond entirely in Dutch.\n"

    context_str = "".join(parts)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context_str},
        ]
    )

    synthesis_trace = {
        "odata_docs_in_context": min(len(odata_results), 10),
        "static_passages_in_context": len(static_passages),
        "opentk_docs_chars": len(opentk_docs),
        "context_chars": len(context_str),
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    if debug:
        print(f"[TRACE] SYNTHESIS: {synthesis_trace}")

    return {
        "final_response": response.content,
        "error": None,
        "synthesis_trace": synthesis_trace,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_political_discover_graph() -> StateGraph:
    graph = StateGraph(PoliticalDiscoverState)

    graph.add_node("plan", _plan_node)
    graph.add_node("search", _search_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
