"""LangGraph subgraph for political discover: term generation → OData search → synthesis.

Replaces the hand-rolled _run_discover loop in political.py with a proper LangGraph
subgraph for automatic tracing, state management, and future Send-based fan-out.
"""

import asyncio
import io
import json
import re
import time
import zipfile
from typing import TypedDict
from urllib.parse import quote

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import format_for_prompt, retrieve_static

_ODATA_BASE = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0"

MAX_ODATA_DOCS_PER_YEAR = 30
DOWNLOAD_PARALLEL = 8  # concurrent full-text downloads per OData bucket
ODATA_EARLIEST_YEAR = 2018
CHUNK_CHARS = 1500
MAX_CHUNKS_PER_DOC = 60
MAX_RANKED_DEBATES = 15
TRIAGE_POOL = 40  # max candidates sent to the LLM triage call, pre-cut by BM25

# All TK document types that are verbatim debate transcripts. 'Verslag van een
# algemeen overleg' is the pre-2021 name for commissiedebat reports.
DEBATE_SOORTEN = [
    "Stenogram",
    "Verslag van een commissiedebat",
    "Verslag van een algemeen overleg",
    "Verslag van een wetgevingsoverleg",
    "Verslag van een notaoverleg",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PoliticalDiscoverState(TypedDict):
    query: str
    language: str
    include_manifestos: bool
    # Plan outputs
    search_terms: list[str]
    odata_keywords: list[str]  # short Dutch root words for OData Onderwerp substring search
    date_from: str
    date_to: str
    static_passages: list[dict]
    year_buckets: list[dict]  # [{date_from, date_to, year_label}, ...] — created by plan
    # Search outputs
    odata_results: list[dict]  # ranked docs with party_excerpts
    # Synthesis
    final_response: str
    coverage_note: str  # non-empty when query predates OData coverage
    error: str | None
    # Debug / observability
    debug: bool
    plan_trace: dict    # timing + params from plan node
    search_trace: dict  # per-bucket OData counts from search node
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
        timeout=600,  # generous — only guards against totally hung requests
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
        if w.strip() and w.strip().isalpha() and 3 <= len(w.strip()) <= 9
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
# 2. Search node — OData by year (parallel via asyncio.gather)
# ---------------------------------------------------------------------------

def _extract_text(blob: bytes) -> str:
    """Extract plain text from a TK document resource (PDF or DOCX)."""
    try:
        if blob[:4] == b"PK\x03\x04":  # docx is a zip container
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", xml)
        else:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(blob))
            text = " ".join(p.extract_text() or "" for p in reader.pages)
        return " ".join(text.split())
    except Exception:
        return ""


async def _fetch_bucket_docs(
    keywords: list[str],
    date_from: str,
    date_to: str,
    max_docs: int = 15,
) -> list[dict]:
    """OData title search for one date bucket; downloads and extracts each doc's full text."""
    kw_filter = " or ".join(f"contains(tolower(Onderwerp),'{k}')" for k in keywords[:5])
    soort_filter = " or ".join(f"Soort eq '{s}'" for s in DEBATE_SOORTEN)
    parts = [
        "Verwijderd eq false",
        f"Datum ge {date_from}",
        f"Datum le {date_to}",
        f"({kw_filter})",
        f"({soort_filter})",
        "not contains(tolower(Onderwerp),'stemming')",
    ]
    odata_filter = " and ".join(parts)

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        url = (
            f"{_ODATA_BASE}/Document?$filter={quote(odata_filter)}"
            f"&$select=Id,DocumentNummer,Onderwerp,Datum&$orderby=Datum desc&$top={max_docs}"
        )
        resp = await c.get(url, headers={"Accept": "application/json"})
        docs = resp.json().get("value", [])
        if not docs:
            return []

        sem = asyncio.Semaphore(DOWNLOAD_PARALLEL)

        async def _download(d: dict):
            async with sem:
                try:
                    r = await c.get(f"{_ODATA_BASE}/Document({d['Id']})/resource")
                    d["text"] = await asyncio.to_thread(_extract_text, r.content)
                except Exception:
                    d["text"] = ""

        await asyncio.gather(*[_download(d) for d in docs])
    return docs


_NL_STOP = {
    "van", "het", "een", "voor", "over", "met", "der", "den", "des", "aan",
    "bij", "naar", "uit", "dat", "die", "deze", "zijn", "niet", "ook", "als",
    "wordt", "worden", "the", "and", "how", "what",
}

_PARTY_PATTERNS = [
    ("GroenLinks-PvdA", r"\bGroenLinks-PvdA\b"),
    ("VVD", r"\bVVD\b"), ("PVV", r"\bPVV\b"), ("CDA", r"\bCDA\b"), ("D66", r"\bD66\b"),
    ("GroenLinks", r"\bGroenLinks\b"), ("PvdA", r"\bPvdA\b"), ("SP", r"\bSP\b"),
    ("ChristenUnie", r"\bChristenUnie\b"), ("SGP", r"\bSGP\b"),
    ("PvdD", r"\bPartij voor de Dieren\b|\bPvdD\b"), ("JA21", r"\bJA21\b"),
    ("NSC", r"\bNSC\b"), ("FVD", r"\bForum voor Democratie\b|\bFVD\b"),
    ("BBB", r"\bBBB\b"), ("Volt", r"\bVolt\b"), ("DENK", r"\bDENK\b"), ("50PLUS", r"\b50PLUS\b"),
]


def _detect_parties(text: str) -> list[str]:
    found = [name for name, pat in _PARTY_PATTERNS if re.search(pat, text)]
    if "GroenLinks-PvdA" in found:
        found = [p for p in found if p not in ("GroenLinks", "PvdA")]
    return found


def _query_tokens(search_terms: list[str], keywords: list[str]) -> list[str]:
    toks = set()
    for phrase in list(search_terms) + list(keywords):
        for t in re.findall(r"[a-zà-ÿ0-9]+", phrase.lower()):
            if len(t) > 2 and t not in _NL_STOP:
                toks.add(t)
    return sorted(toks)


def _bm25_tokens(text: str, q_tokens: list[str]) -> list[str]:
    """Tokenize for BM25 with decompound-lite matching: a document token also emits any
    query token it starts or ends with, so Dutch compounds like 'vrouwenquotum' match
    both 'vrouwen' and 'quotum' queries."""
    toks = [t for t in re.findall(r"[a-zà-ÿ0-9]+", text.lower()) if len(t) > 2]
    extra = []
    for t in toks:
        for q in q_tokens:
            if len(q) >= 5 and q != t and len(t) >= len(q) + 3 and (t.startswith(q) or t.endswith(q)):
                extra.append(q)
    return toks + extra


def _party_excerpts_local(text: str, parties: list[str], q_tokens: list[str]) -> dict:
    """Snippets around party mentions, preferring those containing query terms.
    Replaces the per-party MCP subprocess calls: we already hold the full text."""
    excerpts: dict[str, list[str]] = {}
    tl = text.lower()
    for party in parties[:8]:
        scored = []
        for m in list(re.finditer(re.escape(party.lower()), tl))[:12]:
            start = max(0, m.start() - 100)
            snip = text[start : start + 300]
            hits = sum(1 for q in q_tokens if q in snip.lower())
            scored.append((hits, snip))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for hits, s in scored[:2] if hits > 0]
        if chosen:
            excerpts[party] = chosen
    return excerpts


async def _llm_triage(query: str, results: list[dict]) -> dict | None:
    """One LLM call scoring each candidate's champion passage 0-10 against the query.
    Returns {doc_id: score} or None on failure (caller falls back to BM25 ranking)."""
    from langchain_openai import ChatOpenAI

    cfg = AGENT_CONFIGS.get("opentk_agent") or AGENT_CONFIGS["political_analyst"]
    if not cfg.get("model"):
        cfg = AGENT_CONFIGS["political_analyst"]
    llm = ChatOpenAI(
        base_url=cfg["base_url"], api_key=cfg["api_key"], model=cfg["model"],
        timeout=600, max_retries=1,
    )
    lines = [
        f"{r['doc_id']} | {r['datum']} | {r['onderwerp'][:80]}\n  passage: {r['champion'][:900] or '(no text extracted)'}"
        for r in results
    ]
    prompt = (
        f"Question: {query}\n\n"
        f"Below are {len(results)} Dutch parliamentary debate documents, each with its best-matching passage.\n"
        "Score each document 0-10 for how much it helps answer the question. "
        "10 = directly discusses the question's topic with substance, 0 = unrelated.\n"
        'Reply with ONLY a JSON object mapping document id to integer score, e.g. {"2024D12345": 7}.\n\n'
        + "\n\n".join(lines)
    )
    try:
        resp = await llm.ainvoke([{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", resp.content.strip(), re.DOTALL)
        if not m:
            return None
        out = {}
        for k, v in json.loads(m.group(0)).items():
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                continue
        return out or None
    except Exception as exc:
        print(f"DEBUG_LOG: LLM triage failed, falling back to BM25 ranking: {exc}")
        return None


async def _rank_debates(query: str, docs: list[dict], q_tokens: list[str]) -> list[dict]:
    """BM25 champion chunk per doc + one LLM triage call. Returns the kept debates,
    each carrying its champion passage and locally extracted party excerpts."""
    from rank_bm25 import BM25Okapi

    chunks: list[str] = []
    owner: list[int] = []
    for i, d in enumerate(docs):
        text = d.get("text", "")
        cs = [text[j : j + CHUNK_CHARS] for j in range(0, len(text), CHUNK_CHARS)][:MAX_CHUNKS_PER_DOC]
        for c in cs:
            if len(c) > 200:
                chunks.append(c)
                owner.append(i)

    champion: dict[int, tuple[float, str]] = {}
    if chunks:
        bm25 = BM25Okapi([_bm25_tokens(c, q_tokens) for c in chunks])
        for c, i, s in zip(chunks, owner, bm25.get_scores(q_tokens)):
            if i not in champion or s > champion[i][0]:
                champion[i] = (float(s), c)

    results = []
    for i, d in enumerate(docs):
        bm, champ = champion.get(i, (0.0, ""))
        results.append({
            "doc_id": d["DocumentNummer"],
            "datum": str(d.get("Datum", ""))[:10],
            "onderwerp": d.get("Onderwerp", ""),
            "year_bucket": d.get("year_bucket", ""),
            "bm25": round(bm, 2),
            "champion": champ,
            "score": 0,
            "parties": [],
            "n_parties": 0,
            "chars": len(d.get("text", "")),
            "party_excerpts": {},
        })

    # Pre-cut by BM25 so the triage prompt stays bounded as candidate pools grow
    results.sort(key=lambda r: r["bm25"], reverse=True)
    triage_pool = results[:TRIAGE_POOL]

    triage = await _llm_triage(query, triage_pool)
    if triage:
        for r in results:
            r["score"] = max(0, min(10, triage.get(r["doc_id"], 0))) * 10
        results.sort(key=lambda r: (r["score"], r["bm25"]), reverse=True)
        kept = [r for r in results if r["score"] >= 30][:MAX_RANKED_DEBATES]
        if len(kept) < 3:
            kept = results[:5]
    else:
        results.sort(key=lambda r: r["bm25"], reverse=True)
        for rank, r in enumerate(results):
            r["score"] = max(0, 90 - rank * 10)
        kept = results[:MAX_RANKED_DEBATES]

    by_id = {d["DocumentNummer"]: d for d in docs}
    for r in kept:
        text = by_id[r["doc_id"]].get("text", "")
        r["parties"] = _detect_parties(text)
        r["n_parties"] = len(r["parties"])
        r["party_excerpts"] = _party_excerpts_local(text, r["parties"], q_tokens)
    return kept


async def _search_node(state: PoliticalDiscoverState, config: RunnableConfig | None = None) -> dict:
    """OData discovery (parallel per year via asyncio.gather)."""
    t0 = time.perf_counter()
    debug = state.get("debug", False)
    on_status = None
    if config:
        on_status = (config.get("configurable") or {}).get("on_status")
    search_terms = state.get("search_terms", [])
    date_from_full = state.get("date_from", "2020-01-01")
    date_to_full = state.get("date_to", "2026-01-01")
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
            if len(w) >= 3 and w.isalpha() and w not in _STOP
        ))[:8]
        keywords_source = "fallback"
    print(f"DEBUG_LOG: search keywords ({keywords_source}): {keywords!r}")

    # OData discovery — parallel per year if buckets exist
    bucket_counts: dict[str, int] = {}

    if on_status:
        on_status(f"Searching Tweede Kamer records: *{', '.join(keywords[:5])}*")

    if year_buckets and len(year_buckets) > 1:
        async def _bucket_fetch(b: dict) -> list[dict]:
            try:
                res = await _fetch_bucket_docs(
                    keywords=keywords, date_from=b["date_from"], date_to=b["date_to"],
                    max_docs=MAX_ODATA_DOCS_PER_YEAR,
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

        per_year = await asyncio.gather(*[_bucket_fetch(b) for b in year_buckets])
        raw_docs = [doc for batch in per_year for doc in batch]
    else:
        raw_docs = await _fetch_bucket_docs(
            keywords=keywords, date_from=date_from_full, date_to=date_to_full,
        )
        bucket_counts["all"] = len(raw_docs)

    q_tokens = _query_tokens(search_terms, keywords)
    if raw_docs and on_status:
        on_status(f"Ranking {len(raw_docs)} debates (BM25 + LLM triage)...")
    odata_results = await _rank_debates(state["query"], raw_docs, q_tokens) if raw_docs else []
    if on_status:
        on_status(f"Selected {len(odata_results)} debates")
    print(f"DEBUG_LOG: OData {len(raw_docs)} docs fetched across {len(bucket_counts)} bucket(s), {len(odata_results)} kept after triage")

    search_trace = {
        "keywords_source": keywords_source,
        "keywords": keywords,
        "buckets": bucket_counts,
        "total_odata_docs": len(raw_docs),
        "kept_after_triage": len(odata_results),
        "triage_scores": {r["doc_id"]: r["score"] for r in odata_results},
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    if debug:
        print(f"[TRACE] SEARCH: {search_trace}")

    return {"odata_results": odata_results, "error": None, "search_trace": search_trace}


# ---------------------------------------------------------------------------
# 3. Synthesize node — merge findings, get excerpts, LLM synthesis
# ---------------------------------------------------------------------------

async def _synthesize_node(state: PoliticalDiscoverState, config: RunnableConfig | None = None) -> dict:
    """Format ranked OData results for synthesis, call LLM."""

    from langchain_openai import ChatOpenAI

    t0 = time.perf_counter()
    debug = state.get("debug", False)
    query = state["query"]
    language = state.get("language", "nl")
    odata_results = state.get("odata_results", [])
    static_passages = state.get("static_passages", [])

    cfg = AGENT_CONFIGS.get("political_analyst")
    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        timeout=600,  # generous — only guards against totally hung requests
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
        for doc in odata_results[:MAX_RANKED_DEBATES]:
            parts.append(
                f"[{doc['doc_id']}] {doc['datum']} — {doc['onderwerp'][:100]} "
                f"(relevance: {doc['score']}/100, {doc['n_parties']} parties)\n"
            )
            champ = doc.get("champion", "")
            if champ:
                parts.append(f"  [most relevant passage]: {champ[:800]}\n")
            for party, snips in doc.get("party_excerpts", {}).items():
                for snippet in snips[:2]:
                    parts.append(f"  [{party}]: {snippet}\n")
            parts.append("\n")
    else:
        parts.append("No relevant parliamentary debates found in the official TK database.\n\n")


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
