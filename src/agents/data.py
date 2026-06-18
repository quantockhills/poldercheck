"""Data analyst agent: queries CBS StatLine.

fast mode: fixed pipeline — ChromaDB → parallel direct OData v4 HTTP → LLM synthesis (~30-90s)
deep mode: React agent — search_cbs_catalog (ChromaDB) for discovery,
           then CBS MCP server tools (get_dimensions, query_observations) for data (~60-120s)
"""

import asyncio
from pathlib import Path

import mcp.types
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_cbs_datasets


@tool
def search_cbs_catalog(query: str) -> str:
    """Search the CBS StatLine catalog for relevant datasets using semantic search.
    Returns up to 5 dataset IDs and titles ranked by relevance.
    Call this with a Dutch statistical topic (e.g. 'huurprijzen', 'woningvoorraad').
    """
    candidates = retrieve_cbs_datasets([query], n_results=5)
    if not candidates:
        return "No matching CBS datasets found for this query."
    lines = [f"- {c['identifier']}: {c['title']}" for c in candidates]
    return "\n".join(lines)


@tool
def get_cbs_measure_labels(dataset_id: str) -> str:
    """Fetch human-readable measure labels for a CBS dataset.
    Returns measure codes with their names and units (e.g. 'D001503: Mediane koopkrachtontwikkeling (%)').
    Call this after get_dimensions to understand what the measure IDs mean.
    """
    import httpx
    try:
        resp = httpx.get(
            f"https://datasets.cbs.nl/odata/v1/CBS/{dataset_id}/MeasureCodes",
            timeout=15,
        )
        resp.raise_for_status()
        measures = resp.json()["value"]
        lines = [f"{m['Identifier']}: {m['Title']} ({m['Unit']})" for m in measures]
        return "\n".join(lines) if lines else f"No measures found for {dataset_id}."
    except Exception as exc:
        return f"Failed to fetch measure labels for {dataset_id}: {exc}"


# Hard-pin to the protocol version the CBS Go server supports.
mcp.types.LATEST_PROTOCOL_VERSION = "2024-11-05"

_SYSTEM_PROMPT_BASE = (Path(__file__).parent.parent / "prompts" / "data_analyst.txt").read_text()


def _system_prompt() -> str:
    from datetime import date

    today = date.today().strftime("%-d %B %Y")
    date_preamble = (
        f"Today's date is {today}. Always state the period that CBS data covers. "
        f"Then use judgment: if the question is about *current* statistics or present-day figures, "
        f"flag datasets where the most recent observation is more than 2 years old. "
        f"If the question asks about trends, historical change, or evolution over time, "
        f"older data is part of the answer — present it with its period, do not treat its age as a problem.\n\n"
    )
    return date_preamble + _SYSTEM_PROMPT_BASE


CBS_NOT_FOUND = (
    "I could not find a CBS dataset relevant to this query. The data may exist "
    "under a different search term, or may not be available in CBS StatLine."
)

_MCP_CONFIG = {
    "cbs": {
        "command": "mcp-cbs-cijfers-open-data",
        "args": ["--stdio"],
        "transport": "stdio",
    }
}


async def _fetch_observations(identifier: str) -> tuple[str, str | Exception]:
    """Fetch CBS dataset via direct OData v4 HTTP — labelled measures, all years."""
    try:
        return identifier, await _fetch_cbs_data(identifier)
    except Exception as exc:
        return identifier, exc


async def _fetch_cbs_data(dataset_id: str) -> str:
    """Direct OData v4: get measure codes + dimensions + labelled observations."""
    import httpx

    base = f"https://datasets.cbs.nl/odata/v1/CBS/{dataset_id}"
    async with httpx.AsyncClient(timeout=30) as c:
        # 1. Measure codes (labels + units)
        mc_resp = await c.get(f"{base}/MeasureCodes")
        mc_resp.raise_for_status()
        measures = {m["Identifier"]: f"{m['Title']} ({m['Unit']})" for m in mc_resp.json()["value"]}

        # 2. Dimensions (is there a Perioden dimension?)
        dims_resp = await c.get(f"{base}/Dimensions")
        dims_resp.raise_for_status()
        dimensions = [d["Identifier"] for d in dims_resp.json()["value"]]
        has_perioden = "Perioden" in dimensions

        # 3. Observations: annual data if Perioden exists, else top 100
        if has_perioden:
            obs_url = f"{base}/Observations?$filter=endswith(Perioden,'JJ00')&$orderby=Perioden,Measure"
        else:
            obs_url = f"{base}/Observations?$top=100&$orderby=Measure"
        obs_resp = await c.get(obs_url)
        if obs_resp.status_code != 200:
            obs_resp = await c.get(f"{base}/Observations?$top=100")
        obs_resp.raise_for_status()
        observations = obs_resp.json()["value"]

    # 4. Build labelled output, grouped by measure
    dims_line = f"Dimensions: {', '.join(dimensions)}" if dimensions else "No dimensions found"
    lines = [f"Dataset {dataset_id}", dims_line, ""]

    if measures:
        lines.append("Measures:")
        for k, v in measures.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    by_measure: dict[str, list] = {}
    for row in observations:
        by_measure.setdefault(row["Measure"], []).append(row)

    for meas_id, rows in by_measure.items():
        label = measures.get(meas_id, meas_id)
        vals = []
        for r in rows:
            period = r["Perioden"]
            year = period[:4] if has_perioden else period
            vals.append(f"{year}={r.get('Value', '')}")
        lines.append(f"{label}:")
        lines.append(f"  {', '.join(vals[:40])}")
        if len(vals) > 40:
            lines.append(f"  ... and {len(vals) - 40} more periods")
        lines.append("")

    return "\n".join(lines)


def _political_section(political_context: str | None) -> str:
    if not political_context or len(political_context) < 50:
        return ""
    return (
        "\n\nPolitical analyst findings — use the CBS data to corroborate, "
        "contextualise, or contrast these specific claims. Do not summarise the "
        "political findings; only find the numbers that speak to them:\n" + political_context[:1200]
    )


async def _run_fast(
    query: str,
    cbs_queries: list[str],
    political_context: str | None,
    llm: ChatOpenAI,
    on_status=None,
) -> str:
    """Fixed pipeline: ChromaDB → parallel get_observations → LLM synthesis."""
    DEBUG_LOG = print
    candidates = retrieve_cbs_datasets(cbs_queries, n_results=5)
    DEBUG_LOG(f"DEBUG_LOG: catalog found {len(candidates)} candidates: {[c['identifier'] for c in candidates]}")

    if not candidates:
        return CBS_NOT_FOUND

    if on_status:
        names = ", ".join(f"{c['identifier']} ({c['title'][:40]})" for c in candidates)
        on_status(f"CBS candidates: *{names}*")

    selected_ids = [c["identifier"] for c in candidates]
    DEBUG_LOG(f"DEBUG_LOG: fetching observations for all {len(selected_ids)} candidates in parallel")

    obs_results = await asyncio.gather(*[_fetch_observations(did) for did in selected_ids])

    data_blocks = []
    used_labels = []
    for did, result in obs_results:
        if isinstance(result, Exception):
            DEBUG_LOG(f"DEBUG_LOG: get_observations failed for {did}: {result}")
            continue
        c_meta = next((c for c in candidates if c["identifier"] == did), {})
        title = c_meta.get("title", did)
        period = c_meta.get("period", "")
        period_str = f", {period}" if period else ""
        data_blocks.append(f"**{did} — {title}**\n{result[:1000]}")
        used_labels.append(f"{did} ({title}{period_str})")

    if not data_blocks:
        return CBS_NOT_FOUND

    transparency = f"\n\n**CBS datasets queried:** {'; '.join(used_labels)}"

    result = await llm.ainvoke(
        [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{query}\n\nCBS data:\n\n"
                    + "\n\n---\n\n".join(data_blocks)
                    + _political_section(political_context)
                    + "\n\nEnd your response with:\n"
                    + transparency
                ),
            },
        ]
    )
    return result.content


async def _run_deep(
    query: str,
    cbs_queries: list[str],
    political_context: str | None,
    llm: ChatOpenAI,
    num_datasets: int = 3,
    callbacks: list | None = None,
) -> str:
    """React agent: ChromaDB catalog discovery + CBS MCP server data fetching."""
    from langgraph.prebuilt import create_react_agent

    political_section = _political_section(political_context)
    standalone = not political_section
    search_hints = ", ".join(cbs_queries[:7]) if cbs_queries else query

    # Warm up the ChromaDB catalog collection before launching the MCP client.
    # The React agent may call search_cbs_catalog from a nested async context where
    # instantiating ChromaDB's Rust bindings fails; caching the collection here avoids that.
    try:
        retrieve_cbs_datasets([search_hints], n_results=1)
    except Exception as exc:
        print(f"DEBUG_LOG: could not warm up CBS catalog collection: {exc}")

    _STEPS = (
        f"Steps:\n"
        "1. Call search_cbs_catalog 3-5 times with different Dutch statistical search terms "
        f"to find relevant datasets. Suggested starting terms: {search_hints}.\n"
        f"2. Select up to {num_datasets} datasets. For each, call get_dimensions then get_cbs_measure_labels and "
        "get_dimension_values for every dimension. Never filter without first looking up the values.\n"
        "3. Call query_observations with OData parameters suited to the question.\n"
        "4. Present findings with inline [DatasetID, period] citations."
    )

    if standalone:
        user_content = (
            f"User query: {query}\n\n"
            "Find and present the most relevant CBS statistical data to directly answer this query. "
            "Interpret the query broadly — find datasets that shed light on the topic from multiple angles.\n\n"
            + _STEPS
        )
    else:
        user_content = (
            f"User query: {query}\n"
            + political_section
            + "\n\nBased on the query and political findings above, decide what CBS statistical "
            "data would best support, contextualise, or challenge those findings. "
            "Prioritise datasets that speak directly to claims made in the political findings.\n\n" + _STEPS
        )

    mcp_client = MultiServerMCPClient(_MCP_CONFIG)
    async with mcp_client.session("cbs") as session:
        all_tools = await load_mcp_tools(session)
        allowed = {"get_dimensions", "get_dimension_values", "query_observations"}
        mcp_tools = [t for t in all_tools if t.name in allowed]

        # Per-tool timeout: the MCP server processes tool calls serially over
        # stdio, so one slow CBS API response can block every subsequent tool
        # call. Wrapping with a timeout lets fast calls return independently.
        for t in mcp_tools:
            original_coro = t.coroutine

            async def _timed(orig=original_coro, **kwargs):
                try:
                    async with asyncio.timeout(30):
                        return await orig(**kwargs)
                except (TimeoutError, asyncio.CancelledError):
                    return (
                        "TIMEOUT: This dataset did not respond within 30s. "
                        "Proceed with the data you already have.",
                        None,
                    )

            t.coroutine = _timed

        tools = [search_cbs_catalog, get_cbs_measure_labels] + mcp_tools
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": user_content},
                ]
            },
            config={
                "recursion_limit": 40,
                "callbacks": callbacks or [],
            },
        )

    response_text = result["messages"][-1].content
    if "need more steps" in response_text.lower():
        return CBS_NOT_FOUND
    return response_text


async def run_data_analyst(
    query: str,
    cbs_queries: list[str] | None = None,
    political_context: str | None = None,
    mode: str = "deep",
    num_datasets: int = 3,
    on_status=None,
    callbacks: list | None = None,
) -> str:
    cfg = AGENT_CONFIGS["data_analyst"]
    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        timeout=60,
        max_retries=1,
    )
    queries = cbs_queries if cbs_queries else [query]

    if mode == "fast":
        return await _run_fast(query, queries, political_context, llm, on_status)

    # Deep: agent drives its own search via search_cbs_catalog tool + CBS MCP
    return await _run_deep(query, queries, political_context, llm, num_datasets, callbacks)
