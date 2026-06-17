"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go).

fast mode: fixed pipeline — ChromaDB → parallel get_observations → LLM synthesis (~30-90s)
deep mode: React agent — agent calls search_cbs_catalog (ChromaDB) to find datasets,
           then get_dimensions + query_observations to fetch filtered data (~60-120s)
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

# Hard-pin to the protocol version the CBS Go server supports.
mcp.types.LATEST_PROTOCOL_VERSION = "2024-11-05"

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "data_analyst.txt").read_text()

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


def _tool_text(result) -> str:
    return result[0]["text"] if isinstance(result, list) else str(result)


async def _fetch_observations(identifier: str) -> tuple[str, str | Exception]:
    """Spawn a dedicated MCP session and fetch observations for one dataset."""
    try:
        mcp_client = MultiServerMCPClient(_MCP_CONFIG)
        async with mcp_client.session("cbs") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}
            result = await tools["get_observations"].ainvoke(
                {"catalog": "CBS", "dataset": identifier, "limit": 50}
            )
            return identifier, _tool_text(result)
    except Exception as exc:
        return identifier, exc


def _political_section(political_context: str | None) -> str:
    if not political_context or len(political_context) < 50:
        return ""
    return (
        "\n\nPolitical analyst findings — use the CBS data to corroborate, "
        "contextualise, or contrast these specific claims. Do not summarise the "
        "political findings; only find the numbers that speak to them:\n"
        + political_context[:1200]
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
    DEBUG_LOG(f"DEBUG_LOG: catalog found {len(candidates)} candidates: "
              f"{[c['identifier'] for c in candidates]}")

    if not candidates:
        return CBS_NOT_FOUND

    if on_status:
        names = ", ".join(
            f"{c['identifier']} ({c['title'][:40]})" for c in candidates
        )
        on_status(f"CBS candidates: *{names}*")

    selected_ids = [c["identifier"] for c in candidates]
    DEBUG_LOG(f"DEBUG_LOG: fetching observations for all {len(selected_ids)} candidates in parallel")

    obs_results = await asyncio.gather(*[
        _fetch_observations(did) for did in selected_ids
    ])

    data_blocks = []
    used_labels = []
    for did, result in obs_results:
        if isinstance(result, Exception):
            DEBUG_LOG(f"DEBUG_LOG: get_observations failed for {did}: {result}")
            continue
        c_meta = next((c for c in candidates if c["identifier"] == did), {})
        title = c_meta.get("title", did)
        data_blocks.append(f"**{did} — {title}**\n{result[:1000]}")
        used_labels.append(f"{did} ({title})")

    if not data_blocks:
        return CBS_NOT_FOUND

    transparency = f"\n\n**CBS datasets queried:** {'; '.join(used_labels)}"

    result = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{query}\n\nCBS data:\n\n" + "\n\n---\n\n".join(data_blocks)
            + _political_section(political_context)
            + "\n\nEnd your response with:\n" + transparency
        )},
    ])
    return result.content


async def _run_deep(
    query: str,
    political_context: str | None,
    llm: ChatOpenAI,
    callbacks: list | None = None,
) -> str:
    """React agent: agent searches CBS catalog via ChromaDB tool, evaluates relevance,
    inspects dimensions, and fetches filtered observations."""
    from langgraph.prebuilt import create_react_agent

    political_section = _political_section(political_context)

    user_content = (
        f"User query: {query}\n"
        + political_section
        + "\n\nBased on the query and political findings above, decide what CBS statistical "
        "data would best support, contextualise, or challenge those findings.\n\n"
        "Steps:\n"
        "1. Call search_cbs_catalog 3-5 times with different Dutch statistical search terms "
        "to find relevant datasets (e.g. 'huurprijzen', 'woningvoorraad', 'koopwoningen').\n"
        "2. From the results, select the 2-3 most relevant datasets.\n"
        "3. Call get_dimensions(catalog='CBS', dataset=ID) on each to see available periods.\n"
        "4. Call query_observations(catalog='CBS', dataset=ID, "
        "filter=\"startswith(Perioden,'2020')\", top=50) to get recent data.\n"
        "   Use 'filter' and 'top' as parameter names (no dollar signs).\n"
        "5. Present findings with inline [DatasetID] citations."
    )

    _EXCLUDE = {"query_datasets", "get_catalogs", "get_metadata"}
    mcp_client = MultiServerMCPClient(_MCP_CONFIG)
    async with mcp_client.session("cbs") as session:
        mcp_tools = [t for t in await load_mcp_tools(session) if t.name not in _EXCLUDE]
        tools = [search_cbs_catalog] + mcp_tools
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke(
            {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]},
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
    if mode == "fast":
        queries = cbs_queries if cbs_queries else [query]
        return await _run_fast(query, queries, political_context, llm, on_status)

    # Deep: agent drives its own search via search_cbs_catalog tool
    try:
        return await _run_deep(query, political_context, llm, callbacks)
    except Exception as exc:
        print(f"DEBUG_LOG: deep CBS agent failed, falling back to fast: {exc}")
        fast_queries = cbs_queries if cbs_queries else [query]
        return await _run_fast(query, fast_queries, political_context, llm, on_status)
