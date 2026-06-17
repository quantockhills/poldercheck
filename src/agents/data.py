"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go)."""
import asyncio
from pathlib import Path

import mcp.types
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_cbs_datasets

# The prebuilt CBS MCP server (v0.2.1, built on an early-2025 mcp-go)
# hard-rejects protocol versions it doesn't know instead of negotiating
# down, so offer the one version it supports. The Python client accepts
# 2024-11-05 on the response side, and per spec a modern server offered an
# old version negotiates down, so this is safe process-wide. Remove if the
# CBS server ships a newer release.
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


async def run_data_analyst(query: str, cbs_queries: list[str] | None = None, political_context: str | None = None) -> str:
    """
    Parallel fixed pipeline:

    1. ChromaDB → top 8 candidates (local, ~50ms)
    2. Fast LLM call on titles only → picks top 3
    3. asyncio.gather with separate MCP sessions → get_observations in parallel
    4. Single LLM synthesis with transparency footer
    """
    cfg = AGENT_CONFIGS["data_analyst"]
    DEBUG_LOG = print

    queries = cbs_queries if cbs_queries else [query]
    candidates = retrieve_cbs_datasets(queries, n_results=5)
    DEBUG_LOG(f"DEBUG_LOG: catalog found {len(candidates)} candidates: "
              f"{[c['identifier'] for c in candidates]}")

    if not candidates:
        return CBS_NOT_FOUND

    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        timeout=60,
        max_retries=1,
    )

    # Step 2: parallel get_observations — one MCP process per candidate, no pre-selection
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

    # Step 4: synthesis
    transparency = f"\n\n**CBS datasets queried:** {'; '.join(used_labels)}"

    political_section = ""
    if political_context and len(political_context) > 50:
        political_section = (
            "\n\nPolitical analyst findings — use the CBS data to corroborate, "
            "contextualise, or contrast these specific claims. Do not summarise the "
            "political findings; only find the numbers that speak to them:\n"
            + political_context[:1200]
        )

    result = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{query}\n\nCBS data:\n\n" + "\n\n---\n\n".join(data_blocks)
            + political_section
            + "\n\nEnd your response with:\n" + transparency
        )},
    ])
    return result.content
