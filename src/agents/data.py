"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go)."""
from pathlib import Path

import mcp.types
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"- {c['identifier']}: {c['title']} "
            f"(period: {c['period']}, relevance: {c['relevance_score']})"
        )
    return "\n".join(lines)


async def run_data_analyst(query: str, cbs_query: str = "") -> str:
    """
    Run the data analyst agent using the CBS MCP server.

    Two-phase approach:
    1. Semantic search over pre-indexed CBS catalog -> dataset IDs (fast, local)
    2. MCP agent calls get_dimensions + get_observations on those IDs directly
       - no blind search loop, no backtracking, bounded tool calls
    """
    cfg = AGENT_CONFIGS["data_analyst"]

    # Phase 1: catalog lookup (local ChromaDB, ~50ms)
    search_query = cbs_query or query
    candidates = retrieve_cbs_datasets(search_query, n_results=5)
    DEBUG_LOG = print  # keep for next feature

    DEBUG_LOG(f"DEBUG_LOG: catalog found {len(candidates)} candidates: "
              f"{[c['identifier'] for c in candidates]}")
    candidates_block = (
        "The CBS catalog semantic search has pre-identified these relevant datasets:\n"
        + _format_candidates(candidates)
        + "\n\nStart directly with get_dimensions on the most promising dataset. "
        "Do NOT call search_datasets — you already have the right IDs."
    )

    # Phase 2: MCP agent — bounded to get_dimensions + get_observations only
    try:
        mcp_client = MultiServerMCPClient({
            "cbs": {
                "command": "mcp-cbs-cijfers-open-data",
                "args": ["--stdio"],
                "transport": "stdio",
            }
        })

        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            timeout=60,
            max_retries=1,
        )

        async with mcp_client.session("cbs") as session:
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)

            # 2 iterations per tool call (LLM + tool); 20 allows ~3 datasets explored
            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{query}\n\nStatistical focus: {search_query}\n\n{candidates_block}"},
                    ]
                },
                config={"recursion_limit": 30},
            )

        return result["messages"][-1].content
    except (FileNotFoundError, OSError) as exc:
        DEBUG_LOG(f"DEBUG_LOG: CBS MCP server unavailable: {exc}")
        return CBS_NOT_FOUND
