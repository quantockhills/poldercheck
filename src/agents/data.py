"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go)."""
from pathlib import Path

import mcp.types
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agents.config import AGENT_CONFIGS

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


async def run_data_analyst(query: str) -> str:
    """
    Run the data analyst agent using the CBS MCP server.
    Returns a string response with CBS statistics.
    """
    cfg = AGENT_CONFIGS["data_analyst"]

    try:
        # As of langchain-mcp-adapters 0.1.0+, MultiServerMCPClient is not a
        # context manager; sessions are managed per get_tools() call.
        mcp_client = MultiServerMCPClient({
            "cbs": {
                "command": "mcp-cbs-cijfers-open-data",
                "args": ["--stdio"],
                "transport": "stdio",
            }
        })
        tools = await mcp_client.get_tools()

        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            timeout=60,
            max_retries=1,
        )

        agent = create_react_agent(llm, tools)

        # recursion_limit caps the tool-call loop: without it one confused
        # model turn can spiral into a 20+ minute session.
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            },
            config={"recursion_limit": 12},
        )

        return result["messages"][-1].content
    except (FileNotFoundError, OSError) as exc:
        # CBS MCP binary not installed (Step 8) - degrade honestly rather
        # than crash the whole graph.
        print(f"DEBUG_LOG: CBS MCP server unavailable: {exc}")
        return CBS_NOT_FOUND
