"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go)."""
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agents.config import AGENT_CONFIGS

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
        async with MultiServerMCPClient({
            "cbs": {
                "command": "mcp-cbs-cijfers-open-data",
                "args": ["--stdio"],
                "transport": "stdio",
            }
        }) as mcp_client:
            tools = await mcp_client.get_tools()

            llm = ChatOpenAI(
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                max_tokens=cfg["max_tokens"],
            )

            agent = create_react_agent(llm, tools)

            result = await agent.ainvoke({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            })

            return result["messages"][-1].content
    except (FileNotFoundError, OSError) as exc:
        # CBS MCP binary not installed (Step 8) - degrade honestly rather
        # than crash the whole graph.
        print(f"DEBUG_LOG: CBS MCP server unavailable: {exc}")
        return CBS_NOT_FOUND
