"""Political analyst agent.

v1 (run_political_analyst): static ChromaDB corpus only - used by the PoC.
v2 (run_political_analyst_v2): adds live Tweede Kamer search via the OpenTK
MCP server (Step 11).
"""
from pathlib import Path

from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_static, format_for_prompt

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "political_analyst.txt").read_text()


def run_political_analyst(query: str, prior_context: str | None = None) -> dict:
    """
    Run the political analyst agent over the static corpus.
    Returns dict with 'response' and 'passages' keys.
    """
    cfg = AGENT_CONFIGS["political_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)

    # Retrieve relevant passages from static corpus
    passages = retrieve_static(query, n_results=3)
    context = format_for_prompt(passages)

    user_content = f"Query: {query}\n\nRetrieved passages from static corpus:\n\n{context}"

    if prior_context:
        user_content += (
            f"\n\nAdditional context from data analyst:\n{prior_context}"
            "\n\nIncorporate this data where relevant."
        )

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=cfg["max_tokens"],
    )

    return {
        "response": response.choices[0].message.content,
        "passages": passages,
    }


async def run_political_analyst_v2(query: str, prior_context: str | None = None) -> dict:
    """
    Political analyst with both live OpenTK MCP search and static ChromaDB
    retrieval (Step 11). Requires Node/npx for the opentk-mcp server.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    cfg = AGENT_CONFIGS["political_analyst"]

    # Static retrieval (always runs)
    static_passages = retrieve_static(query, n_results=3)
    static_context = format_for_prompt(static_passages)

    mcp_client = MultiServerMCPClient({
        "opentk": {
            "command": "npx",
            "args": ["-y", "@r-huijts/opentk-mcp"],
            "transport": "stdio",
        }
    })
    tools = await mcp_client.get_tools()

    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
    )

    agent = create_react_agent(llm, tools)

    user_content = (
        f"Query: {query}\n\n"
        f"Retrieved passages from static corpus (manifestos, CPB, PBL):\n\n"
        f"{static_context}\n\n"
        f"Use the search_tk tool to find relevant recent parliamentary debates "
        f"that complement the above. Retrieve at most 3 documents. "
        f"Use analyze_document_relevance before loading full content to avoid "
        f"loading irrelevant documents."
    )

    if prior_context:
        user_content += f"\n\nCBS data context:\n{prior_context}"

    result = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    })

    return {
        "response": result["messages"][-1].content,
        "passages": static_passages,
    }
