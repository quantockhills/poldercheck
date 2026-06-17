"""Political analyst agent.

v1 (run_political_analyst): static ChromaDB corpus only - used by the PoC.
v2 (run_political_analyst_v2): adds live Tweede Kamer search via the OpenTK
MCP server (Step 11).
"""
import asyncio
from pathlib import Path

from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_static, format_for_prompt

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
    return _BASE_PROMPT + (_LANG_EN if language == "en" else _LANG_NL)


def run_political_analyst(query: str, prior_context: str | None = None, language: str = "nl") -> dict:
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
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": user_content},
        ],
        max_tokens=cfg["max_tokens"],
    )

    return {
        "response": response.choices[0].message.content,
        "passages": passages,
    }


OPENTK_NOT_FOUND = (
    "No relevant recent parliamentary debates found via OpenTK for this query."
)

OPENTK_TIMEOUT_S = 150

_MCP_CONFIG = {
    "opentk": {
        "command": "npx",
        "args": ["-y", "@r-huijts/opentk-mcp"],
        "transport": "stdio",
    }
}


async def run_political_analyst_v2(
    query: str,
    prior_context: str | None = None,
    language: str = "nl",
    mode: str = "deep",
) -> dict:
    """
    Political analyst with live OpenTK MCP search + static ChromaDB retrieval.

    mode="fast": fixed pipeline — search → parallel analyze → parallel fetch → 1 LLM call (~25s)
    mode="deep": React agent — flexible multi-step reasoning (~55s, more thorough)
    """
    import re
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    DEBUG_LOG = print

    static_passages = retrieve_static(query, n_results=3)
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

    async def _run_fast() -> dict | None:
        """Fixed pipeline: search → parallel analyze → parallel fetch → 1 LLM call."""
        # Translate user query to Dutch keywords (OpenTK is Dutch-only)
        kw_response = await llm.ainvoke([{
            "role": "user",
            "content": (
                f"Convert this query to 1-2 Dutch keywords for searching the Dutch parliament "
                f"(Tweede Kamer) database. Single words only, no phrases, no explanation.\n\nQuery: {query}"
            )
        }])
        dutch_query = kw_response.content.strip().split('\n')[0].strip()
        DEBUG_LOG(f"DEBUG_LOG: fast pipeline Dutch query: {dutch_query!r}")

        mcp_client = MultiServerMCPClient(_MCP_CONFIG)
        async with mcp_client.session("opentk") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}

            # 1. Search for documents (type=Document gives Document IDs directly)
            search_raw = await tools["search_tk_filtered"].ainvoke({
                "query": dutch_query, "type": "Document", "limit": 6, "format": "full"
            })
            search_text = search_raw[0]["text"] if isinstance(search_raw, list) else str(search_raw)
            doc_ids = list(dict.fromkeys(re.findall(r'\b\d{4}D\d+\b', search_text)))
            DEBUG_LOG(f"DEBUG_LOG: fast pipeline found doc IDs: {doc_ids[:5]}")

            if not doc_ids:
                return None

            # 2. Analyze relevance in parallel
            search_terms = query.split()[:5]
            analyses = await asyncio.gather(*[
                tools["analyze_document_relevance"].ainvoke({
                    "docId": did, "searchTerms": search_terms
                })
                for did in doc_ids[:5]
            ], return_exceptions=True)

            scored = []
            for did, analysis in zip(doc_ids[:5], analyses):
                if isinstance(analysis, Exception):
                    continue
                text = analysis[0]["text"] if isinstance(analysis, list) else str(analysis)
                nums = [int(n) for n in re.findall(r'\b(\d{1,3})\b', text) if int(n) <= 100]
                scored.append((max(nums, default=0), did))
            scored.sort(reverse=True)
            top_ids = [did for _, did in scored[:2]] or doc_ids[:2]

            # 3. Fetch content in parallel
            contents = await asyncio.gather(*[
                tools["get_document_content"].ainvoke({"docId": did, "maxLength": 2500})
                for did in top_ids
            ], return_exceptions=True)

            docs_block = "\n\n---\n\n".join(
                f"Document {did}:\n{c[0]['text'] if isinstance(c, list) else str(c)}"
                for did, c in zip(top_ids, contents)
                if not isinstance(c, Exception)
            )
            if not docs_block:
                return None

            # 4. Single LLM synthesis
            response = await llm.ainvoke([
                {"role": "system", "content": _system_prompt(language)},
                {"role": "user", "content": (
                    f"Query: {query}\n\n"
                    f"Retrieved passages from static corpus (manifestos, CPB, PBL):\n\n{static_context}\n\n"
                    f"Parliamentary documents from Tweede Kamer:\n\n{docs_block}\n\n"
                    f"Cite each parliamentary document by its ID and date."
                )},
            ])
        return {"response": response.content, "passages": static_passages}

    async def _run_deep() -> dict:
        """React agent: flexible multi-step reasoning over OpenTK tools."""
        user_content = (
            f"Query: {query}\n\n"
            f"Retrieved passages from static corpus (manifestos, CPB, PBL):\n\n{static_context}\n\n"
            f"Now use search_tk_filtered (type=Activiteit) to find recent parliamentary debates. "
            f"After search_tk returns results, call analyze_document_relevance on ALL returned "
            f"documents simultaneously in a single response. Then load the top 2 most relevant "
            f"documents using get_document_content. "
            f"Cite the document title and date for any parliamentary source you use."
        )
        if prior_context:
            user_content += f"\n\nCBS data context:\n{prior_context}"

        mcp_client = MultiServerMCPClient(_MCP_CONFIG)
        async with mcp_client.session("opentk") as session:
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [
                    {"role": "system", "content": _system_prompt(language)},
                    {"role": "user", "content": user_content},
                ]},
                config={"recursion_limit": 60},
            )
        response_text = result["messages"][-1].content
        # LangGraph emits this string when it hits the recursion limit mid-run
        if "need more steps" in response_text.lower():
            response_text = (
                "I did not find relevant information on this topic in the current corpus. "
                "Other sources may exist that I do not have access to."
            )
        return {"response": response_text, "passages": static_passages}

    async def _run_with_opentk() -> dict:
        if mode == "fast":
            result = await _run_fast()
            if result is not None:
                return result
            DEBUG_LOG("DEBUG_LOG: fast pipeline returned no results, falling back to deep")
        return await _run_deep()

    try:
        return await asyncio.wait_for(_run_with_opentk(), timeout=OPENTK_TIMEOUT_S)
    except Exception as exc:
        DEBUG_LOG(f"DEBUG_LOG: OpenTK MCP unavailable, falling back to static-only: {exc}")
        response = await llm.ainvoke([
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": (
                f"Query: {query}\n\nStatic corpus:\n\n{static_context}\n\n"
                f"Note: live parliamentary search is unavailable. {OPENTK_NOT_FOUND}"
            )},
        ])
        return {"response": response.content, "passages": static_passages}
