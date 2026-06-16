"""LangGraph graph wiring the political analyst, data analyst, and synthesis."""
import asyncio
import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.agents.political import run_political_analyst_v2
from src.agents.data import run_data_analyst, CBS_NOT_FOUND

DATA_NODE_TIMEOUT_S = 60


class PolderState(TypedDict):
    query: str
    language: str   # "nl" | "en"
    mode: str       # "fast" | "deep"
    cbs_query: str  # statistical Dutch reformulation for CBS catalog search
    political_response: str
    political_passages: list
    data_response: str
    final_response: str


async def query_planner_node(state: PolderState) -> dict:
    """Translate user query into Dutch statistical terms for CBS catalog search."""
    cfg = AGENT_CONFIGS["data_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=15)
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": (
                f"Extract the underlying statistical topic from this query as 1-2 Dutch CBS StatLine "
                f"search terms. Ignore political framing (party stances, debates, opinions). "
                f"Focus on the measurable phenomenon: e.g. 'huurprijzen', 'woningvoorraad', 'koopwoningen'. "
                f"Return only the Dutch search terms, space-separated, nothing else.\n\nQuery: {state['query']}"
            )}],
            max_tokens=30,
            extra_body={"thinking": {"type": "disabled"}},
        )
        cbs_query = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"DEBUG_LOG: query planner failed, using raw query: {exc}")
        cbs_query = state["query"]
    print(f"DEBUG_LOG: cbs_query={cbs_query!r}")
    return {"cbs_query": cbs_query}


async def political_node(state: PolderState) -> dict:
    """Political analyst node: static corpus + live OpenTK parliamentary search."""
    t0 = time.monotonic()
    try:
        result = await run_political_analyst_v2(
            query=state["query"],
            language=state.get("language", "nl"),
            mode=state.get("mode", "deep"),
        )
    except Exception as exc:
        print(f"DEBUG_LOG: political node failed: {type(exc).__name__}: {exc}")
        result = {"response": "Political analysis unavailable.", "passages": []}
    print(f"DEBUG_LOG: political node took {time.monotonic() - t0:.1f}s")
    return {
        "political_response": result["response"],
        "political_passages": result["passages"],
    }


async def data_node(state: PolderState) -> dict:
    """Data analyst node : queries CBS via MCP."""
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(
            run_data_analyst(state["query"], cbs_query=state.get("cbs_query", "")),
            timeout=DATA_NODE_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        print(f"DEBUG_LOG: data node failed/timed out: {type(exc).__name__}: {exc}")
        response = CBS_NOT_FOUND
    print(f"DEBUG_LOG: data node took {time.monotonic() - t0:.1f}s")
    return {"data_response": response}


def synthesis_node(state: PolderState) -> dict:
    """Synthesis node : combines political and data responses."""
    t0 = time.monotonic()
    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)

    lang = state.get("language", "nl")
    lang_instruction = (
        "Respond entirely in English. Translate Dutch terms and source titles; "
        "preserve Dutch quotes with English translation first: 'translation' [origineel]."
        if lang == "en" else
        "Respond entirely in Dutch."
    )

    prompt = f"""You are synthesising two expert responses into a single clear answer.

Query: {state['query']}

Political analyst response:
{state['political_response']}

Data analyst response:
{state['data_response']}

Write a single engaging response that:
- Opens with a strong, direct answer in the first sentence
- Uses varied sentence structures — no semicolon-separated lists; build paragraphs with natural connectives ("but", "while", "in contrast", "notably")
- Groups parties by position rather than listing each one individually
- Connects what parliament said to what the data shows
- Keeps all inline citations from both responses
- Flags any disagreement between political claims and statistical evidence
- Is at most 300 words
- Ends with "Sources: [list all cited sources]"

If one of the responses says no information was found, note this clearly.

{lang_instruction}"""

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["max_tokens"],
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )

    print(f"DEBUG_LOG: synthesis node took {time.monotonic() - t0:.1f}s")
    return {"final_response": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(PolderState)

    graph.add_node("query_planner", query_planner_node)
    graph.add_node("political", political_node)
    graph.add_node("data", data_node)
    graph.add_node("synthesis", synthesis_node)

    # query_planner runs first, then political + data fan out in parallel
    graph.add_edge(START, "query_planner")
    graph.add_edge("query_planner", "political")
    graph.add_edge("query_planner", "data")
    graph.add_edge("political", "synthesis")
    graph.add_edge("data", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def _langfuse_callbacks() -> list:
    """Langfuse tracing (Step 13) - active only when keys are configured."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return []
    try:
        # Import path moved between Langfuse major versions.
        try:
            from langfuse.langchain import CallbackHandler  # v3
        except ImportError:
            from langfuse.callback import CallbackHandler  # v2
        return [CallbackHandler()]
    except Exception as exc:
        print(f"DEBUG_LOG: Langfuse tracing disabled: {exc}")
        return []


async def run_query(query: str, language: str = "nl", mode: str = "deep") -> dict:
    graph = build_graph()
    initial_state = PolderState(
        query=query,
        language=language,
        mode=mode,
        cbs_query="",
        political_response="",
        political_passages=[],
        data_response="",
        final_response="",
    )
    result = await graph.ainvoke(
        initial_state,
        config={"callbacks": _langfuse_callbacks()},
    )
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(run_query(
        "What has parliament debated about housing affordability and what does CBS data show?"
    ))
    print(result["final_response"])
