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

DATA_NODE_TIMEOUT_S = 90


class PolderState(TypedDict):
    query: str
    language: str    # "nl" | "en"
    mode: str        # "fast" | "deep"
    cbs_queries: list  # LLM-generated Dutch CBS search term variants
    political_response: str
    political_passages: list
    data_response: str
    final_response: str


async def query_planner_node(state: PolderState) -> dict:
    """Generate Dutch CBS statistical search term variants via LLM."""
    cfg = AGENT_CONFIGS["data_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=15)
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": (
                f"Extract the statistical topic from this query as 5-7 Dutch CBS StatLine search terms. "
                f"Ignore political framing. Focus on measurable phenomena — generate semantically diverse "
                f"variants that would match different CBS dataset titles, e.g. for housing affordability: "
                f"huurprijzen, koopwoningen, woningvoorraad, sociale huur, woz waarde, huurmarkt, woningmarkt. "
                f"Return only the Dutch terms, one per line, nothing else.\n\nQuery: {state['query']}"
            )}],
            max_tokens=60,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = response.choices[0].message.content.strip()
        cbs_queries = [t.strip() for t in raw.splitlines() if t.strip()]
    except Exception as exc:
        print(f"DEBUG_LOG: query planner failed, using raw query: {exc}")
        cbs_queries = [state["query"]]
    print(f"DEBUG_LOG: cbs_queries={cbs_queries!r}")
    return {"cbs_queries": cbs_queries}


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
            run_data_analyst(state["query"], cbs_queries=state.get("cbs_queries", [])),
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
- Preserves every inline citation exactly as it appears in the source responses — every number and every political claim must have its [citation] immediately after it in the same sentence; never strip or consolidate inline citations
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
        cbs_queries=[],
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
