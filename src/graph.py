"""LangGraph graph wiring the political analyst, data analyst, and synthesis."""
import asyncio
import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END
from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.agents.political import run_political_analyst
from src.agents.data import run_data_analyst, CBS_NOT_FOUND

# Product constraint: a user-facing query must answer fast or not at all.
# The data analyst (autonomous CBS StatLine exploration) is the only
# unbounded step, so it gets a hard 60s budget. If it has not produced a
# usable answer by then, it is not worth continuing - degrade to an honest
# not-found and let the political/synthesis path carry the response.
DATA_NODE_TIMEOUT_S = 60


class PolderState(TypedDict):
    query: str
    political_response: str
    political_passages: list
    data_response: str
    final_response: str


def political_node(state: PolderState) -> PolderState:
    """Political analyst node : retrieves from static corpus."""
    t0 = time.monotonic()
    result = run_political_analyst(
        query=state["query"],
        prior_context=state.get("data_response"),
    )
    print(f"DEBUG_LOG: political node took {time.monotonic() - t0:.1f}s")
    return {
        **state,
        "political_response": result["response"],
        "political_passages": result["passages"],
    }


async def data_node(state: PolderState) -> PolderState:
    """Data analyst node : queries CBS via MCP."""
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(
            run_data_analyst(state["query"]), timeout=DATA_NODE_TIMEOUT_S
        )
    except (asyncio.TimeoutError, Exception) as exc:
        print(f"DEBUG_LOG: data node failed/timed out: {type(exc).__name__}: {exc}")
        response = CBS_NOT_FOUND
    print(f"DEBUG_LOG: data node took {time.monotonic() - t0:.1f}s")
    return {**state, "data_response": response}


def synthesis_node(state: PolderState) -> PolderState:
    """Synthesis node : combines political and data responses."""
    t0 = time.monotonic()
    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)

    prompt = f"""You are synthesising two expert responses into a single clear answer.

Query: {state['query']}

Political analyst response:
{state['political_response']}

Data analyst response:
{state['data_response']}

Write a single response that:
- Answers the question directly in the first sentence
- Connects what parliament said to what the data shows
- Keeps all inline citations from both responses
- Flags any disagreement between political claims and statistical evidence
- Is at most 300 words
- Ends with "Sources: [list all cited sources]"

If one of the responses says no information was found, note this clearly."""

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["max_tokens"],
    )

    print(f"DEBUG_LOG: synthesis node took {time.monotonic() - t0:.1f}s")
    return {**state, "final_response": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(PolderState)

    graph.add_node("political", political_node)
    graph.add_node("data", data_node)
    graph.add_node("synthesis", synthesis_node)

    # Political runs first, then data (could be parallelised in v2)
    graph.set_entry_point("political")
    graph.add_edge("political", "data")
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


async def run_query(query: str) -> dict:
    graph = build_graph()
    initial_state = PolderState(
        query=query,
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
