"""LangGraph graph wiring the political analyst, data analyst, and synthesis."""

import asyncio
import os
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.agents.data import CBS_NOT_FOUND, run_data_analyst
from src.agents.political import run_political_analyst_v2

DATA_NODE_TIMEOUT_FAST_S = 90
DATA_NODE_TIMEOUT_DEEP_S = 150


class PolderState(TypedDict):
    query: str
    language: str  # "nl" | "en"
    mode: str  # "fast" | "deep"
    pedagogical: bool  # if True, synthesis explains Dutch terms inline
    include_manifestos: bool  # if False, static search uses only CPB/PBL (no party PDFs)
    include_tk: bool  # if False, skips OpenTK live parliamentary search
    include_cbs: bool  # if False, skips CBS data node entirely
    num_datasets: int  # how many CBS datasets to query
    cbs_queries: list  # LLM-generated Dutch CBS search term variants
    political_response: str
    political_passages: list
    data_response: str
    final_response: str


async def query_planner_node(state: PolderState) -> dict:
    """Generate Dutch CBS statistical search term variants via LLM (fast mode only)."""
    if state.get("mode") == "deep":
        # Deep mode: CBS React agent searches for itself; no pre-generated terms needed
        return {"cbs_queries": []}
    cfg = AGENT_CONFIGS["data_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=15)
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extract the statistical topic from this query as 5-7 Dutch CBS StatLine search terms. "
                        f"Ignore political framing. Focus on measurable phenomena — generate semantically diverse "
                        f"variants that would match different CBS dataset titles, e.g. for housing affordability: "
                        f"huurprijzen, koopwoningen, woningvoorraad, sociale huur, woz waarde, huurmarkt, woningmarkt. "
                        f"Return only the Dutch terms, one per line, nothing else.\n\nQuery: {state['query']}"
                    ),
                }
            ],
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
            include_manifestos=state.get("include_manifestos", True),
            include_tk=state.get("include_tk", True),
        )
    except Exception as exc:
        print(f"DEBUG_LOG: political node failed: {type(exc).__name__}: {exc}")
        result = {"response": "Political analysis unavailable.", "passages": []}
    print(f"DEBUG_LOG: political node took {time.monotonic() - t0:.1f}s")
    return {
        "political_response": result["response"],
        "political_passages": result["passages"],
    }


async def data_node(state: PolderState, config=None) -> dict:
    """Data analyst node : queries CBS via MCP.

    Accepts the LangGraph RunnableConfig as an optional second parameter so we
    can extract the outer callbacks (for Stop + status) and pass them into the
    deep-mode React agent.
    """
    if not state.get("include_cbs", True):
        return {"data_response": ""}

    t0 = time.monotonic()
    political = state.get("political_response", "")
    extra = [political] if political and len(political) > 50 else []

    on_status = None
    outer_callbacks: list = []
    if config:
        on_status = (config.get("configurable") or {}).get("on_status")
        outer_callbacks = config.get("callbacks") or []

    mode = state.get("mode", "deep")
    timeout = DATA_NODE_TIMEOUT_FAST_S if mode == "fast" else DATA_NODE_TIMEOUT_DEEP_S
    try:
        response = await asyncio.wait_for(
            run_data_analyst(
                state["query"],
                cbs_queries=state.get("cbs_queries", []) + extra,
                political_context=political if political else None,
                mode=mode,
                num_datasets=state.get("num_datasets", 3),
                on_status=on_status,
                callbacks=outer_callbacks,
            ),
            timeout=timeout,
        )
    except (TimeoutError, Exception) as exc:
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
    pedagogical = state.get("pedagogical", False)

    lang_instruction = (
        "Respond entirely in English. Translate Dutch terms and source titles; "
        "preserve Dutch quotes with English translation first: 'translation' [origineel]."
        if lang == "en"
        else "Respond entirely in Dutch."
    )

    ped_instruction = (
        "\nPedagogical mode is ON: after every Dutch term, policy name, or institutional "
        "abbreviation a non-native speaker may not know, add a brief parenthetical explanation — "
        "e.g. 'uitpondgolf (wave of landlords selling off rental properties)', "
        "'woonquote (share of income spent on housing)', 'CBS (Statistics Netherlands)', "
        "'Tweede Kamer (lower house of parliament)'. Keep each explanation to 5-10 words."
        if pedagogical
        else ""
    )

    has_political = bool(state.get("political_response", "").strip())
    has_data = bool(state.get("data_response", "").strip())

    # Build inputs block — only include sections that have content
    inputs_block = f"Query: {state['query']}\n"
    if has_political:
        inputs_block += f"\nPolitical analyst response:\n{state['political_response']}\n"
    if has_data:
        inputs_block += f"\nData analyst response:\n{state['data_response']}\n"

    # Build the synthesis instruction based on which sources are present
    if has_political and has_data:
        synthesis_bullets = (
            "- Opens by directly answering what was asked: a comparison question gets the comparison first, "
            "a data question gets the numbers first, a policy question gets what parties said first. Never open with your own editorial verdict\n"
            "- Then shows what CBS data says about the same phenomenon\n"
            "- Explicitly flags where the data supports or contradicts what politicians claimed\n"
        )
    elif has_political:
        synthesis_bullets = (
            "- Opens by directly answering what was asked based on the political findings. Never open with your own editorial verdict\n"
            "- Draws on both parliamentary debates and manifesto/policy report passages where available\n"
        )
    else:  # data only
        synthesis_bullets = (
            "- Opens by directly answering what was asked using the CBS data. Lead with the numbers\n"
            "- Explains what the statistics show and what trends they reveal\n"
        )

    prompt = f"""You are synthesising expert research into a single clear answer.

{inputs_block}
Write a single response that:
{synthesis_bullets}- Uses varied sentence structures — no semicolon-separated lists; build paragraphs with natural connectives ("but", "while", "in contrast", "notably")
- Groups parties by position rather than listing each one individually
- Numbers every citation as a superscript in order of first appearance: ^1, ^2, ^3, etc. — place each immediately after the claim it supports. If the same source appears again, reuse the same number. Never drop a citation.
- Is at most 300 words of prose (excluding the sources section)
- Ends with a blank line then "## Sources" followed by a numbered list: "^1 Full source name, Date [ID]"

Only note absence of information if a response contains truly nothing useful. Never open with a meta-comment about what the experts did or did not find.
{ped_instruction}
{lang_instruction}"""

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["max_tokens"],
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )

    print(f"DEBUG_LOG: synthesis node took {time.monotonic() - t0:.1f}s")
    return {"final_response": response.choices[0].message.content}


def _route_political(state: PolderState) -> str:
    """Skip political node entirely when both manifesto and TK toggles are off."""
    if state.get("include_manifestos", True) or state.get("include_tk", True):
        return "political"
    return "data"


def build_graph():
    graph = StateGraph(PolderState)

    graph.add_node("query_planner", query_planner_node)
    graph.add_node("political", political_node)
    graph.add_node("data", data_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "query_planner")
    graph.add_conditional_edges("query_planner", _route_political, {"political": "political", "data": "data"})
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


async def run_query(
    query: str,
    language: str = "nl",
    mode: str = "deep",
    pedagogical: bool = False,
    include_manifestos: bool = True,
    include_tk: bool = True,
    include_cbs: bool = True,
    num_datasets: int = 3,
    extra_callbacks: list | None = None,
    on_status=None,
) -> dict:
    graph = build_graph()
    initial_state = PolderState(
        query=query,
        language=language,
        mode=mode,
        pedagogical=pedagogical,
        include_manifestos=include_manifestos,
        include_tk=include_tk,
        include_cbs=include_cbs,
        num_datasets=num_datasets,
        cbs_queries=[],
        political_response="",
        political_passages=[],
        data_response="",
        final_response="",
    )
    configurable = {}
    if on_status:
        configurable["on_status"] = on_status
    result = await graph.ainvoke(
        initial_state,
        config={
            "callbacks": _langfuse_callbacks() + (extra_callbacks or []),
            "configurable": configurable,
        },
    )
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(
        run_query("What has parliament debated about housing affordability and what does CBS data show?")
    )
    print(result["final_response"])
