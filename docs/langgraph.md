# LangGraph

Low-level orchestration framework for building stateful, multi-actor LLM applications as graphs.

Source: extracted from installed package `langgraph` via `help(StateGraph)`.

---

## Core Concept

A `StateGraph` is a directed graph where:
- **Nodes** are Python functions `State -> dict` (return only the keys they update)
- **Edges** define execution order and fan-out
- **State** is a shared `TypedDict` — each node reads the full state and writes a partial update

`StateGraph` is a **builder** — you must call `.compile()` before invoking.

---

## StateGraph API

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    query: str
    result: str

graph = StateGraph(MyState)
```

### `add_node(name, fn)`
Register a node. `fn` receives the full state and returns a `dict` of keys to update.

```python
async def my_node(state: MyState) -> dict:
    return {"result": "done"}

graph.add_node("my_node", my_node)
```

### `add_edge(start_key, end_key)`
Add a directed edge. Two forms:

```python
# Sequential: A → B
graph.add_edge("A", "B")

# Fan-out: START → A and START → B (run in parallel)
graph.add_edge(START, "A")
graph.add_edge(START, "B")

# Fan-in: both A and B must complete before C runs
graph.add_edge("A", "C")
graph.add_edge("B", "C")
```

> When multiple start nodes point to the same end node, LangGraph waits for ALL of them to complete before running the end node. This is the fan-out/fan-in pattern used in poldercheck.

### `add_conditional_edges(source, path_fn)`
Dynamic routing based on state value.

### `compile()`
Returns a `CompiledStateGraph` with `invoke()`, `ainvoke()`, `stream()`, `astream()`.

```python
compiled = graph.compile()
result = await compiled.ainvoke(initial_state, config={"callbacks": [...]})
```

---

## Parallel Fan-out Pattern (used in poldercheck)

```python
# query_planner runs first, then political + data run simultaneously
graph.add_edge(START, "query_planner")
graph.add_edge("query_planner", "political")
graph.add_edge("query_planner", "data")
graph.add_edge("political", "synthesis")
graph.add_edge("data", "synthesis")
graph.add_edge("synthesis", END)
```

**Critical**: nodes that run in parallel must return only the keys they own — never spread the full state (`{**state, "key": val}`), which would overwrite the other branch's writes.

---

## Links
- GitHub: https://github.com/langchain-ai/langgraph
- Docs: https://docs.langchain.com/langgraph
