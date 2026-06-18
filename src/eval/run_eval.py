"""RAGAS evaluation over the benchmark query set (Step 14, layer 2).

Runs the full graph on every query in eval_set.jsonl, then scores with RAGAS:
  - faithfulness      -> "responses are anchored to retrieved text"
  - answer_relevancy  -> response actually addresses the query
  - context_precision -> retrieval quality (are top chunks relevant?)

Also applies the deterministic response contract to every response.
Writes eval_results.json and exits non-zero if thresholds are not met.

Requires: built chroma_db corpus + OPENROUTER_API_KEY. Costs LLM-judge API
calls - run on main / pre-release, not on every push.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from src.eval.contract import check_response_contract

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
RESULTS_FILE = Path("eval_results.json")

# Starting thresholds - tune after the first real run.
THRESHOLDS = {
    "faithfulness": 0.80,
    "context_precision": 0.60,
}


def load_eval_set() -> list[dict]:
    with open(EVAL_SET) as f:
        return [json.loads(line) for line in f if line.strip()]


async def collect_responses(cases: list[dict]) -> list[dict]:
    """Run the graph over all eval queries, collecting response + contexts."""
    from src.graph import run_query

    rows = []
    for case in cases:
        print(f"Running: {case['query']}")
        result = await run_query(case["query"])
        rows.append(
            {
                "user_input": case["query"],
                "response": result["final_response"],
                "retrieved_contexts": [p["text"] for p in result.get("political_passages", [])],
                "expected_behaviour": case["expected_behaviour"],
                "contract_violations": check_response_contract(result["final_response"]),
            }
        )
    return rows


def score_with_ragas(rows: list[dict]) -> dict:
    """Score collected responses with RAGAS using an OpenRouter judge."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="anthropic/claude-sonnet-4-6",
        )
    )

    dataset = Dataset.from_list(
        [
            {
                "question": r["user_input"],
                "answer": r["response"],
                "contexts": r["retrieved_contexts"],
            }
            for r in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge,
    )
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def main() -> int:
    if not os.path.exists("./chroma_db"):
        print("No chroma_db corpus found - build it first (src/ingest/chunk.py). Skipping eval.")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set. Skipping eval.")
        return 0

    cases = load_eval_set()
    rows = asyncio.run(collect_responses(cases))

    contract_failures = [
        {"query": r["user_input"], "violations": r["contract_violations"]} for r in rows if r["contract_violations"]
    ]

    scores = score_with_ragas(rows)

    output = {
        "scores": scores,
        "thresholds": THRESHOLDS,
        "contract_failures": contract_failures,
        "n_cases": len(rows),
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))

    print("\n=== Eval results ===")
    for metric, value in scores.items():
        threshold = THRESHOLDS.get(metric)
        marker = ""
        if threshold is not None:
            marker = "PASS" if value >= threshold else "FAIL"
        print(f"  {metric}: {value:.3f} {marker}")
    if contract_failures:
        print(f"  contract failures: {len(contract_failures)}")
        for failure in contract_failures:
            print(f"    {failure['query']}: {failure['violations']}")

    failed = contract_failures or any(scores.get(m, 1.0) < t for m, t in THRESHOLDS.items())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
