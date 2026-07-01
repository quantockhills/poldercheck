"""RAGAS evaluation over the benchmark query set.

Runs the full graph on every query in eval_set.jsonl, then scores with RAGAS:
  - faithfulness      -> synthesis claims are grounded in retrieved context
  - answer_relevancy  -> response actually addresses the query

Also applies the deterministic response contract to every response.
Writes eval_results.json and exits non-zero if thresholds are not met.

Requires: built chroma_db corpus + OPENROUTER_API_KEY. Costs LLM-judge API
calls — run on main / pre-release, not on every push.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# RAGAS unconditionally imports ChatVertexAI from langchain_community, which
# was removed in langchain-community 0.4.x. Stub it out — we never use VertexAI.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from src.eval.contract import check_response_contract  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
RESULTS_FILE = Path("eval_results.json")

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
}


def load_eval_set() -> list[dict]:
    with open(EVAL_SET) as f:
        return [json.loads(line) for line in f if line.strip()]


async def collect_responses(cases: list[dict]) -> list[dict]:
    """Run the graph over all eval queries, collecting response + all context."""
    from src.graph import run_query

    rows = []
    for case in cases:
        print(f"Running: {case['query']}")
        result = await run_query(case["query"])

        # Include all context the synthesis drew from: static passages,
        # political agent output (contains live TK excerpts), and CBS data.
        contexts: list[str] = [p["text"] for p in result.get("political_passages", [])]
        if result.get("political_response"):
            contexts.append(result["political_response"])
        if result.get("data_response"):
            contexts.append(result["data_response"])

        rows.append(
            {
                "user_input": case["query"],
                "response": result["final_response"],
                "retrieved_contexts": contexts,
                "expected_behaviour": case["expected_behaviour"],
                "contract_violations": check_response_contract(result["final_response"]),
            }
        )
    return rows


def score_with_ragas(rows: list[dict]) -> dict:
    """Score collected responses with RAGAS 0.4.x using an OpenRouter judge."""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="anthropic/claude-sonnet-4-6",
        )
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["user_input"],
                response=r["response"],
                retrieved_contexts=r["retrieved_contexts"],
            )
            for r in rows
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=judge,
    )
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def main() -> int:
    if not os.path.exists("./chroma_db"):
        print("No chroma_db found — build it first (scripts/rebuild_embeddings.py). Skipping eval.")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set. Skipping eval.")
        return 0

    cases = load_eval_set()
    rows = asyncio.run(collect_responses(cases))

    contract_failures = [
        {"query": r["user_input"], "violations": r["contract_violations"]}
        for r in rows
        if r["contract_violations"]
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
