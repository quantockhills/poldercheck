"""RAGAS evaluation over the benchmark query set.

Runs the full graph on every enabled query in eval_set.jsonl, capturing the
raw evidence each run actually retrieved (DuckDB tool calls via a callback
tap, TK documents via the political trace), then scores with RAGAS 0.4.x:

  faithfulness       -> response claims are grounded in the retrieved evidence
                        (graded against the FULL tool trace, incl. schema/label
                        lookups — the judge needs those to decode measure IDs)
  context_precision  -> the data-bearing evidence was relevant to the query
                        (graded against a filtered trace: setup calls, schema
                        exploration and failed queries excluded — an agent's
                        chronological tool trace is not a ranked retriever, and
                        scoring the raw trace just measures exploration noise).
                        Thresholded only for TK-only cases, where retrieval IS
                        a ranked doc list; report-only when CBS is involved,
                        because the SQL agent legitimately cross-checks
                        overlapping datasets and probes before pulling.
  answer_relevancy   -> response addresses the query (not thresholded on
                        refusal-type cases, where not answering is correct)
  rubric             -> per-case behavioural judgment driven by the case's
                        expected_behaviour (catches refusals, causal hedging)

Also applies the deterministic response contract to every response.
Writes eval_results.json with per-case and aggregate scores.

Report-only by default; pass --gate to exit non-zero on threshold breaches.
Pass --cases 2,3,6 to run a subset (1-based line numbers in eval_set.jsonl).

Requires: built chroma_db corpus + OPENROUTER_API_KEY. Costs LLM-judge API
calls and full graph runs — run on main / pre-release, not on every push.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make `src.` imports work when invoked as a file path (python src/eval/run_eval.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ragas.llms.base unconditionally imports ChatVertexAI from langchain_community,
# which was removed in langchain-community 0.4.x. Stub it out — never used here.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from dotenv import load_dotenv  # noqa: E402

from src.eval.contract import check_response_contract  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
RESULTS_FILE = Path("eval_results.json")
ARTIFACTS_FILE = Path("eval_artifacts.json")  # full captured contexts, for re-scoring

JUDGE_MODEL_DEFAULT = "deepseek-v4-pro"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

THRESHOLDS = {
    "faithfulness": 0.80,
    "context_precision": 0.60,
    "answer_relevancy": 0.70,  # not applied to refusal-type cases
    "rubric": 4.0,  # rubric scores are 1-5
}


def load_eval_set() -> list[dict]:
    cases = []
    with open(EVAL_SET) as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                case = json.loads(line)
                case["case_id"] = i
                cases.append(case)
    return cases


from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402


class ToolCaptureHandler(BaseCallbackHandler):
    """Records every LangChain tool invocation (name, input, output) during a run.

    Deterministic evidence capture: what the model actually pulled from DuckDB
    et al., straight from the tool layer, independent of what any LLM later
    said about it.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.calls.append(
            {
                "run_id": str(run_id),
                "name": (serialized or {}).get("name", "unknown_tool"),
                "input": str(input_str),
                "output": None,
            }
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        text = getattr(output, "content", output)
        for call in self.calls:
            if call["run_id"] == str(run_id) and call["output"] is None:
                call["output"] = str(text)
                break


# Tool calls that locate/prepare data rather than retrieve it. Their outputs
# (dataset IDs, dimension codes, measure labels) never support answer claims
# directly, so they are excluded from the context_precision evidence list.
_SETUP_TOOLS = {
    "search_cbs_catalog",
    "download_cbs_dataset",
    "get_dimensions",
    "get_dimension_values",
    "get_cbs_measure_labels",
}


def is_data_evidence(call: dict) -> bool:
    """True if a captured tool call retrieved actual data (vs setup/exploration).

    run_sql counts only when it queried an *_Observations table and returned
    rows — schema pokes (*_MeasureCodes, *_Dimensions, *Codes) and failed
    queries are exploration, not evidence. Unknown tools with real output are
    kept: better to over-include than silently drop evidence.
    """
    output = str(call.get("output") or "")
    if call["name"] in _SETUP_TOOLS:
        return False
    if not output.strip() or output.startswith(("SQL error", "Failed")):
        return False
    if call["name"] == "run_sql":
        sql = str(call["input"]).lower()
        return "observations" in sql and "information_schema" not in sql
    return True


def build_contexts(result: dict, tool_calls: list[dict]) -> list[str]:
    """Assemble the raw retrieved evidence for RAGAS.

    Deliberately excludes the intermediate agent prose (political_response,
    data_response): faithfulness must grade the final answer against what was
    actually retrieved, not against another LLM's summary of it.
    """
    contexts: list[str] = []

    # CBS side: every DuckDB/tool call the workers made, verbatim.
    for call in tool_calls:
        if call["output"] is None:
            continue
        contexts.append(f"[{call['name']}] input: {call['input']}\noutput: {call['output']}")

    # TK side: raw ranked documents from the political discover trace.
    for doc in (result.get("political_trace") or {}).get("odata_docs", []):
        parts = [f"[{doc['doc_id']}] {doc['datum']} — {doc['onderwerp']}"]
        if doc.get("champion"):
            parts.append(f"passage: {doc['champion']}")
        for party, snippets in (doc.get("party_excerpts") or {}).items():
            for snippet in snippets:
                parts.append(f"{party}: {snippet}")
        contexts.append("\n".join(parts))

    # Static corpus passages (empty while manifestos/CPB/PBL are disabled).
    for passage in result.get("political_passages", []):
        text = passage.get("text") if isinstance(passage, dict) else str(passage)
        if text:
            contexts.append(text)

    return contexts


def build_rubrics(expected_behaviour: str) -> dict[str, str]:
    return {
        "score1_description": (
            f"The response clearly violates the expected behaviour: {expected_behaviour}"
        ),
        "score3_description": (
            f"The response partially meets the expected behaviour ({expected_behaviour}) "
            "but misses at least one required element."
        ),
        "score5_description": (
            f"The response fully satisfies the expected behaviour: {expected_behaviour}"
        ),
    }


async def run_case(case: dict) -> dict:
    """Run the graph for one eval case, capturing retrieved evidence."""
    from src.graph import run_query

    sources = case.get("sources", ["tk", "cbs"])
    language = case.get("language", "en")
    handler = ToolCaptureHandler()

    result = await run_query(
        case["query"],
        language=language,
        mode="deep",
        include_manifestos=False,  # presentation mode: manifesto pipeline off
        include_tk="tk" in sources,
        include_cbs="cbs" in sources,
        extra_callbacks=[handler],
    )

    # Save the full conversation in the app's history schema so eval runs can
    # be curated into the public Examples page (copy data/history/<file> to
    # data/examples/) — the eval queries double as showcase queries.
    from src.storage import save_conversation

    conv_id = save_conversation(
        case["query"],
        result,
        settings={
            "language": language,
            "mode": "deep",
            "include_manifestos": False,
            "include_tk": "tk" in sources,
            "include_cbs": "cbs" in sources,
            "eval_case": case["case_id"],
        },
    )
    print(f"  conversation saved to data/history/ (id {conv_id}) — copy to data/examples/ to publish")

    response = result.get("final_response", "")
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "type": case.get("type", "standard"),
        "sources": sources,
        "expected_behaviour": case["expected_behaviour"],
        "response": response,
        "retrieved_contexts": build_contexts(result, handler.calls),
        # Data-bearing subset for context_precision: setup/exploration calls
        # stripped, TK docs and static passages kept (they are real evidence).
        "evidence_contexts": build_contexts(
            result, [c for c in handler.calls if is_data_evidence(c)]),
        "n_tool_calls": len(handler.calls),
        "contract_violations": check_response_contract(response),
    }


def _audited_faithfulness(judge):
    """Faithfulness subclass that keeps the judge's per-claim verdicts.

    Upstream ragas computes statement-level verdicts internally, uses them for
    the score, and discards them. Same pipeline, same prompts, same score —
    we just retain the audit trail in MetricResult.reason.
    """
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.result import MetricResult

    class AuditedFaithfulness(Faithfulness):
        async def ascore(self, user_input, response, retrieved_contexts):
            statements = await self._create_statements(user_input, response)
            if not statements:
                return MetricResult(value=float("nan"))
            verdicts = await self._create_verdicts(statements, "\n".join(retrieved_contexts))
            result = MetricResult(value=float(self._compute_score(verdicts)))
            result.reason = json.dumps(
                [
                    {"statement": s.statement, "verdict": bool(s.verdict), "reason": s.reason}
                    for s in verdicts.statements
                ],
                ensure_ascii=False,
            )
            return result

    return AuditedFaithfulness(llm=judge)


async def score_case(row: dict, judge, embeddings) -> dict:
    """Score one collected case with all four metrics. None = metric errored."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        InstanceSpecificRubrics,
    )

    q, resp, ctx = row["query"], row["response"], row["retrieved_contexts"]
    evidence = row.get("evidence_contexts", ctx)
    scores: dict[str, float | None] = {}
    row["faithfulness_verdicts"] = None

    async def _safe(name, coro):
        try:
            scores[name] = round(float((await coro).value), 4)
        except Exception as exc:
            print(f"  WARN {name} failed for case {row['case_id']}: {exc}")
            scores[name] = None

    if ctx:
        try:
            fr = await _audited_faithfulness(judge).ascore(
                user_input=q, response=resp, retrieved_contexts=ctx)
            scores["faithfulness"] = round(float(fr.value), 4)
            row["faithfulness_verdicts"] = json.loads(fr.reason) if fr.reason else None
        except Exception as exc:
            print(f"  WARN faithfulness failed for case {row['case_id']}: {exc}")
            scores["faithfulness"] = None
        if evidence:
            await _safe("context_precision", ContextPrecisionWithoutReference(llm=judge).ascore(
                user_input=q, response=resp, retrieved_contexts=evidence))
        else:
            scores["context_precision"] = None
            print(f"  NOTE case {row['case_id']}: no data-bearing evidence among "
                  f"{len(ctx)} contexts — context_precision skipped")
    else:
        # No evidence retrieved: nothing to grade grounding against. Contract +
        # rubric decide whether an empty-retrieval response was handled honestly.
        scores["faithfulness"] = None
        scores["context_precision"] = None
        print(f"  NOTE case {row['case_id']}: no retrieved contexts captured")

    await _safe("answer_relevancy", AnswerRelevancy(llm=judge, embeddings=embeddings).ascore(
        user_input=q, response=resp))
    await _safe("rubric", InstanceSpecificRubrics(llm=judge).ascore(
        rubrics=build_rubrics(row["expected_behaviour"]), user_input=q, response=resp))

    return scores


def check_thresholds(rows: list[dict]) -> list[str]:
    """Returns human-readable threshold breaches across all scored cases."""
    breaches = []
    for row in rows:
        for metric, threshold in THRESHOLDS.items():
            if metric == "answer_relevancy" and row["type"] in ("refusal", "absence"):
                continue  # correctly refusing / reporting absence legitimately does not answer
            if metric == "context_precision" and "cbs" in row["sources"]:
                # Report-only for CBS cases: the SQL agent explores overlapping
                # datasets and probes before pulling, which is healthy agent
                # behaviour but reads as noise to a ranked-retrieval metric.
                # TK-only cases keep the gate — BM25+triage IS a ranked retriever.
                continue
            value = row["scores"].get(metric)
            if value is not None and value < threshold:
                breaches.append(
                    f"case {row['case_id']} ({row['query'][:40]}...): "
                    f"{metric} {value} < {threshold}"
                )
        if row["contract_violations"]:
            breaches.append(
                f"case {row['case_id']}: contract violations {row['contract_violations']}"
            )
    return breaches


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for metric in THRESHOLDS:
        values = [
            r["scores"][metric]
            for r in rows
            if r["scores"].get(metric) is not None
            and not (metric == "answer_relevancy" and r["type"] in ("refusal", "absence"))
        ]
        out[metric] = round(sum(values) / len(values), 4) if values else None
    return out


async def main_async(case_ids: list[int] | None, gate: bool) -> int:
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    all_cases = load_eval_set()
    cases = [c for c in all_cases if c.get("enabled", True)]
    skipped = [c for c in all_cases if not c.get("enabled", True)]
    for c in skipped:
        print(f"SKIP case {c['case_id']} ({c['query'][:50]}): {c.get('disabled_reason', 'disabled')}")
    if case_ids:
        cases = [c for c in cases if c["case_id"] in case_ids]
    if not cases:
        print("No cases to run.")
        return 1

    judge_model = os.environ.get("POLDERCHECK_MODEL") or JUDGE_MODEL_DEFAULT
    judge_base = os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com"
    judge_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""

    judge_client = AsyncOpenAI(base_url=judge_base, api_key=judge_key)
    embed_client = AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    # Large output budget: faithfulness enumerates a verdict per claim, and with
    # 50+ captured tool contexts the default 2-3k cap truncates the judge's JSON.
    judge = llm_factory(judge_model, client=judge_client, max_tokens=16000)
    embeddings = OpenAIEmbeddings(client=embed_client, model=EMBED_MODEL)

    rows = []
    for case in cases:
        print(f"\n=== Case {case['case_id']}: {case['query']} (sources: {case.get('sources', ['tk', 'cbs'])})")
        row = await run_case(case)
        print(f"  graph done — {row['n_tool_calls']} tool calls, "
              f"{len(row['retrieved_contexts'])} context blocks "
              f"({len(row['evidence_contexts'])} data-bearing), "
              f"{len(row['response'].split())} words")
        row["scores"] = await score_case(row, judge, embeddings)
        print(f"  scores: {row['scores']}")
        rows.append(row)

    # Merge with any previously collected cases: runs accumulate (a re-run of
    # a case replaces its old record), so the eval can be built up one case at
    # a time and aggregates always reflect everything collected so far.
    new_records = [
        {k: row[k] for k in (
            "case_id", "query", "type", "sources", "scores",
            "contract_violations", "n_tool_calls", "response",
        )}
        | {
            "n_contexts": len(row["retrieved_contexts"]),
            "n_evidence_contexts": len(row["evidence_contexts"]),
        }
        for row in rows
    ]
    merged: dict[int, dict] = {}
    if RESULTS_FILE.exists():
        try:
            for record in json.loads(RESULTS_FILE.read_text()).get("cases", []):
                merged[record["case_id"]] = record
        except Exception as exc:
            print(f"WARN: could not merge previous {RESULTS_FILE}: {exc}")
    for record in new_records:
        merged[record["case_id"]] = record
    all_records = [merged[cid] for cid in sorted(merged)]

    breaches = check_thresholds(all_records)
    aggregates = aggregate(all_records)

    output = {
        "judge_model": judge_model,
        "n_cases": len(all_records),
        "aggregates": aggregates,
        "thresholds": THRESHOLDS,
        "threshold_breaches": breaches,
        "gated": gate,
        "cases": all_records,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    artifacts: dict = {}
    if ARTIFACTS_FILE.exists():
        try:
            artifacts = json.loads(ARTIFACTS_FILE.read_text())
        except Exception:
            pass
    for r in rows:
        artifacts[str(r["case_id"])] = {
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "evidence_contexts": r["evidence_contexts"],
            "faithfulness_verdicts": r.get("faithfulness_verdicts"),
        }
    ARTIFACTS_FILE.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False))

    print("\n=== Eval results ===")
    for metric, value in aggregates.items():
        print(f"  {metric}: {value if value is not None else 'n/a'} (threshold {THRESHOLDS[metric]})")
    if breaches:
        print(f"  breaches ({len(breaches)}):")
        for b in breaches:
            print(f"    {b}")
    print(f"  full per-case results: {RESULTS_FILE}")

    if gate and breaches:
        return 1
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="comma-separated 1-based case numbers, e.g. 2,3,6")
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero on threshold breaches (default: report only)")
    args = parser.parse_args()

    if not os.path.exists("./chroma_db"):
        print("No chroma_db found — build it first (scripts/rebuild_embeddings.py). Skipping eval.")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set. Skipping eval.")
        return 0

    case_ids = [int(x) for x in args.cases.split(",")] if args.cases else None
    return asyncio.run(main_async(case_ids, args.gate))


if __name__ == "__main__":
    sys.exit(main())
