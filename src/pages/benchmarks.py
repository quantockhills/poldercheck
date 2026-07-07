"""Public benchmarks page: how Poldercheck is tested, with per-case reports.

Reads eval_results.json and eval_artifacts.json from the repo root, both
written by src/eval/run_eval.py. The page renders whatever the latest eval
run recorded, so it never goes stale; if the files are absent it degrades
to the methodology text alone.
"""

import json
from pathlib import Path

import streamlit as st

from src.ui import inject_frosted_main, inject_page_css

st.set_page_config(page_title="Benchmarks — Poldercheck", page_icon="🌊", layout="wide")

inject_page_css()
inject_frosted_main()

_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(name: str):
    try:
        return json.loads((_ROOT / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


results = _load_json("eval_results.json")
artifacts = _load_json("eval_artifacts.json") or {}

st.markdown(
    "<div style='text-align:center;margin-bottom:0.5rem'>"
    "<span class='pc-box'><h1 style='margin:0;padding:0'>BENCHMARKS</h1></span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "An AI tool that answers political questions has one obligation above all others: "
    "do not make things up. Poldercheck's design is built around that obligation, and we "
    "test whether it is met rather than assume it. This page explains how the testing "
    "works and lets you read the raw reports, including every individual judgement, for "
    "the same searches you can browse on the [Examples page](/examples)."
)

st.markdown("### The examiner")
st.markdown(
    "Every test question runs through the full pipeline, exactly as a real search would. "
    "The answer is then handed to a separate language model that acts as an examiner. The "
    "examiner receives three things: the question, the answer, and the raw evidence the "
    "pipeline retrieved, meaning the actual debate excerpts and the actual statistical "
    "query results. It is not told what we hoped the answer would say.\n\n"
    "Its core task is called faithfulness checking. The examiner splits the answer into "
    "individual claims and asks one blunt question about each: does the retrieved evidence "
    "support this sentence? A claim can be well written, plausible, and even true, and "
    "still fail if no retrieved excerpt backs it. That strictness is deliberate. It is the "
    "property that separates a research tool from a confident storyteller.\n\n"
    "The examiner writes down its reasoning for every verdict, and we keep all of it. The "
    "reports at the bottom of this page show those verdicts unedited, including the failures."
)

st.markdown("### What we measure")
st.markdown(
    "| Measure | The question it answers | Our bar |\n"
    "|---|---|---|\n"
    "| Faithfulness | What share of the answer's claims does the retrieved evidence support? | at least 0.80 |\n"
    "| Context precision | How much of what was retrieved was actually useful for this question? | at least 0.60 |\n"
    "| Answer relevancy | Does the answer address what was asked? | at least 0.70 |\n"
    "| Behaviour rubric | Did the answer do what this specific question demands? Graded 1 to 5. | at least 4.0 |\n"
    "| Response contract | Are citations present, and is the answer within its word budget? Checked by plain code, with no model involved. | must pass |\n"
)
st.markdown(
    "Two refinements keep these numbers honest. For statistical questions, context "
    "precision is reported but not enforced, because the data agent deliberately "
    "cross-checks overlapping datasets and a precision metric reads that diligence as "
    "noise. And for trap questions, answer relevancy is reported but not enforced, "
    "because refusing an improper question is the correct behaviour and scores low on "
    "relevancy by design."
)

st.markdown("### Some questions are traps on purpose")
st.markdown(
    "The test set does not only contain questions the system should answer well. It also "
    "contains questions the system should decline or admit ignorance about: a request for "
    "a voting recommendation, a causal claim the data cannot support, a question about "
    "local politics outside the corpus. The behaviour rubric grades each of these against "
    "what the right response looks like, and the reports below show how each one went."
)

# Worked example: one supported and one unsupported claim from a real report.
_example = artifacts.get("3") or {}
_verdicts = _example.get("faithfulness_verdicts", [])
_passed = next((v for v in _verdicts if v.get("verdict")), None)
_failed = next((v for v in _verdicts if not v.get("verdict")), None)
_example_case = None
if results:
    _example_case = next((c for c in results.get("cases", []) if c.get("case_id") == 3), None)

if _passed and _failed and _example_case:
    st.markdown("### A worked example")
    st.markdown(
        "Here is the examiner grading a real answer to a real test question, taken "
        "directly from the latest report. First a claim that passed, then one that failed."
    )
    st.markdown(f"**The question:** {_example_case['query']}")
    st.success(
        f"**Claim from the answer:** {_passed['statement']}\n\n"
        f"**The examiner's verdict:** supported.\n\n"
        f"**Its reasoning:** {_passed['reason']}"
    )
    st.error(
        f"**Claim from the answer:** {_failed['statement']}\n\n"
        f"**The examiner's verdict:** not supported.\n\n"
        f"**Its reasoning:** {_failed['reason']}"
    )
    st.markdown(
        "Failures like this are the point of the exercise. In an earlier run of this same "
        "question, the answer credited a parliamentary motion to the wrong party, while "
        "the transcript named two members of other parties as its proposers. The examiner "
        "failed that claim, we traced the cause to how multi-speaker debate excerpts were "
        "labelled, and the fix is tracked openly in the "
        "[project's issue tracker](https://github.com/quantockhills/poldercheck/issues/58). "
        "The score alone would not have told us any of that; the written verdicts did."
    )

st.markdown("### The reports")

if not results:
    st.info(
        "No evaluation reports are bundled with this deployment yet. The methodology "
        "above still describes exactly how the suite runs; the harness lives in "
        "[src/eval](https://github.com/quantockhills/poldercheck/tree/main/src/eval)."
    )
else:
    agg = results.get("aggregates", {})
    st.markdown(
        f"One report per test question, {results.get('n_cases', 0)} in the latest run. "
        f"Averages across all of them: faithfulness {agg.get('faithfulness', 0):.2f}, "
        f"context precision {agg.get('context_precision', 0):.2f}, answer relevancy "
        f"{agg.get('answer_relevancy', 0):.2f}, behaviour rubric {agg.get('rubric', 0):.1f} "
        f"out of 5. Judge model: `{results.get('judge_model', 'unknown')}`. Scores come "
        "from the exact files the test harness writes; nothing is edited for this page."
    )

    thresholds = results.get("thresholds", {})
    for case in results.get("cases", []):
        cid = case.get("case_id")
        ctype = case.get("type", "standard")
        sources = case.get("sources", [])
        scores = case.get("scores", {})
        header = f"Case {cid} · {case.get('query', '')[:70]}"

        with st.expander(header):
            st.markdown(f"**{case.get('query', '')}**")
            st.caption(
                f"Question type: {ctype} · sources: {', '.join(sources) or 'none'} · "
                f"graded against {case.get('n_evidence_contexts', '?')} evidence excerpts"
            )

            rows = ["| Measure | Score | Bar | Result |", "|---|---|---|---|"]
            for metric, label in [
                ("faithfulness", "Faithfulness"),
                ("context_precision", "Context precision"),
                ("answer_relevancy", "Answer relevancy"),
                ("rubric", "Behaviour rubric"),
            ]:
                score = scores.get(metric)
                bar = thresholds.get(metric)
                waived = (metric == "context_precision" and "cbs" in sources) or (
                    metric == "answer_relevancy" and ctype in ("refusal", "absence")
                )
                if score is None:
                    rows.append(f"| {label} | n/a | {bar} | not measurable |")
                elif waived:
                    rows.append(f"| {label} | {score:.2f} | {bar} | report-only |")
                else:
                    verdict = "pass" if score >= bar else "**below bar**"
                    rows.append(f"| {label} | {score:.2f} | {bar} | {verdict} |")
            violations = case.get("contract_violations", [])
            contract = "pass" if not violations else "**failed:** " + "; ".join(violations)
            rows.append(f"| Response contract | | | {contract} |")
            st.markdown("\n".join(rows))

            verdicts = (artifacts.get(str(cid)) or {}).get("faithfulness_verdicts", [])
            if verdicts:
                st.markdown("**Every claim, as the examiner judged it:**")
                st.dataframe(
                    [
                        {
                            "Supported": "yes" if v.get("verdict") else "NO",
                            "Claim from the answer": v.get("statement", ""),
                            "The examiner's reasoning": v.get("reason", ""),
                        }
                        for v in verdicts
                    ],
                    use_container_width=True,
                )

st.markdown(
    "You can [browse the example searches](/examples) these reports grade, "
    "[run your own question](/), or [read more about the project](/about)."
)
