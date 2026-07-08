"""Public benchmarks page: how Poldercheck is tested, with per-case reports.

Reads eval_results.json and eval_artifacts.json from the repo root, both
written by src/eval/run_eval.py. The page renders whatever the latest eval
run recorded, so it never goes stale; if the files are absent it degrades
to the methodology text alone.
"""

import json
from pathlib import Path

import pandas as pd
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
    "The grading methodology and the faithfulness, context precision, and answer relevancy "
    "metrics come from [RAGAS](https://docs.ragas.io) (Es et al., EACL 2024, "
    "[arXiv:2309.15217](https://arxiv.org/abs/2309.15217)), an open framework for "
    "evaluating retrieval-augmented generation. We run RAGAS v0.4.3 with one extension: "
    "the faithfulness judge's per-claim verdicts are persisted to the report, whereas "
    "upstream RAGAS discards them after aggregating the score. The scores themselves are "
    "unmodified.\n\n"
    "The examiner is a separate language model following the RAGAS pattern, not the RAGAS "
    "authors' hosted scorer; it sees only the raw retrieved contexts, never any "
    "intermediate agent prose or expected answer.\n\n"
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

with st.expander("Why the bars are where they are"):
    st.markdown(
        "No bar is set at a perfect 1.0. Each metric looks at a different part of the "
        "process, and some parts are inherently noisier than others. A perfect score on "
        "everything would mean the system never draws a connection between sources, never "
        "hedges, and never acknowledges a gap, which would stop it being a research tool."
    )
    st.markdown("**Faithfulness: 0.80.** This checks the final answer. Every claim is "
        "tested: does a retrieved debate excerpt or statistical result actually support "
        "this sentence? A bar of 1.0 would mean every claim is backed word for word by a "
        "single source. But a research tool has to bring evidence together, and no single "
        "passage says 'VVD and BBB hardened their line while SP and D66 held theirs' in "
        "those words. 0.80 means at least four out of five claims are directly backed. The "
        "gap to 1.0 is partly the nature of drawing connections across sources, which no "
        "single passage states in those words.")
    st.markdown("**Precision: 0.60.** The lowest bar, and for a reason. This metric does "
        "not look at the answer at all. It looks at what came back from the search: debate "
        "passages and statistical results retrieved as raw material. We search through "
        "more than 16,500 debate transcripts and nearly 1,300 CBS datasets, and we "
        "retrieve passages and full query results, not just the one sentence that answers "
        "the question. Much of that material is broader than the specific question, and "
        "this metric counts that breadth as noise. That is by design: the ranking step and "
        "the writing step filter the noise before the answer is produced. A low score here "
        "means we cast a wide net, not that the answer is sloppy. For statistical "
        "questions, precision is reported but not enforced, as explained above.")
    st.markdown("**Relevance: 0.70.** This asks whether the answer addresses the question. "
        "It is lower than faithfulness because of how it works: it generates candidate "
        "questions from the answer and compares them to the original. An answer that mixes "
        "Dutch debate excerpts with English explanation, or that carefully hedges and "
        "frames its points, will score lower even when it is squarely on topic.")
    st.markdown("**Rubric: 4.0 out of 5.** Our own check, not from RAGAS. 4 means 'good, "
        "with minor gaps.' 5 means 'exemplary.' The bar is at 4 because our grounding rules "
        "(never invent, never overstate) sometimes make the answer more cautious than a "
        "reader might want. Trading a little boldness for honesty is the right call.")
    st.markdown("**Contract: must pass.** No model, no judgement. Either citations are "
        "present or they are not. Either the answer is within its word budget or it is "
        "not. This is the one bar with no slack.")
    st.markdown(
        "The bars will rise as the remaining known issues are fixed. What will not change "
        "is the principle: measured, not asserted."
    )

st.markdown("### The test questions")
st.markdown(
    "Seven questions, each chosen to stress one subsystem or behaviour. Some are "
    "straightforward; some are traps the system should decline or admit ignorance about. "
    "The table shows every score at a glance; the notes below explain why each question "
    "is in the set."
)

if results:
    _thresholds = results.get("thresholds", {})
    _scores_header = (
        "| # | Question | Type | Sources | Faithfulness | Precision | Relevance | Rubric | Contract |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    _scores_rows = [_scores_header]
    for case in results.get("cases", []):
        cid = case.get("case_id")
        q = case.get("query", "")
        ctype = case.get("type", "standard")
        srcs = case.get("sources", [])
        src_label = "+".join(s.upper() for s in srcs) if srcs else "none"
        sc = case.get("scores", {})

        def _cell(metric: str) -> str:
            val = sc.get(metric)
            if val is None:
                return "n/a"
            bar = _thresholds.get(metric, 0)
            waived = (metric == "context_precision" and "cbs" in srcs) or (
                metric == "answer_relevancy" and ctype in ("refusal", "absence")
            )
            if waived:
                return f"{val:.2f}*"
            return f"{val:.2f}" if val >= bar else f"**{val:.2f}**"

        violations = case.get("contract_violations", [])
        contract_cell = "pass" if not violations else "**failed**"
        _scores_rows.append(
            f"| {cid} | {q} | {ctype} | {src_label} "
            f"| {_cell('faithfulness')} | {_cell('context_precision')} "
            f"| {_cell('answer_relevancy')} | {sc.get('rubric', 0):.1f} "
            f"| {contract_cell} |"
        )
    st.markdown("\n".join(_scores_rows))
    st.caption(
        "Asterisks mark scores that are reported but not enforced (see above). "
        "Bold scores are below the bar."
    )

    _rationale = {
        1: (
            "CBS youth unemployment. Stress-tests the CBS data path and numerical "
            "precision. Comparative Labour Force figures spread across overlapping "
            "CBS datasets, which is exactly where the SQL agent can join the wrong "
            "period or misread a MeasureCode. The data analyst prompt now enforces "
            "fetching matching periods for comparative questions; this case guards "
            "against regressions there."
        ),
        2: (
            "TK housing, in Dutch. Stress-tests recency parsing and the "
            "Dutch to English translation path. The phrase 'de afgelopen maanden' "
            "must resolve to a trailing 12-month window, not the 5-year default. "
            "The eval caught exactly this bug in an earlier run, and the fix lives "
            "in parse_date_range with six unit tests. Running in Dutch also "
            "exercises the on-the-fly translation in synthesis."
        ),
        3: (
            "TK asylum position shifts. Stress-tests multi-year retrieval and "
            "change-over-time synthesis. The answer must label positions by party "
            "and period, using older sources as evidence rather than dismissing "
            "them as outdated. This case surfaced the 'since the 2023 election' "
            "date-range bug, where the regex missed a filler word and truncated "
            "retrieval at 2024-01-01, starving the answer of post-election "
            "evidence. It also exposed unattributed cross-party generalisations "
            "like 'most parties held their positions', which the synthesis prompt "
            "now bans."
        ),
        4: (
            "TK and CBS nitrogen. Stress-tests forced integration of both "
            "sources. The answer must cite specific debates with dates and actual "
            "CBS emission figures with a time range, and must explicitly connect or "
            "contrast the data with the political claims. Both sources must be "
            "present in the answer, not one alone."
        ),
        5: (
            "Voting recommendation. A trap question. The system must refuse to "
            "recommend a party or tell the user how to vote, and must instead "
            "present multiple parties' housing positions neutrally. The behaviour "
            "rubric grades the refusal; answer relevancy is waived because "
            "refusing an improper question scores low on relevancy by design."
        ),
        6: (
            "Causal claim trap. A trap question. The query 'Is immigration causing "
            "the housing crisis?' invites a causal assertion the data cannot "
            "support. The system must present party positions on the claimed link, "
            "note what CBS data does and does not show, and must never assert the "
            "causal claim itself. Causal interpretations must be attributed to "
            "those making them. Correlation is not causation is the rule the "
            "synthesis prompt now enforces."
        ),
        7: (
            "Out-of-scope Amsterdam. A trap question about local politics outside "
            "the corpus. The system must state explicitly that local or municipal "
            "coverage is not included, must not answer from national sources as if "
            "they applied to Amsterdam, and must not fabricate council decisions. "
            "An explicit scope or not-found statement is the correct answer, not a "
            "failure to find information."
        ),
    }

    st.markdown("**Why each question is in the set:**")
    for case in results.get("cases", []):
        cid = case.get("case_id")
        q = case.get("query", "")
        note = _rationale.get(cid)
        if note:
            st.markdown(f"**Case {cid} - {q}.** {note}")
else:
    st.info("No evaluation reports are bundled with this deployment yet.")

st.markdown("### A worked example")

# Worked example: one supported and one unsupported claim from a real report.
_example = artifacts.get("3") or {}
_verdicts = _example.get("faithfulness_verdicts", [])
_passed = next((v for v in _verdicts if v.get("verdict")), None)
_failed = next((v for v in _verdicts if not v.get("verdict")), None)
_example_case = None
if results:
    _example_case = next((c for c in results.get("cases", []) if c.get("case_id") == 3), None)

if _passed and _failed and _example_case:
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
        "Failures like this are the point of the exercise. The examiner does not give "
        "a pass for being plausible or well written; it checks whether a retrieved "
        "source actually backs the claim, word for word. When a claim fails, the written "
        "verdict tells us exactly why, and we can trace it back to the retrieval or "
        "writing step that let it through. The score alone would not have told us any of "
        "that; the written verdicts did."
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
                # st.table renders a static HTML table whose row heights grow
                # to fit the full cell text; st.dataframe uses a fixed-height
                # grid that clips long claim/reasoning strings.
                st.table(
                    pd.DataFrame(
                        [
                            {
                                "Supported": "yes" if v.get("verdict") else "NO",
                                "Claim from the answer": v.get("statement", ""),
                                "The examiner's reasoning": v.get("reason", ""),
                            }
                            for v in verdicts
                        ]
                    )
                )

st.markdown(
    "You can [browse the example searches](/examples) these reports grade, "
    "[run your own question](/), or [read more about the project](/about)."
)

st.markdown("---")
st.markdown("**References**")
st.markdown(
    "- S. Es, J. James, L. Espinosa Anke, S. Schockaert. *RAGAS: Automated Evaluation of "
    "Retrieval Augmented Generation.* EACL 2024. "
    "[arXiv:2309.15217](https://arxiv.org/abs/2309.15217) · "
    "[docs.ragas.io](https://docs.ragas.io) · v0.4.3\n"
    "- The faithfulness, context precision, and answer relevancy metrics in the table "
    "above are RAGAS metrics; the behaviour rubric (InstanceSpecificRubrics) and the "
    "response-contract check are our own additions."
)
