#!/usr/bin/env bash
# One-time GitHub bootstrap: create the private repo, push, and file the
# feature backlog as issues. Requires: gh auth login (interactive) first.
set -euo pipefail

cd "$(dirname "$0")/.."

gh repo create poldercheck --private --source . --push

issue() {
  gh issue create --title "$1" --body "$2" >/dev/null
  echo "created: $1"
}

issue "Fetch manifesto corpus (Step 1A)" \
"Run src/ingest/fetch_manifestos.py. Needs MANIFESTO_API_KEY in .env (free signup: https://manifesto-project.wzb.eu/signup).

- [ ] Verify all party codes against the Manifesto Project codebook (a wrong code silently returns nothing)
- [ ] Consider adding BBB and NSC for the 2023 election
- [ ] Sanity-check quasi-sentence counts per party/election"

issue "Download CPB/PBL PDFs (Step 1B)" \
"Manual download into data/static/:
- CPB Charted Choices 2025-2028: https://www.cpb.nl/en/charted-choices-2025-2028 -> cpb_2025.pdf
- CPB Charted Choices 2027-2030: https://www.cpb.nl/en/publication/charted-choices-2027-2030 -> cpb_2027.pdf
- Most recent PBL climate analysis from https://www.pbl.nl -> pbl_climate.pdf"

issue "Build ChromaDB store + verify retrieval (Steps 2-3)" \
"Run src/ingest/chunk.py after the corpus exists, then src/ingest/retrieve.py as smoke test (housing query should return relevant manifesto chunks with sources). Depends on the two corpus issues."

issue "Install CBS MCP server (prebuilt binary - no Go needed)" \
"The repo ships prebuilt Linux binaries, so skip the build plan's go install:

    curl -sL https://github.com/dstotijn/mcp-cbs-cijfers-open-data/releases/download/v0.2.1/mcp-cbs-cijfers-open-data_Linux_x86_64.tar.gz | tar xz
    mv mcp-cbs-cijfers-open-data ~/.local/bin/

src/agents/data.py invokes it by name on PATH and degrades to an honest not-found if missing."

issue "First end-to-end query through the graph" \
"python -m src.graph (needs OPENROUTER_API_KEY in .env + built corpus). Verify: cited synthesis response, political passages present, CBS data or honest not-found. This is the PoC milestone."

issue "Wire OpenTK live parliamentary search into the graph (Step 11)" \
"run_political_analyst_v2 (live Tweede Kamer search via opentk-mcp over npx) exists in src/agents/political.py but the graph still calls v1. Swap it in, cap at 3 documents, use relevance triage before loading full content. npx server verified working (v1.0.17)."

issue "Filtered retrieval by policy category (cmp_code)" \
"Manifesto chunks carry cmp_code metadata. For topical queries (housing -> 501/502 etc.) filter before semantic search for much higher precision than unfiltered search. See note in build plan Step 2."

issue "Critic agent node for evaluative questions (off by default)" \
"src/prompts/critic.txt exists. Add a LangGraph node activated for evaluative queries ('has X kept its promises?') that produces 'case for / case against / question to consider' instead of a verdict. UI toggle in Streamlit."

issue "Docker build + Azure Container Apps deployment (Step 12)" \
"Dockerfile exists. Build, test locally, then deploy per build plan (resource group, ACR, Container Apps env, scale-to-zero). Output: public HTTPS URL for the CV."

issue "Verify Langfuse tracing end-to-end (Step 13)" \
"Tracing hook exists in src/graph.py (active when LANGFUSE_PUBLIC_KEY is set). Create free project at cloud.langfuse.com, run one query, confirm a trace with all three node spans, token usage, latency, and MCP tool calls. Check import path (v2 vs v3 handled in code)."

issue "Run RAGAS eval, tune thresholds, publish scores (Step 14)" \
"src/eval/run_eval.py + eval_set.jsonl exist; contract tests green. After corpus + keys: run full eval, tune faithfulness/context_precision thresholds against reality, put scores in README ('measured, not asserted'). Eval CI job runs on main only."

issue "Local parties: Kiesraad registry awareness" \
"Minimum honesty fix for the AP's local-party criticism: load Kiesraad registered-party lists per municipality so the system can say 'these local parties exist but are not yet in my corpus' instead of omitting them."

issue "Local parties: Open Raadsinformatie retrieval path (beta)" \
"Open State Foundation API aggregating municipal council documents (motions, votes, minutes) - the local analog of OpenTK. Add as third retrieval path or separate local-analyst node. Post-PoC. Municipality-level CBS data pairs with this (StatLine regional dimensions, existing MCP server)."

issue "README positioning: BeleidsRadar differentiation + AP standard" \
"Add a 'how this differs' section: BeleidsRadar searches 1M+ parliamentary documents; Poldercheck confronts claims with CBS statistics and makes CPB/PBL manifesto scorings queryable - that combination is unoccupied (verified June 2026). Frame honesty mechanisms against the AP's published failure list. Not a stemhulp; refuses voting advice."

issue "Parallelise political and data nodes in the graph" \
"Nodes run sequentially (political -> data -> synthesis). They are independent until synthesis; run them concurrently in LangGraph to roughly halve latency. Noted as v2 in the build plan."

echo
echo "Done. Repo + $(gh issue list --limit 100 | wc -l) issues created."
