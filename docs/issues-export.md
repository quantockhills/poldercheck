# GitHub Issues Export

## #22 [OPEN] Expand PBL corpus: housing, nitrogen, spatial planning reports
**Labels:** none  **Created:** 2026-06-15

Currently only one PBL report in the static corpus (climate 2023). PBL publishes extensively on housing affordability, construction targets, nitrogen policy, and spatial planning — all major political topics.

## What to add

- PBL housing affordability reports (woningmarkt)
- PBL nitrogen/stikstof analysis
- PBL spatial planning (ruimtelijke ordening)
- Older climate reports for longitudinal coverage

## How

Scrape pbl.nl/publicaties, filter by topic and year, download PDFs, run through `src/ingest/chunk.py`. No API available — PDF scraping only.

---

## #21 [OPEN] Expand CPB corpus: historical Charted Choices reports
**Labels:** none  **Created:** 2026-06-15

Currently two CPB Charted Choices reports (2025, 2027). CPB has published these since 2006, scoring every party programme on fiscal and economic impact each election cycle.

## What to add

- Charted Choices 2021, 2017, 2012 at minimum
- CPB election analysis reports (doorrekeningen)

## Why

Enables longitudinal comparisons: how has a party's fiscal position changed across election cycles? Much stronger than just the latest snapshot.

## How

Download PDFs from cpb.nl/publicaties, run through `src/ingest/chunk.py`.

---

## #20 [OPEN] Streamlit frontend: query UI and response layout
**Labels:** none  **Created:** 2026-06-15

## What to build

A clean web UI that's the only interface a user needs:

- Query input box (NL/EN, single field)
- Loading state while political + data nodes run
- Response layout: synthesis answer up top, expandable sections for political passages and CBS data below
- Party colour coding on political passages (VVD blue, PVV dark, SP red, GL green, etc.)
- Source citations rendered as links where possible
- Mobile-friendly layout

## Context

Streamlit is already the framework (`src/app.py` or similar). This issue is about making it actually presentable for the portfolio — not just a text dump. Deployment is tracked separately in #9.

---

## #19 [OPEN] Wire up Langfuse tracing and explore skill
**Labels:** near-term  **Created:** 2026-06-15

## What to do

1. Sign up at cloud.langfuse.com and create a project
2. Fill in `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`
3. Run a query and verify traces appear in Langfuse dashboard
4. Use the `/langfuse` Claude Code skill to query traces programmatically

## Context

Langfuse is already wired in — `_langfuse_callbacks()` in `src/graph.py` activates automatically when keys are set. Package installed (v4.7.1). Skill installed at `~/.claude/skills/langfuse/`.

Useful for debugging which node is slow / where output goes wrong.

---

## #18 [OPEN] Test political analyst end-to-end
**Labels:** near-term  **Created:** 2026-06-15

## What to test

Run a political query through the full graph and verify the political analyst:

- Returns results from **left-wing parties** (SP, GroenLinks, PvdA) — the embedding model switch to Qwen3-Embedding-8B was motivated by these not surfacing with MiniLM
- Returns results from right-wing parties (VVD, PVV) for contrast queries
- Citations match the retrieved manifesto passages (no hallucinated sources)
- Response is in the correct language (NL query → NL answer, EN query → EN answer)

## Suggested test queries

1. `woningmarkt betaalbaarheid huurprijzen` — housing affordability
2. `klimaatbeleid emissiedoelstellingen` — climate targets
3. `inkomensverdeling belastingen` — income distribution (should surface SP/PvdA)
4. `migratie asielbeleid` — migration (should surface PVV/VVD)

## Pass criteria

- All 4 queries return ≥1 passage from the expected party spectrum
- No passages from parties not represented in the manifesto corpus
- Relevance scores ≥ 0.5 for top result

---

## #17 [OPEN] Add data visualizations (charts/graphs in Streamlit)
**Labels:** none  **Created:** 2026-06-15

- [ ] Line charts for time-series responses (e.g. housing price index 2015-2025)
- [ ] Bar charts for cross-sectional comparisons  
- [ ] Embedded in Streamlit alongside the text response
- [ ] Library: Plotly or Altair (both work well with Streamlit)
- [ ] Save-to-PNG export for sharing/social media
- [ ] Visual stays honest: axes labeled, sources cited, no misleading y-axis zero cuts

---

## #16 [OPEN] stance_shift tool: quantitative emphasis analysis over coded manifestos
**Labels:** none  **Created:** 2026-06-12

From docs/use_cases.md section 1: 'how did party X's stance on topic Y change between elections' is the killer demo, and its quantitative half (share of manifesto devoted to a cmp_code per election) is better served by aggregating the coded CSV than by retrieval. Add a tool the political analyst can call: stance_shift(party, topic) -> emphasis percentages per election, combined with retrieval for the qualitative quotes. Numbers + quotes in one cited answer is the most differentiating output the tool can produce.

---

## #15 [OPEN] Parallelise political and data nodes in the graph
**Labels:** none  **Created:** 2026-06-12

Nodes run sequentially (political -> data -> synthesis). They are independent until synthesis; run them concurrently in LangGraph to roughly halve latency. Noted as v2 in the build plan.

---

## #14 [OPEN] README positioning: BeleidsRadar differentiation + AP standard
**Labels:** none  **Created:** 2026-06-12

Add a 'how this differs' section: BeleidsRadar searches 1M+ parliamentary documents; Poldercheck confronts claims with CBS statistics and makes CPB/PBL manifesto scorings queryable - that combination is unoccupied (verified June 2026). Frame honesty mechanisms against the AP's published failure list. Not a stemhulp; refuses voting advice.

---

## #13 [OPEN] Local parties: Open Raadsinformatie retrieval path (beta)
**Labels:** none  **Created:** 2026-06-12

Open State Foundation API aggregating municipal council documents (motions, votes, minutes) - the local analog of OpenTK. Add as third retrieval path or separate local-analyst node. Post-PoC. Municipality-level CBS data pairs with this (StatLine regional dimensions, existing MCP server).

---

## #12 [OPEN] Local parties: Kiesraad registry awareness
**Labels:** none  **Created:** 2026-06-12

Minimum honesty fix for the AP's local-party criticism: load Kiesraad registered-party lists per municipality so the system can say 'these local parties exist but are not yet in my corpus' instead of omitting them.

---

## #11 [OPEN] Run RAGAS eval, tune thresholds, publish scores (Step 14)
**Labels:** none  **Created:** 2026-06-12

src/eval/run_eval.py + eval_set.jsonl exist; contract tests green. After corpus + keys: run full eval, tune faithfulness/context_precision thresholds against reality, put scores in README ('measured, not asserted'). Eval CI job runs on main only.

---

## #10 [OPEN] Verify Langfuse tracing end-to-end (Step 13)
**Labels:** none  **Created:** 2026-06-12

Tracing hook exists in src/graph.py (active when LANGFUSE_PUBLIC_KEY is set). Create free project at cloud.langfuse.com, run one query, confirm a trace with all three node spans, token usage, latency, and MCP tool calls. Check import path (v2 vs v3 handled in code).

---

## #9 [OPEN] Docker build + Azure Container Apps deployment
**Labels:** none  **Created:** 2026-06-12

Dockerfile exists. Build, test locally, then deploy per build plan (resource group, ACR, Container Apps env, scale-to-zero). Output: public HTTPS URL for the CV.

---

## #8 [OPEN] Critic agent node for evaluative questions (off by default)
**Labels:** none  **Created:** 2026-06-12

src/prompts/critic.txt exists. Add a LangGraph node activated for evaluative queries ('has X kept its promises?') that produces 'case for / case against / question to consider' instead of a verdict. UI toggle in Streamlit.

---

## #7 [OPEN] Filtered retrieval by policy category (cmp_code)
**Labels:** none  **Created:** 2026-06-12

Manifesto chunks carry cmp_code metadata. For topical queries (housing -> 501/502 etc.) filter before semantic search for much higher precision than unfiltered search. See note in build plan Step 2.

---

## #6 [OPEN] Wire OpenTK live parliamentary search into the graph (Step 11)
**Labels:** none  **Created:** 2026-06-12

run_political_analyst_v2 (live Tweede Kamer search via opentk-mcp over npx) exists in src/agents/political.py but the graph still calls v1. Swap it in, cap at 3 documents, use relevance triage before loading full content. npx server verified working (v1.0.17).

---

## #5 [CLOSED] First end-to-end query through the graph
**Labels:** none  **Created:** 2026-06-12

python -m src.graph (needs OPENROUTER_API_KEY in .env + built corpus). Verify: cited synthesis response, political passages present, CBS data or honest not-found. This is the PoC milestone.

---

## #4 [CLOSED] Install CBS MCP server (prebuilt binary - no Go needed)
**Labels:** none  **Created:** 2026-06-12

The repo ships prebuilt Linux binaries, so skip the build plan's go install:

    curl -sL https://github.com/dstotijn/mcp-cbs-cijfers-open-data/releases/download/v0.2.1/mcp-cbs-cijfers-open-data_Linux_x86_64.tar.gz | tar xz
    mv mcp-cbs-cijfers-open-data ~/.local/bin/

src/agents/data.py invokes it by name on PATH and degrades to an honest not-found if missing.

---

## #3 [CLOSED] Build ChromaDB store + verify retrieval (Steps 2-3)
**Labels:** none  **Created:** 2026-06-12

Run src/ingest/chunk.py after the corpus exists, then src/ingest/retrieve.py as smoke test (housing query should return relevant manifesto chunks with sources). Depends on the two corpus issues.

---

## #2 [CLOSED] Download CPB/PBL PDFs (Step 1B)
**Labels:** none  **Created:** 2026-06-12

Manual download into data/static/:
- CPB Charted Choices 2025-2028: https://www.cpb.nl/en/charted-choices-2025-2028 -> cpb_2025.pdf
- CPB Charted Choices 2027-2030: https://www.cpb.nl/en/publication/charted-choices-2027-2030 -> cpb_2027.pdf
- Most recent PBL climate analysis from https://www.pbl.nl -> pbl_climate.pdf

---

## #1 [CLOSED] Fetch manifesto corpus (Step 1A)
**Labels:** none  **Created:** 2026-06-12

Run src/ingest/fetch_manifestos.py. Needs MANIFESTO_API_KEY in .env (free signup: https://manifesto-project.wzb.eu/signup).

- [ ] Verify all party codes against the Manifesto Project codebook (a wrong code silently returns nothing)
- [ ] Consider adding BBB and NSC for the 2023 election
- [ ] Sanity-check quasi-sentence counts per party/election

---

