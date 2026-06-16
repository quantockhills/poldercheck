# Nosyco + Poldercheck as a Dutch AI-job portfolio — assessment & plan

## Context

Madhav is applying for AI roles in the Netherlands in 2026, casting a wide net across both common profiles: "AI/GenAI engineer" (RAG, LangChain, agents, production LLM apps) and ML/LLM engineer (fine-tuning, evals, alignment). Two portfolio projects are planned: **nosyco** (`/home/madhav22m/gitrepos/nosyco/nosyco-plan/`) and **poldercheck** (docs at `/mnt/c/Users/Tamura/Desktop/Temporary/poldercheck_readme.md` and `poldercheck_build_plan.md`). He also recently shipped **spidercrab** (C++ REAPER extension + web frontend) in ~10 days, so plans assume fast iteration. Question: is nosyco + poldercheck a good mix for the Dutch market?

## Verdict: yes — it's close to the ideal two-project pairing

The two projects split the 2026 market's two skill clusters between them with almost no overlap and no gaps:

| 2026 NL market signal | Nosyco | Poldercheck |
|---|---|---|
| LLM fine-tuning (QLoRA, SFT, GRPO, DPO) — hottest specialized skill | ✓ core | — |
| Eval literacy — "biggest signal of actually building with LLMs" | ✓ core (BullshitBench, OR-Bench, Pareto frontier, reward design) | partial (benchmark queries) |
| RAG / vector DBs — most common production pattern | — | ✓ core (ChromaDB, Qwen3-Embedding-8B via OpenRouter, multilingual) |
| LangChain / LangGraph / agent orchestration | — | ✓ core (two-agent LangGraph + critic node) |
| MCP integration — newly listed in 2026 skill lists | — | ✓ core (two MCP servers, Node + Go) |
| Cloud deployment / Docker / CI | — | ✓ (Azure Container Apps, GitHub Actions) |
| Safety / guardrails / honesty | ✓ (pushback training) | ✓ (groundedness, refusal-to-adjudicate design) |

**Three extra strengths of the combination:**

1. **Coherent narrative, not two random demos.** Both projects attack the same problem — AI confidently producing nonsense — from the two opposite ends: nosyco *trains the model* to push back; poldercheck *constrains the system* with retrieval-first grounding. Poldercheck's README already cites BullshitBench, the exact benchmark nosyco fine-tunes against. CV/interview line: "I work on making LLMs epistemically honest — at the model level (fine-tuning) and at the system level (grounded RAG)."
2. **Poldercheck is NL-market catnip.** Tweede Kamer, CBS, CPB/PBL, bilingual NL/EN corpus — a Dutch hiring manager immediately sees local commitment, Dutch-language capability (backed by NT2-II), and domain knowledge. No imported tutorial project does this.
3. **With spidercrab as third piece**, the portfolio also shows non-AI engineering depth (C++, real-time protocols, web UI) — answers the "3+ yrs production software" line in JDs.

**Risks to manage:**

- **Both are currently unfinished.** Nosyco: Phase 0 data ~80%, no training run, `src/methods`/`reward`/`eval` empty. Poldercheck: PoC working end-to-end (CBS data node 31s, manifesto retrieval fixed, housing query returns cited CBS numbers); remaining: Azure deployment (#9) and RAGAS eval (#11). Two half-finished repos are weaker than one finished one — sequencing matters (below).
- **Poldercheck lacks a real eval.** It makes honesty claims (groundedness, citation faithfulness, refusal behavior) but the build plan only has 7 manual benchmark queries. Adding a small automated eval (citation-faithfulness / groundedness scoring, LLM-as-judge over the benchmark queries in CI) turns its biggest claim into a measured result and hits the eval-literacy signal twice.
- Small bug to fix when building poldercheck: `DUTCH_PARTIES` dict in the build plan lists key `22110: "VVD"` twice (silently dedupes; one party intended is lost).

## Plan (sequenced, not time-boxed — done when done)

### 1. Finish poldercheck PoC first
It's the faster path to a public, clickable artifact and covers the keywords in the *most numerous* NL vacancies (RAG, LangGraph, MCP, Azure). Follow `poldercheck_build_plan.md` steps 0–12; fix the duplicate party-code bug in Step 1.
**Add beyond the plan:** automated groundedness/citation eval over the benchmark queries, run in CI — this is the differentiator over every other RAG portfolio project.

### 2. Finish nosyco through at least RAIT + SFT→GRPO
(Wall-clock here is dominated by GPU runs, not coding effort.)
- Complete Phase 0 per `handover-2026-06-10.md`: fix 400 BCT examples + 200 negative controls; add legal (LegalHalBench) and finance (FinDVer) domains (currently 0 each); hand-author ~150–200 physics examples
- Implement `src/eval/` **first** (BullshitBench v2 + OR-Bench subset + MMLU delta via vllm), then RAIT baseline on Kaggle free T4, then SFT→GRPO on Vast.ai within the €50–100 budget; RFT+mixing only if time permits
- Log to W&B; produce the Pareto frontier plot

### 3. Publish both
- Public GitHub repos; READMEs leading with results (nosyco: results table + Pareto plot; poldercheck: live demo URL + eval scores)
- Nosyco: model + dataset on HuggingFace, Gradio demo on HF Spaces (base vs fine-tuned side-by-side), submit to BullshitBench leaderboard
- Poldercheck: public Azure URL (scale-to-zero keeps cost near nil)

### 4. Optional synergy (strong interview material, do only if time)
Swap nosyco's fine-tuned pushback model in as poldercheck's critic/synthesis agent (poldercheck is already bring-your-own-model via OpenAI-spec `base_url`; vllm serves that). One sentence in interviews: "my fine-tuned model runs inside my RAG system."

### 5. CV framing
- nosyco → "fine-tuned & aligned LLMs (QLoRA, SFT, GRPO), designed eval suites and reward functions, published model + dataset + leaderboard entry"
- poldercheck → "production RAG: LangGraph multi-agent orchestration, two MCP integrations (CBS StatLine + Tweede Kamer), Qwen3 multilingual embeddings via OpenRouter, Langfuse observability, Dockerized Azure deployment with CI and automated evals"
- spidercrab → production software engineering (C++, real-time protocols, web UI)

## Verification
- Poldercheck: the 7 benchmark queries in `poldercheck_build_plan.md` pass their stated fail-conditions; eval scores in CI; public URL loads and answers a housing-affordability query with citations
- Nosyco: eval numbers per checkpoint in W&B; end state = meaningful BullshitBench Green-rate lift over Qwen3-4B baseline, <5pt MMLU drop, acceptable OR-Bench over-refusal; public HF + leaderboard links
- Portfolio test: every CV bullet has a clickable artifact behind it

## Evidence from concrete NL postings (May–June 2026)

Verified against actual vacancy texts, not summary articles:

| Posting | Date | Named requirements |
|---|---|---|
| [Team Rockstars IT — GenAI Engineer](https://www.teamrockstars.nl/werken-bij/vacatures/genai-engineer/) (€75–130k) | live Jun 2026 | 4+ yrs eng, 1+ yr GenAI; RAG (retrieval quality, chunking, hybrid search, cost/perf); LangChain+LangGraph, CrewAI, PydanticAI; LlamaIndex; FAISS/pgvector/Qdrant/Weaviate; **evals: RAGAS, Giskard, TruLens, DeepEval, A/B tests**; **observability: Langfuse, MLflow**; vLLM/TGI/Ollama; Docker/K8s/Terraform/GH Actions; Azure/AWS/GCP/Databricks |
| [Enexis — Senior GenAI Engineer](https://werkenbij.enexis.nl/vacatures/senior-genai-engineer-14997) | live Jun 2026 | 5+ yrs Python production services/APIs; deep LLM/RAG/embeddings/vector DBs (Azure AI Search, Pinecone, Weaviate, Milvus); LangChain/LlamaIndex/Semantic Kernel; **agentic workflows, MCP patterns**; LLMOps (token usage, latency, drift); embedding fine-tuning; Azure OpenAI/AI Studio/Container Apps, Terraform |
| [Rabobank — Senior Full Stack AI Engineer (GenAI & Agents)](https://www.banken.nl/vacatures/45033/rabobank/senior-full-stack-ai-engineer-generative-ai-agents) | recently closed | 5+ yrs; production AI workloads; **LangGraph, AutoGen, CrewAI**; vector DBs + RAG patterns; prompt chaining/memory; Azure OpenAI/ML/AKS/Functions; Terraform/Bicep, Docker/K8s |
| [ABN AMRO — Medior AI Engineer](https://magnet.me/nl-NL/vacature/1027082/medior-ai-engineer) (€5.1–7.3k/mo) | 2026-05-03 | "Built and deployed AI applications using **LLMs, RAG, agents, fine-tuning, voice AI** or equivalent"; Python in production; Azure/Databricks; **"ship something people actually used", side projects valued; fluent Dutch + English** |
| [ABN AMRO — ML & AI Engineer](https://www.datacarriere.com/vacature/machine-learning-ai-engineer-amsterdam-abn-amro-dep4jw2isg6tauyz) | 2026-05-05 | 8+ yrs; **classic ML**: NumPy/pandas/scikit-learn/PyTorch/TF, PySpark, SQL, Databricks; MLOps; bias/fairness/explainability |
| [Q42 — AI Engineer](https://werkenbij.q42.nl/ai-engineer) | live | Pragmatic LLM/AI-as-a-service integration, full-stack, cloud deploy, model validation & monitoring; **portfolio of completed work as nice-to-have** |

**What this confirms / changes:**
- RAG + agents + LangChain/LangGraph + vector DBs: in every GenAI posting → poldercheck is squarely on target.
- **Azure dominates NL** (all four enterprise postings) → poldercheck's Azure Container Apps deployment is the right call.
- **Eval/observability tools are named explicitly** (RAGAS, DeepEval, TruLens, Langfuse, MLflow) → build poldercheck's groundedness eval with RAGAS or DeepEval and add Langfuse tracing, so the CV matches JD keywords letter-for-letter.
- **MCP is now in actual JDs** (Enexis "MCP patterns") → poldercheck's two MCP servers are a concrete differentiator.
- Full-model fine-tuning is a differentiator, not a baseline requirement (ABN lists it among equivalents; Enexis wants embedding fine-tuning) → nosyco positions above the bar rather than meeting it; vLLM serving experience from nosyco is itself a named JD skill.
- Bank "ML engineer" titles still want classic ML (scikit-learn, PySpark, Databricks) — a known portfolio gap; PhD background partially covers, not worth a dedicated project.
- ABN explicitly values shipped side projects and requires fluent Dutch — both favorable (NT2-II).

## CBS OData v4 API — Documentation Review

**Why this section exists:** We rebuilt `build_cbs_catalog.py` and `retrieve.py` for the v4 API before fully reading the docs. This section records what the docs actually say, validates our implementation against them, and lists the small gaps worth fixing.

### What the v4 API actually provides

**Base URL:** `https://datasets.cbs.nl/odata/v1/CBS/`

**Catalog endpoint:** `/CBS/Datasets` — returns dataset metadata with these fields:
- `Identifier` — alphanumeric table ID (e.g., `83163NED`); **always NED in v4**, no ENG variants
- `Title` — Dutch dataset name
- `Description` — free-text Dutch description; contains "Beschikbaar vanaf: YYYY"
- `Modified` — ISO-8601 datetime, last catalog update
- `Status` — one of `"Regulier"`, `"Gediscontinueerd"` (at minimum; other values possible but not documented)
- `Distributions` — array of `{Format, DownloadUrl}`; `Format == "odata"` gives the data URL
- `Language` — always `"nl"` in the active 1,297 datasets

**Per-dataset data endpoints:**
- `/{id}/Observations` — long-format data; each row is `{Id, Measure, Value, ValueAttribute, {DimName}...}`
- `/{id}/Dimensions` — dimension metadata
- `/{id}/{DimName}Codes` — codelist for that dimension (e.g., `PeriodenCodes`, `WijkenEnBuurtenCodes`)
- `/{id}/Properties` — table-level singleton metadata

**OData query parameters:** `$filter`, `$select`, `$top`, `$skip`, `$count`, `$expand` all standard.

**Pagination:** max 100,000 cells per response; CBS recommends ≤10,000 rows per call. Truncated responses include `@odata.nextLink` with the next-page URL pre-formed.

**Rate limits:** None documented; no auth required.

**ValueAttribute values** (observation quality flags): `"None"` (valid), `"Zero"`, and descriptive missing-data reasons. Provisional data is flagged `"Voorlopig"` / `"Nader voorlopig"` in description, not in a dedicated field.

### Validation: what our code gets right

| Assumption in code | Status |
|---|---|
| `$filter=Status ne 'Gediscontinueerd'` selects active datasets | **Correct** — 1,297 returned, all `"Regulier"` |
| Identifier = NED suffix only in v4 | **Correct** — ENG identifiers are absent; our catalog has zero ENG entries |
| `/{id}/{DimName}Codes` endpoint pattern | **Correct** per docs — MCP patch to use this (not `DimensionValues?$filter=...`) was right |
| `Distributions[].Format == "odata"` → `DownloadUrl` | **Correct** field names per API |
| `Modified[:10]` gives `YYYY-MM-DD` | **Correct** — CBS returns ISO-8601 datetime |
| `@odata.value` response envelope | **Correct** — OData v4 standard |
| 60s agent timeout ≥ MCP 30s HTTP timeout | **Correct** — covers MCP internal timeout |
| Recency filter `_end_year(period) >= 2015` using `modified` date | **Works** — all 1,297 active datasets have `modified` in 2019–2025 range so they all pass |

### Two small gaps worth fixing

**Gap 1 — Pagination robustness** (`build_cbs_catalog.py`): **FIXED** — now follows `@odata.nextLink` instead of manually incrementing `$skip`.

**Gap 2 — Status value enumeration unknown:**
We filter `Status ne 'Gediscontinueerd'` but the docs don't list all possible Status values. If CBS has values like `"Gearchiveerd"` or `"Experimenteel"` we might be including or excluding datasets unexpectedly. Before the next catalog refresh, do a quick: `GET /CBS/Datasets?$select=Status&$apply=groupby((Status))` to get all unique values and decide if the filter should be a whitelist (`Status eq 'Regulier'`) instead of a blacklist.

### What does NOT need changing

- `{DimName}Codes` pattern in MCP — already correct
- `_extract_period()` regex for "Beschikbaar vanaf" — fine; even when it returns empty the `modified` date fallback works
- `embed_text = f"{title}. {summary}"` format — summary is description[:300], which is the most semantically rich text available
- Recency filter threshold of 2015 — all active datasets pass it via `modified` date, so the filter is effectively a no-op on this catalog; harmless

### Files to change (when plan is approved)

1. `scripts/build_cbs_catalog.py` — replace manual `$skip` loop with `@odata.nextLink` follower (~8 lines changed)
2. After next catalog run: spot-check unique `Status` values and decide whitelist vs blacklist filter

## Sources (market signals)
- [The AI Skills Employers Actually Want in 2026 (EU job listings) — SearchQualify](https://searchqualify.com/blog/ai-skills-employers-actually-want-2026-eu-job-listings)
- [AI Developer Hiring 2026: Skills That Actually Matter — digitalapplied.com](https://www.digitalapplied.com/blog/ai-developer-hiring-skills-that-matter-2026)
- [Top 10 Most In-Demand AI Engineering Skills 2026 — Second Talent](https://www.secondtalent.com/resources/most-in-demand-ai-engineering-skills-and-salary-ranges/)
- [AI Engineer Recruitment Netherlands — iduet.nl](https://iduet.nl/en/data-ai-recruitment-data-scientists-machine-learning-engineers/ai-engineer-recruitment/)
- [ML engineer jobs in Netherlands — Glassdoor](https://www.glassdoor.com/Job/netherlands-machine-learning-engineer-jobs-SRCH_IL.0,11_IN178_KO12,37.htm)
