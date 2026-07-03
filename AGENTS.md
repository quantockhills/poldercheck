# Poldercheck — Agent Notes

Compact developer reference. For the public overview see `README.md`.
AGENTS.md is gitignored — local-only context.

---

## Project basics

- Python 3.12. `pytest.ini` sets `pythonpath = .`.
- `requirements.txt` pins CPU-only PyTorch via `--extra-index-url https://download.pytorch.org/whl/cpu`. **Do not remove** this line — the CUDA wheel pulls ~6 GB of dead weight on CPU-only machines.
- `chroma_db/` and `docs/opentk-mcp/` are gitignored; fresh clones must build both.
- `src/graph.py` is the entry point. `src/app.py` is the Streamlit UI.

## Environment setup

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill keys.
4. Clone and build OpenTK MCP server (needed for supplementary TK full-text search):

   ```bash
   git clone https://github.com/r-huijts/opentk-mcp docs/opentk-mcp
   cd docs/opentk-mcp && npm install && npm run build
   ```

   Node must be in PATH. Without it, political analysis still works via the OData API — only the supplementary OpenTK full-text pass is skipped.

### Required env vars (see `src/agents/config.py`)

| Var | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | API key; falls back to `OPENROUTER_API_KEY`, then `ANTHROPIC_AUTH_TOKEN` |
| `POLDERCHECK_MODEL` | Default model for all agents |
| `OPENROUTER_API_KEY` | For embeddings (`qwen/qwen3-embedding-8b` via OpenRouter); also falls back to `LLM_API_KEY` |
| `MANIFESTO_API_KEY` | Free Manifesto Project API key |

Optional per-agent overrides: `POLDERCHECK_POLITICAL_MODEL`, `POLDERCHECK_DATA_MODEL`, `POLDERCHECK_SYNTHESIS_MODEL`, plus matching `_BASE_URL` / `_API_KEY`. An `opentk_agent` config also exists for the political discover subgraph's triage LLM calls.

Hosting-only: `ACCESS_TOKEN` (gates app behind `/?token=<value>`), `PRESENTATION_MODE=1` (disables manifesto corpus and shows notice).

## Build the local corpus

```bash
# Requires data/static/*.pdf and data/processed/manifesto_corpus.csv
python scripts/rebuild_embeddings.py
```

Drops and rebuilds two ChromaDB collections in `chroma_db/`:
- `poldercheck_static` — Manifesto Project CSV quasi-sentences, party manifesto PDFs, CPB/PBL PDFs. Resumable: skips already-indexed IDs.
- `cbs_catalog` — 1,297 CBS dataset titles + summaries with metadata, built from the committed `data/catalog/cbs_catalog.jsonl`.

Refresh the catalog JSONL with `python scripts/build_cbs_catalog.py` (hits the OData v4 API). ChromaDB builds from the JSONL lazily on first catalog search.

**Important**: `src/ingest/retrieve.py` excludes `type == "manifesto"` (raw Manifesto CSV quasi-sentences). Agents only see `manifesto_pdf`, `cpb_analysis`, and `pbl_analysis` chunks.

Note: manifesto/CPB/PBL corpus work is currently paused — the focus is on Tweede Kamer live search via OData + CBS data via DuckDB.

## Issues and roadmap

GitHub issues serve as both bug tracker and development roadmap. Labels: `enhancement`, `bug`, `quickfix`, `near-term`, `priority: high`, `potential improvement`.
The README references specific issues as roadmap anchors (e.g. #8 critic agent, #21/#22 additional CPB/PBL reports). When making feature changes, check if there's a linked issue and reference the number in commits.

## Running and testing

```bash
python -m src.graph         # quick query (uses the hardcoded English query in __main__)
streamlit run src/app.py    # UI at http://localhost:8501

pytest tests/ -v                          # all tests
pytest tests/test_response_contract.py -v # contract tests only
ruff check src/ tests/                    # lint (line-length 120, N806 suppressed)

python src/eval/run_eval.py               # RAGAS eval — costs LLM-judge calls; CI only on main
```

Tests auto-load `.env` via `tests/conftest.py`. Langfuse tracing auto-enables when `LANGFUSE_PUBLIC_KEY` is set (`graph.py:_langfuse_callbacks`). Set `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` (defaults to `cloud.langfuse.com`). Each graph run emits a full trace — useful for diagnosing slow nodes, tool calls, and CBS MCP responses.

When working on LangChain/LangGraph code in this project, use the LangChain docs MCP (`docs.langchain.com/mcp`) for reference — it's configured in `~/.config/opencode/opencode.jsonc`.

## Architecture — full node walkthrough

### 1. query_planner (`graph.py`)

LLM generates 5-7 Dutch CBS search terms from the query. Only runs in `mode="fast"`; in deep mode it returns `[]` (the data agent discovers its own terms).

### 2. political (`graph.py` → `src/agents/political.py` → `src/agents/political_discover.py`)

Wraps a **political discover subgraph** with a 300s timeout. On timeout/exception, falls back to static-only LLM synthesis using the `political_analyst.txt` prompt.

The subgraph has three nodes:

**2a. plan (`_plan_node`)**:
- LLM generates 15 diverse Dutch search terms + 3-5 short OData root keywords (4-9 chars for Onderwerp substring matching)
- Extracts date range from query (default: last 5 years; clamped to >= 2018)
- Creates year buckets for parallel OData search
- Searches ChromaDB static corpus (15 passages via `retrieve_static`)

**2b. search (`_search_node`)**:
- **OData primary search** — per-year-bucket parallel calls to `gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document`, filtering on Onderwerp substring + 5 debate types + excluding stemming. Downloads full text (PDF/DOCX extraction) up to 30 docs/year.
- **BM25 ranking** — chunked at 1,500 chars, compound-aware tokenizer (`vrouwenquotum` matches `quotum`), champion chunk per debate.
- **LLM triage** — one call scores BM25 top-40 champion passages 0-10 against the query. Keeps top-15 debates.
- **Party excerpt extraction** — purely local regex over full text (no MCP). Finds snippets around party name mentions, preferring those with query terms.
- **OpenTK MCP supplementary search** — runs only when `include_tk=True`. Calls `search_tk_filtered` (up to 10 terms), `analyze_document_relevance` (up to 15 docs), `get_document_content` (top 3). Skips gracefully if MCP binary unavailable.

**2c. synthesize (`_synthesize_node`)**:
- Assembled prompt from static passages + OData ranked results (doc ID, date, champion passage, party excerpts) + OpenTK content + date range + coverage note
- Uses its own system prompt (date-aware + language-aware), **not** the `political_analyst.txt` file
- Output: a single response with `[DocumentID, Date]` citations

### 3. data (`graph.py` → `src/agents/data.py`)

Routes on `mode` and `cbs_mode`:

| mode | cbs_mode | Function | Data source |
|---|---|---|---|
| `fast` | (ignored) | `_run_fast` | ChromaDB catalog → direct OData v4 HTTP via `_fetch_cbs_data` |
| `deep` | `duckdb` **(default)** | `_run_duckdb_parallel` | ChromaDB catalog → CSV ZIP download → in-memory DuckDB SQL |
| `deep` | `mcp` | `_run_deep` | ChromaDB catalog → CBS MCP server (`mcp-cbs-cijfers-open-data`) |

`_run_deep_duckdb` at `data.py:595` is dead code — the routing at line 678 calls `_run_duckdb_parallel` instead.

**3a. `_run_fast`** (mode=fast):
- Calls `retrieve_cbs_datasets` (ChromaDB catalog search)
- Parallel `_fetch_observations` → `_fetch_cbs_data` (direct `httpx` OData v4 HTTP to `datasets.cbs.nl/odata/v1/CBS/{id}`)
- LLM synthesis of raw observations

**3b. `_run_duckdb_parallel`** (mode=deep, cbs_mode=duckdb):
Orchestrator-worker LangGraph subgraph:
- **discover**: LLM decomposes query → 3-5 Dutch sub-topics → per-sub-topic ChromaDB catalog search → LLM ranks candidates (prioritizing diversity)
- **fanout**: Sends to N parallel workers (one per CBS dataset)
- **worker**: Downloads CSV ZIP from `datasets.cbs.nl/CSV/CBS/nl/{id}` → `download_cbs_dataset` loads into in-memory DuckDB → `run_sql` tool for SQL exploration → finding
- **synthesize**: Merges worker findings into one response

DuckDB is `:memory:` connection. `_reset_duck()` drops all tables between runs. Quoted identifiers: `"85773NED_Observations"`, `"85773NED_MeasureCodes"`. Always join with MeasureCodes for readable labels.

**3c. `_run_deep`** (mode=deep, cbs_mode=mcp):
React agent with `search_cbs_catalog` + CBS MCP server tools:
- `get_dimensions`, `get_dimension_values`, `query_observations`
- Per-tool 30s asyncio timeout
- `mcp.types.LATEST_PROTOCOL_VERSION` pinned to `"2024-11-05"`

All three modes load `src/prompts/data_analyst.txt` as the system prompt (via `_system_prompt()`).

### 4. synthesis (`graph.py:synthesis_node`)

Receives `political_response` and `data_response` as pre-synthesized strings (not raw passages). Varies the synthesis instruction based on which sources are present (political only, data only, or both). Uses `reasoning_effort: "high"` with `thinking.type: "enabled"` — only works with DeepSeek or Claude.

Enforces: `^N` inline citations, max ~300 words prose, `## Sources` numbered list. Pedagogical mode adds parenthetical Dutch term explanations.

### Streamlit UI (`src/app.py`)

Current sidebar state:
- **Mode**: only "Deep (thorough)" — fast mode code exists but is "coming soon"
- **CBS query mode**: only "DuckDB (local SQL)" — MCP mode code exists but is "coming soon"
- **Language**: NL/EN toggle (drives synthesis language + on-the-fly translation)
- **Sources**: TK debates, manifestos/CPB (disabled when ChromaDB unavailable), CBS data
- **CBS datasets to query**: 1-10, default 5
- **Pedagogical mode**: explains Dutch terms inline

Two main tabs: Search (with live status polling + stop button) and History (past searches with delete).

### History and storage

`src/storage.py`: conversations saved as JSON in `data/history/`. Each entry: query, settings, final_response, political_response, data_response, political_passages.

## Response contract

`src/eval/contract.py` enforces at test time:
- Every response must have a `^N` inline citation **or** an explicit "I did not find..." sentence
- Max 350 words (300 budget + citation slack)

CI runs `test_response_contract.py` on every push. RAGAS eval (`run_eval.py`) runs only on `main`.

## Key gotchas

- **CBS catalog vs README**: catalog has 1,297 datasets (the README says 4,000+ — the catalog may be filtered or incomplete).
- **OData v4 only**: use `datasets.cbs.nl`. v3 `opendata.cbs.nl` identifiers return 404.
- **Embedding model mismatch**: `src/ingest/embed.py` header says `Qwen3-Embedding-0.6B`, but `EMBED_MODEL_ID` is actually `qwen/qwen3-embedding-8b`. Trust `EMBED_MODEL_ID`.
- **Recency filter**: `retrieve_cbs_datasets` prefers datasets with `modified`/`period` >= 2015.
- **Chunk.py is resumable**: uses `get_or_create_collection`, skips existing IDs.
- **Synthesis uses thinking mode**: `reasoning_effort: "high"` + `thinking.type: "enabled"` — DeepSeek or Claude only.
- **Political prompt file is fallback-only**: `political_analyst.txt` is only used in the static-only fallback path (`political.py:172`). The real political synthesis prompt is in `_synthesize_node`.
- **`_run_deep_duckdb` is dead code**: never called by the routing logic at `data.py:678`.
- **OpenTK is supplementary**: the OData API is the primary search. OpenTK MCP adds full-text content for the top 3 matched docs. Party excerpts are entirely local.
- **query_planner terms flow into political discover**: the `search_terms` generated by `_plan_node` are what feed both the BM25 ranking and the OpenTK search. The top-level `query_planner` just feeds the data node.
- **Tests need chroma_db**: `test_retrieve.py` skips gracefully if `chroma_db/` doesn't exist.
