# Poldercheck — Agent Notes

Compact guide for working in this repo. For the public overview see `README.md`; for server deployment see `docs/HANDOVER.md`.

---

## Project basics

- Python 3.12. Single package, not a monorepo. Tests expect `pytest` from the repo root (`pytest.ini` sets `pythonpath = .`).
- `requirements.txt` pins CPU-only PyTorch via `--extra-index-url https://download.pytorch.org/whl/cpu`. Do not remove this line; runtime and CI have no GPU, and the CUDA wheel pulls ~6 GB of nvidia packages.
- `src/graph.py` is the main entry point. `src/app.py` is the Streamlit UI.

## Environment setup

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill keys.

Required env vars (see `src/agents/config.py` for full resolution):

| Var | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint (MVP: `https://api.deepseek.com`) |
| `LLM_API_KEY` | Key for LLM calls; falls back to `OPENROUTER_API_KEY`, then `ANTHROPIC_AUTH_TOKEN` |
| `POLDERCHECK_MODEL` | Default model for all agents |
| `OPENROUTER_API_KEY` | Required for embeddings (model is `qwen/qwen3-embedding-8b` via OpenRouter) |

Optional per-agent overrides: `POLDERCHECK_POLITICAL_MODEL`, `POLDERCHECK_DATA_MODEL`, `POLDERCHECK_SYNTHESIS_MODEL`, plus matching `_BASE_URL` / `_API_KEY`.

## Build the local corpus

`chroma_db/` is gitignored. A fresh clone must build it:

```bash
# Needs data/static/*.pdf and data/processed/manifesto_corpus.csv present.
python scripts/rebuild_embeddings.py
```

This drops and rebuilds two ChromaDB collections:

- `poldercheck_static` — `src/ingest/chunk.py` indexes Manifesto Project CSV quasi-sentences, CPB/PBL PDFs, and party manifesto PDFs.
- `cbs_catalog` — built from the committed `data/catalog/cbs_catalog.jsonl`.

Intentional exclusion: `src/ingest/retrieve.py` excludes `type == "manifesto"` (raw Manifesto Project CSV quasi-sentences) and only returns `manifesto_pdf`, `cpb_analysis`, and `pbl_analysis` chunks. The CSV is still produced by `fetch_manifestos.py` but is not served to agents; the PDF manifestos are preferred.

## Running and testing

```bash
# Quick query
python -m src.graph

# UI
streamlit run src/app.py

# Tests
pytest tests/ -v
pytest tests/test_response_contract.py -v

# RAGAS eval (costs LLM-judge calls; only on main in CI)
python src/eval/run_eval.py
```

## Architecture

LangGraph flow in `src/graph.py`:

```
query_planner → political → data → synthesis
```

- `political` runs before `data`; the data analyst receives `political_response` as context.
- `query_planner` only runs in `mode="fast"`, generating Dutch CBS search terms.
- State toggles: `include_manifestos`, `include_tk`, `include_cbs`, plus `language` and `pedagogical`.

### Political analyst (`src/agents/political.py`)

- Static retrieval from `poldercheck_static`.
- Live Tweede Kamer search via the OpenTK MCP server (`npx -y @r-huijts/opentk-mcp`), launched over stdio. Requires Node/npm in PATH.
- Falls back to static-only if OpenTK is unavailable.
- `mode="fast"` uses a fixed pipeline; `mode="deep"` uses a React agent.

### Data analyst (`src/agents/data.py`)

- Fast mode: semantic search over `cbs_catalog`, then parallel direct OData v4 HTTP fetches of top datasets (`httpx`).
- Deep mode: React agent that discovers datasets via `search_cbs_catalog` (ChromaDB), then calls the CBS MCP server (`mcp-cbs-cijfers-open-data --stdio`) for `get_dimensions`, `get_dimension_values`, and `query_observations`.
- The deep agent must never hardcode a year filter such as `startswith(Perioden,'2020')`; use `$orderby=Perioden desc` for recent figures and fetch full series for trends.
- Protocol version is pinned process-wide: `mcp.types.LATEST_PROTOCOL_VERSION = "2024-11-05"`.
- If the MCP binary is missing from PATH, deep mode falls back to fast mode.

### Synthesis (`src/graph.py`)

- Uses the `synthesis` agent config.
- Prompt enforces inline `[Source, Year]` citations, max ~300 words, and a `## Sources` section.

## Response contract

`src/eval/contract.py` enforces:

- Every response must contain a `[Source, Year]` citation **or** an explicit not-found sentence.
- Max 350 words.

CI runs this via `pytest` on every push/PR. The RAGAS eval runs only on `main`.

## Key gotchas

- **CBS catalog source of truth**: `data/catalog/cbs_catalog.jsonl` is committed. Refresh it with `python scripts/build_cbs_catalog.py`. ChromaDB rebuilds from this JSONL on first use.
- **OData v4 only**: use `datasets.cbs.nl`. v3 `opendata.cbs.nl` identifiers with `ENG` suffix return 404.
- **Embedding model mismatch in comments**: some headers say `Qwen3-Embedding-0.6B`, but the actual model in `src/ingest/embed.py` is `qwen/qwen3-embedding-8b`. Trust `EMBED_MODEL_ID`.
- **Recency filtering**: `retrieve_cbs_datasets` prefers datasets whose `modified`/`period` year is ≥ 2015.
- **Chunk.py is resumable**: it skips IDs already present in `poldercheck_static`, so re-running is safe.
- **No opencode.json / .cursorrules** in this repo; this file is the only agent instructions source.
