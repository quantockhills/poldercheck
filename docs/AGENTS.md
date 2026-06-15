# Poldercheck — Technical Reference

Internal developer guide. For the public-facing overview see `README.md`.

---

## File structure

```
poldercheck/
├── src/
│   ├── graph.py                  # LangGraph wiring — the main entry point
│   ├── app.py                    # Streamlit UI
│   ├── agents/
│   │   ├── config.py             # Provider-agnostic LLM config from env vars
│   │   ├── political.py          # Political analyst agent (static + OpenTK)
│   │   └── data.py               # Data analyst agent (CBS MCP)
│   ├── ingest/
│   │   ├── embed.py              # OpenRouter embeddings client (Qwen3-8B)
│   │   ├── chunk.py              # Build poldercheck_static ChromaDB collection
│   │   ├── retrieve.py           # Semantic search over both collections
│   │   └── fetch_manifestos.py   # Fetch Manifesto Project CSV via API
│   ├── eval/
│   │   ├── eval_set.jsonl        # 7 benchmark queries with pass/fail contracts
│   │   ├── contract.py           # Contract checker
│   │   └── run_eval.py           # Run eval suite
│   └── prompts/
│       ├── political_analyst.txt
│       ├── data_analyst.txt
│       └── critic.txt
├── scripts/
│   ├── build_cbs_catalog.py      # Fetch CBS OData v4 catalog → JSONL
│   └── rebuild_embeddings.py     # Drop + rebuild both ChromaDB collections
├── data/
│   ├── catalog/cbs_catalog.jsonl # 1,297 active CBS datasets (in git)
│   ├── static/                   # CPB + PBL PDFs (gitignored, rsync to server)
│   └── processed/
│       └── manifesto_corpus.csv  # 56k quasi-sentences from Manifesto Project (gitignored)
├── chroma_db/                    # ChromaDB on disk (gitignored, rsync to server)
├── tests/
│   ├── test_response_contract.py
│   └── test_retrieve.py
├── docs/
│   ├── AGENTS.md                 # this file
│   └── HANDOVER.md               # server setup guide
├── .env                          # API keys (gitignored)
├── .env.example
├── requirements.txt
└── Dockerfile
```

---

## Architecture

### LangGraph graph (`src/graph.py`)

```
User query
    │
    ▼
data_node (async, 60s timeout)
    │  run_data_analyst(query)
    │  → CBS catalog semantic search (local ChromaDB, ~50ms)
    │  → CBS MCP server (Go binary, stdio)
    │      → get_dimensions → get_observations
    ▼
political_node (async, 60s timeout)
    │  run_political_analyst_v2(query, prior_context=data_response)
    │  → retrieve_static (ChromaDB, manifesto + CPB/PBL)
    │  → OpenTK MCP server (Node, via npx @r-huijts/opentk-mcp, stdio)
    │      → search_tk → analyze_document_relevance → get_document_details
    │  Falls back to static-only if OpenTK unavailable
    ▼
synthesis_node
    │  DeepSeek v4 Pro, max 250 words, inline citations
    ▼
PolderState.final_response
```

`data_node` runs first so its output is available as `prior_context` to the political analyst. The graph is sequential; parallelisation is tracked in issue #15.

### ChromaDB collections

| Collection | Built by | Contents | Size |
|---|---|---|---|
| `poldercheck_static` | `chunk.py` | Manifesto quasi-sentences (excl. H-coded headers) + CPB/PBL PDF chunks | ~54k + ~3k chunks |
| `cbs_catalog` | `retrieve.py` or `build_cbs_catalog.py` | 1,297 active CBS dataset titles + summaries | 1,297 entries |

Both use cosine similarity space. Embeddings via `qwen/qwen3-embedding-8b` through OpenRouter (`src/ingest/embed.py`).

`cbs_catalog` is auto-built on first run from `data/catalog/cbs_catalog.jsonl` if missing. `poldercheck_static` requires running `scripts/rebuild_embeddings.py` (or is rsync'd from another machine).

### CBS data flow

```
CBS OData v4 API (datasets.cbs.nl)
    │  build_cbs_catalog.py (run quarterly or after elections)
    ▼
data/catalog/cbs_catalog.jsonl   ← committed to git
    │  retrieve.py: _build_cbs_collection_from_jsonl()
    ▼
ChromaDB: cbs_catalog            ← gitignored, built locally
    │  retrieve_cbs_datasets(query) — semantic search, 3 query variants
    ▼
CBS MCP server (Go)              ← binary in PATH or local
    │  get_dimensions + get_observations
    ▼
Actual CBS numbers in agent response
```

### Static corpus data flow

```
Manifesto Project API            CBS PDFs (manual download)
    │  fetch_manifestos.py           │
    ▼                                ▼
data/processed/manifesto_corpus.csv  data/static/*.pdf
    │                                │
    └──────── chunk.py ──────────────┘
                   │  embed via OpenRouter
                   ▼
        ChromaDB: poldercheck_static
                   │
              retrieve_static(query)
```

---

## Environment variables

All in `.env`. Never commit this file.

| Variable | Required | Purpose |
|---|---|---|
| `LLM_BASE_URL` | Yes | OpenAI-compatible base URL for LLM calls |
| `LLM_API_KEY` | Yes | API key for LLM calls (also tried: `OPENROUTER_API_KEY`, `ANTHROPIC_AUTH_TOKEN`) |
| `POLDERCHECK_MODEL` | Yes | Default model for all agents |
| `POLDERCHECK_DATA_MODEL` | No | Override model for data analyst only |
| `OPENROUTER_API_KEY` | Yes | OpenRouter key for Qwen3 embeddings |
| `MANIFESTO_API_KEY` | Only for re-fetch | Manifesto Project API key |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing (optional) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing (optional) |
| `LANGFUSE_HOST` | No | Langfuse host (default: cloud.langfuse.com) |

Per-agent model overrides: `POLDERCHECK_POLITICAL_MODEL`, `POLDERCHECK_SYNTHESIS_MODEL`, and matching `_BASE_URL` / `_API_KEY` variants. See `src/agents/config.py` for full resolution order.

Current `.env` (MVP):
```
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=...
POLDERCHECK_MODEL=deepseek-v4-pro
POLDERCHECK_DATA_MODEL=deepseek-v4-flash
OPENROUTER_API_KEY=...
MANIFESTO_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# One-time: build static corpus (requires data/static/ PDFs and manifesto CSV)
python scripts/rebuild_embeddings.py

# Run a query directly
python -m src.graph

# Streamlit UI
streamlit run src/app.py
```

The CBS MCP binary (`mcp-cbs-cijfers-open-data`) must be in PATH. It is a Go binary compiled from `github.com/dstotijn/mcp-cbs-cijfers-open-data`. A patched version (fixing `DimensionValues` → `{dimension}Codes`) is in `docs/cbs-mcp-src/` — compile with `go build -o mcp-cbs-cijfers-open-data .` and add to PATH.

OpenTK MCP (`@r-huijts/opentk-mcp`) is fetched automatically via `npx` at query time. Requires Node/npm.

---

## Debugging

### DEBUG_LOG

Every node prints timing and key decisions to stdout:

```
DEBUG_LOG: catalog found 5 candidates: ['85819NED', '83163NED', ...]
DEBUG_LOG: data node took 31.7s
DEBUG_LOG: political node took 15.5s
DEBUG_LOG: synthesis node took 37.2s
```

`DEBUG_LOG = print` is intentionally kept even when features work — it gives the first signal when something regresses.

### Langfuse tracing

Every graph run emits a trace to Langfuse (cloud.langfuse.com) when keys are set in `.env`. Each trace contains:
- All LLM calls with prompts, completions, token counts, latency
- Agent tool calls and their results
- Node-level timing

To inspect a run: go to cloud.langfuse.com → Traces. Each trace is one full query through the graph. Useful for diagnosing which node is slow, which tool calls the agent is making, and what the CBS MCP actually returns.

The callback handler is in `src/graph.py` (`_langfuse_callbacks()`). It fails silently if keys are missing.

### Tracing a CBS timeout

1. Open the Langfuse trace for the failing run
2. Expand the `data_node` span
3. Look at tool calls — if the agent is calling `query_datasets` repeatedly, it's ignoring the catalog pre-filter (regression in the prompt)
4. If it's calling `get_observations` on ENG-suffixed identifiers (e.g. `85773ENG`), the catalog was built from OData v3 — rebuild from v4

### MCP protocol debugging

```bash
# Test CBS MCP directly
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | mcp-cbs-cijfers-open-data

# Inspect OpenTK MCP
npx @modelcontextprotocol/inspector npx @r-huijts/opentk-mcp
```

### Embedding issues

If retrieval returns section headers instead of policy text, check for `cmp_code == 'H'` rows in the manifesto corpus — these are headings with no policy content. `chunk.py` filters them at build time; if you see them in results the collection needs rebuilding.

To verify what's in ChromaDB:
```python
import chromadb
client = chromadb.PersistentClient("./chroma_db")
col = client.get_collection("poldercheck_static")
print(col.count())
results = col.query(query_texts=["betaalbare huren"], n_results=5)
print(results["documents"])
```

---

## CBS OData v4 API — key facts

Base URL: `https://datasets.cbs.nl/odata/v1/CBS/`

- Catalog: `/CBS/Datasets?$filter=Status ne 'Gediscontinueerd'` → 1,297 active datasets, all Dutch (NED suffix), no auth required
- Data: `/{id}/Observations`, `/{id}/Dimensions`, `/{id}/{DimName}Codes`
- Pagination via `@odata.nextLink` in response body
- `Distributions[].Format == "odata"` → `DownloadUrl` gives the OData endpoint for a dataset
- No rate limits documented; CBS recommends ≤10,000 rows per request

OData v3 (`opendata.cbs.nl`) had ENG/NED variants; v4 (`datasets.cbs.nl`) is Dutch-only. Always use v4 — ENG identifiers return 404 from the MCP server.

Refresh the catalog quarterly or after elections:
```bash
python scripts/build_cbs_catalog.py
```

---

## Known quirks

| Quirk | Location | Notes |
|---|---|---|
| CBS MCP rejects modern protocol versions | `src/agents/data.py` top | `mcp.types.LATEST_PROTOCOL_VERSION = "2024-11-05"` set process-wide |
| OpenTK needs `npx` in PATH | `src/agents/political.py` | Fails gracefully to static-only; check `which npx` on server |
| Manifesto passages are quasi-sentences | `data/processed/` | Short by design; the Manifesto Project codes at sentence level, not paragraph |
| `period` field is empty for ~20% of CBS datasets | `data/catalog/cbs_catalog.jsonl` | `modified` date is used instead for recency filtering; all active datasets have modified dates post-2019 so the 2015 threshold passes all |
| `feat/opentk-mcp` branch | git | OpenTK integration is code-complete but not merged — needs one end-to-end test first |
