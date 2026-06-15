# Poldercheck — Server Handover

Steps to get poldercheck running on a new server. The target is:
`sasha@46.225.153.227:/mnt/HC_Volume_105854834/poldercheck/`

---

## Prerequisites on the server

```bash
# Python 3.11+
python3 --version

# Node + npm (for OpenTK MCP)
node --version && npm --version

# Go (only needed if recompiling CBS MCP binary)
go version
```

---

## Step 1 — SSH key for GitHub (private repo)

On the server:
```bash
ssh-keygen -t ed25519 -C "poldercheck-server"
cat ~/.ssh/id_ed25519.pub
```

Add the output to: GitHub → Settings → SSH and GPG keys → New SSH key.

---

## Step 2 — Clone the repo

```bash
cd /mnt/HC_Volume_105854834
git clone git@github.com:quantockhills/poldercheck.git
cd poldercheck
```

---

## Step 3 — Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 4 — Copy API keys

From your local machine:
```bash
scp "/home/madhav22m/gitrepos/AI projects/poldercheck/.env" \
    sasha@46.225.153.227:/mnt/HC_Volume_105854834/poldercheck/.env
```

---

## Step 5 — Transfer data (gitignored files)

These three directories are not in git and must be rsync'd from the machine where the embeddings were built.

From your local machine:
```bash
# ChromaDB — pre-built vector store (~few hundred MB)
rsync -av --progress \
    "/home/madhav22m/gitrepos/AI projects/poldercheck/chroma_db/" \
    sasha@46.225.153.227:/mnt/HC_Volume_105854834/poldercheck/chroma_db/

# Manifesto corpus CSV (~56k rows)
rsync -av --progress \
    "/home/madhav22m/gitrepos/AI projects/poldercheck/data/processed/" \
    sasha@46.225.153.227:/mnt/HC_Volume_105854834/poldercheck/data/processed/

# CPB + PBL PDFs
rsync -av --progress \
    "/home/madhav22m/gitrepos/AI projects/poldercheck/data/static/" \
    sasha@46.225.153.227:/mnt/HC_Volume_105854834/poldercheck/data/static/
```

`data/catalog/cbs_catalog.jsonl` is already in git — no transfer needed.

---

## Step 6 — CBS MCP binary

The CBS MCP server is a Go binary. The compiled binary lives locally at
`/home/madhav22m/gitrepos/AI projects/poldercheck/docs/cbs-mcp-src/mcp-cbs-cijfers-open-data`
(gitignored — not in the repo). Transfer it:

```bash
scp "/home/madhav22m/gitrepos/AI projects/poldercheck/docs/cbs-mcp-src/mcp-cbs-cijfers-open-data" \
    sasha@46.225.153.227:/usr/local/bin/mcp-cbs-cijfers-open-data
ssh sasha@46.225.153.227 "chmod +x /usr/local/bin/mcp-cbs-cijfers-open-data"
```

Or recompile on the server (requires Go). One patch is required: in `main.go`,
the `get_dimension_values` tool must call `/{dataset}/{dimension}Codes` not
`/DimensionValues?$filter=Dimension eq '...'`. See AGENTS.md for details.

```bash
git clone https://github.com/dstotijn/mcp-cbs-cijfers-open-data
cd mcp-cbs-cijfers-open-data
# Apply patch described in docs/AGENTS.md, then:
go build -o /usr/local/bin/mcp-cbs-cijfers-open-data .
```

Verify:
```bash
which mcp-cbs-cijfers-open-data
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | mcp-cbs-cijfers-open-data
```

---

## Step 7 — Verify

```bash
cd /mnt/HC_Volume_105854834/poldercheck
source venv/bin/activate

# Quick retrieval test
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.ingest.retrieve import retrieve_static, retrieve_cbs_datasets
r = retrieve_static('betaalbare huren', n_results=2)
print('Static:', [p['metadata']['source'] for p in r])
c = retrieve_cbs_datasets('huurprijzen woningen', n_results=3)
print('CBS:', [d['identifier'] for d in c])
"

# Full pipeline test (takes ~90s, makes real API calls)
python -c "
import asyncio
from dotenv import load_dotenv; load_dotenv()
from src.graph import run_query
result = asyncio.run(run_query('wat zijn de huurprijzen in nederland?'))
print(result['final_response'][:300])
"
```

---

## Updating

```bash
cd /mnt/HC_Volume_105854834/poldercheck
git pull origin main

# If src/ingest/ changed, rebuild the static ChromaDB collection:
source venv/bin/activate
python scripts/rebuild_embeddings.py

# If CBS catalog is stale (run quarterly):
python scripts/build_cbs_catalog.py
```

---

## What lives where

| Thing | In git? | Notes |
|---|---|---|
| All Python source | Yes | `src/`, `scripts/`, `tests/` |
| CBS catalog JSONL | Yes | `data/catalog/cbs_catalog.jsonl` |
| ChromaDB | No | rsync from build machine, or rebuild (~10 min) |
| Manifesto CSV | No | rsync from build machine, or re-fetch via `fetch_manifestos.py` |
| CPB/PBL PDFs | No | rsync from build machine, or re-download from cpb.nl / pbl.nl |
| `.env` | No | scp from build machine |
| CBS MCP binary | No | scp from build machine or recompile |
