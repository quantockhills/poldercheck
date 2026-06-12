# Poldercheck : Build Plan

*This document is written for an AI coding agent. Follow the steps in order.
Do not skip ahead. Each step builds on the previous one.*

---

## Environment

- OS: WSL2 on Windows (Ubuntu)
- Python: 3.11+
- Node: required for opentk-mcp (install via nvm if not present)
- Go: required for CBS MCP server (install via `sudo apt install golang-go`)
- API: OpenRouter key stored in `.env`

---

## Directory structure

```
poldercheck/
├── data/
│   └── static/                  # downloaded PDFs (manifestos, CPB, PBL)
├── src/
│   ├── ingest/
│   │   ├── download.py          # download and save static PDFs
│   │   ├── chunk.py             # chunk and embed into ChromaDB
│   │   └── retrieve.py          # semantic search over ChromaDB
│   ├── agents/
│   │   ├── config.py            # model config per agent
│   │   ├── political.py         # political analyst agent (LangGraph node)
│   │   └── data.py              # data analyst agent (LangGraph node)
│   ├── graph.py                 # LangGraph graph definition
│   ├── prompts/
│   │   ├── political_analyst.txt
│   │   └── data_analyst.txt
│   ├── eval/
│   │   ├── eval_set.jsonl       # benchmark queries + expectations (Step 14)
│   │   └── run_eval.py          # RAGAS groundedness/faithfulness eval (Step 14)
│   └── app.py                   # Streamlit frontend
├── chroma_db/                   # persisted vector store (gitignored)
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieve.py
│   ├── test_agents.py
│   └── test_response_contract.py  # deterministic citation/not-found checks (Step 14)
├── Dockerfile
├── .env                         # gitignored
├── .github/
│   └── workflows/
│       └── tests.yml
├── requirements.txt
└── README.md
```

---

## Step 0 : Environment setup (30 min)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies.
# IMPORTANT: install CPU-only torch FIRST. sentence-transformers depends on
# torch, and the default Linux wheel is the CUDA build (~6 GB of nvidia-*
# packages that are dead weight on machines without a GPU - dev box, CI
# runners, and the Azure container are all CPU-only).
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install langchain langchain-community langchain-openai \
            chromadb sentence-transformers langgraph \
            openai streamlit pypdf requests python-dotenv pytest \
            ragas langfuse

# Install Node (for opentk-mcp) via nvm if not present
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# Test opentk-mcp installs correctly
npx -y @r-huijts/opentk-mcp --help

# Install Go (for CBS MCP server)
sudo apt update && sudo apt install -y golang-go

# Test CBS MCP server
go run github.com/dstotijn/mcp-cbs-cijfers-open-data@latest --help
```

Create `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

Create `requirements.txt` (the extra-index line makes pip resolve the +cpu
torch wheel everywhere, including CI and Docker):
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch

langchain
langchain-community
langchain-openai
chromadb
sentence-transformers
langgraph
openai
streamlit
pypdf
requests
python-dotenv
pytest
ragas
langfuse
```

---

## Step 1 : Acquire static corpus (half day)

The static corpus has two parts:

**Part A: Manifesto Project API (no PDFs)**

The Manifesto Project (manifestoproject.wzb.eu) provides a free API giving access
to every major Dutch party manifesto since 1945, coded at the quasi-sentence level.
Each quasi-sentence has a policy category code (housing, immigration, environment,
economy, etc.), party name, and election year. This is dramatically cleaner than PDF
parsing: the text is already chunked, labelled, and in a queryable format.

Register for a free API key at: https://manifesto-project.wzb.eu/signup

```bash
pip install manifestopy  # unofficial Python client, or use requests directly
```

```python
# src/ingest/fetch_manifestos.py
import requests
import pandas as pd
from pathlib import Path

# Manifesto Project REST API
API_KEY = "your_manifesto_api_key"  # from .env
BASE_URL = "https://manifesto-project.wzb.eu/api/v1"

# Dutch party codes in the Manifesto Project
# Netherlands country code: 21
# Key parties and their codes:
# IMPORTANT: verify every code against the party list in the Manifesto Project
# documentation/codebook before fetching. An earlier draft had 22110 listed
# twice ("VVD") - a duplicate dict key silently drops a party. Also consider
# adding the parties that emerged around the 2023 election (BBB, NSC) since
# ELECTIONS includes 202311.
DUTCH_PARTIES = {
    22110: "VVD",
    22320: "PvdA",
    22526: "D66",
    22410: "CDA",
    22720: "GroenLinks",
    22951: "PVV",
    22220: "SP",
    22521: "ChristenUnie",
}

assert len(DUTCH_PARTIES) == 8, "duplicate party code dropped an entry"

# Elections to include (format: YYYYMM)
ELECTIONS = ["202311", "202111", "202103", "201703"]  # 2023, 2021, 2017


def fetch_manifesto_corpus(party_id: int, election_date: str) -> list[dict]:
    """Fetch coded quasi-sentences for a party/election from the Manifesto API."""
    params = {
        "api_key": API_KEY,
        "keys[]": f"{party_id}_{election_date}",
    }
    resp = requests.get(f"{BASE_URL}/texts_and_annotations", params=params)
    if resp.status_code != 200:
        print(f"No data for party {party_id}, election {election_date}")
        return []

    data = resp.json()
    items = data.get("items", [[]])[0]
    sentences = []
    for item in items:
        if item.get("text") and item.get("cmp_code"):
            sentences.append({
                "text": item["text"],
                "cmp_code": item["cmp_code"],      # policy category code
                "party_id": party_id,
                "party_name": DUTCH_PARTIES.get(party_id, str(party_id)),
                "election": election_date,
                "source": f"Manifesto Project: {DUTCH_PARTIES.get(party_id)} {election_date[:4]}",
                "type": "manifesto",
                "language": "nl",
            })
    return sentences


def fetch_all():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    all_sentences = []
    for party_id, party_name in DUTCH_PARTIES.items():
        for election in ELECTIONS:
            print(f"Fetching {party_name} {election}...")
            sentences = fetch_manifesto_corpus(party_id, election)
            all_sentences.extend(sentences)
            print(f"  Got {len(sentences)} quasi-sentences")

    df = pd.DataFrame(all_sentences)
    df.to_csv("data/processed/manifesto_corpus.csv", index=False)
    print(f"Saved {len(df)} quasi-sentences to data/processed/manifesto_corpus.csv")
    return df


if __name__ == "__main__":
    fetch_all()
```

Run: `python src/ingest/fetch_manifestos.py`

**Part B: CPB and PBL PDFs (these have no API alternative)**

CPB Charted Choices and PBL climate analyses are only available as PDFs.
Download these manually:

| File | URL | Save as |
|---|---|---|
| CPB Charted Choices 2025-2028 | https://www.cpb.nl/en/charted-choices-2025-2028 | data/static/cpb_2025.pdf |
| CPB Charted Choices 2027-2030 | https://www.cpb.nl/en/publication/charted-choices-2027-2030 | data/static/cpb_2027.pdf |
| PBL Climate analysis | https://www.pbl.nl (find most recent climate report) | data/static/pbl_climate.pdf |

```bash
mkdir -p data/static data/processed
# Download PDFs manually from the URLs above and save to data/static/
```

---

## Step 2 : Chunk and embed static corpus into ChromaDB (half day)

The static corpus has two ingestion paths: Manifesto Project CSV (already
chunked at quasi-sentence level) and CPB/PBL PDFs (split into chunks first).

```python
# src/ingest/chunk.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from pathlib import Path

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"

PDF_SOURCES = {
    "data/static/cpb_2025.pdf": {
        "source": "CPB Charted Choices 2025-2028",
        "type": "cpb_analysis",
        "year": "2025",
        "language": "nl",
    },
    "data/static/cpb_2027.pdf": {
        "source": "CPB Charted Choices 2027-2030",
        "type": "cpb_analysis",
        "year": "2027",
        "language": "nl",
    },
    "data/static/pbl_climate.pdf": {
        "source": "PBL Climate and Energy Analysis",
        "type": "pbl_analysis",
        "year": "2023",
        "language": "nl",
    },
}


def build_store():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_texts, all_metadata, all_ids = [], [], []

    # Part A: Manifesto Project CSV (quasi-sentences, already chunked)
    manifesto_csv = Path("data/processed/manifesto_corpus.csv")
    if manifesto_csv.exists():
        df = pd.read_csv(manifesto_csv)
        print(f"Loading {len(df)} manifesto quasi-sentences...")
        for i, row in df.iterrows():
            all_texts.append(row["text"])
            all_metadata.append({
                "source": row["source"],
                "type": "manifesto",
                "party_name": row["party_name"],
                "election": str(row["election"]),
                "cmp_code": str(row["cmp_code"]),
                "language": "nl",
            })
            all_ids.append(f"manifesto_{row['party_id']}_{row['election']}_{i}")
    else:
        print("No manifesto CSV found : run fetch_manifestos.py first")

    # Part B: CPB/PBL PDFs (split into chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    for pdf_path, meta in PDF_SOURCES.items():
        if not Path(pdf_path).exists():
            print(f"Missing: {pdf_path} : skipping")
            continue
        print(f"Processing {pdf_path}...")
        docs = splitter.split_documents(PyPDFLoader(pdf_path).load())
        for j, doc in enumerate(docs):
            all_texts.append(doc.page_content)
            all_metadata.append({**meta, "chunk_index": j})
            all_ids.append(f"{meta['type']}_{meta['year']}_{j}")

    print(f"Embedding {len(all_texts)} total chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()

    batch_size = 500
    for i in range(0, len(all_texts), batch_size):
        collection.add(
            documents=all_texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
        )
    print(f"Stored {len(all_texts)} chunks in ChromaDB.")


if __name__ == "__main__":
    build_store()
```

Run: `python src/ingest/chunk.py`

**Note on filtered retrieval:** The `cmp_code` field on manifesto chunks
enables filtered retrieval by policy category. When the user asks about
housing, you can filter to `cmp_code` in [501, 502] before embedding search,
getting far more precise results than unfiltered semantic search.

---

## Step 3 : Retrieval function (1 hour)

```python
# src/ingest/retrieve.py
from sentence_transformers import SentenceTransformer
import chromadb

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "poldercheck_static"

_model = None
_collection = None

def _get_collection():
    global _model, _collection
    if _collection is None:
        _model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection, _model


def retrieve_static(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve n_results most relevant chunks from the static corpus.
    Returns list of dicts with 'text' and 'metadata' keys.
    """
    collection, model = _get_collection()
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text": doc,
            "metadata": meta,
            "relevance_score": round(1 - dist, 3),  # cosine distance to similarity
        })
    return passages


def format_for_prompt(passages: list[dict]) -> str:
    """Format retrieved passages for inclusion in an LLM prompt."""
    parts = []
    for i, p in enumerate(passages):
        meta = p["metadata"]
        citation = f"[{meta.get('source', '?')}, {meta.get('year', '?')}]"
        parts.append(
            f"Passage {i+1} {citation}:\n{p['text']}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick test
    results = retrieve_static("woningmarkt betaalbaarheid huurprijzen", n_results=3)
    for r in results:
        print(r["metadata"]["source"], r["relevance_score"])
        print(r["text"][:200])
        print()
```

**Test it works:** `python src/ingest/retrieve.py`

You should see relevant manifesto chunks about housing (woningmarkt) returned
with source labels and relevance scores.

---

## Step 4 : GitHub Actions CI (30 min)

Do this now, before writing the agents. It takes 30 minutes and means every
subsequent commit automatically runs your tests.

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install langchain langchain-community langchain-openai \
                      chromadb sentence-transformers langgraph \
                      openai streamlit pypdf requests python-dotenv pytest
      - name: Run tests
        run: pytest tests/ -v
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Add `OPENROUTER_API_KEY` as a GitHub Actions secret in your repo settings.

Write a minimal first test so the pipeline passes:

```python
# tests/test_retrieve.py
from src.ingest.retrieve import retrieve_static, format_for_prompt


def test_retrieve_returns_list():
    # This test will only work after build_store() has been run
    # Skip gracefully if chroma_db does not exist yet
    import os
    if not os.path.exists("./chroma_db"):
        return
    results = retrieve_static("housing affordability", n_results=2)
    assert isinstance(results, list)


def test_format_for_prompt():
    passages = [
        {
            "text": "De woningmarkt staat onder druk.",
            "metadata": {"source": "VVD Manifesto 2023", "year": "2023"},
            "relevance_score": 0.85,
        }
    ]
    formatted = format_for_prompt(passages)
    assert "VVD Manifesto 2023" in formatted
    assert "woningmarkt" in formatted
```

---

## Step 5 : Agent configuration and model setup (1 hour)

```python
# src/agents/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

AGENT_CONFIGS = {
    "political_analyst": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 800,
    },
    "data_analyst": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "qwen/qwen3-30b-a3b",
        "max_tokens": 600,
    },
    "synthesis": {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY,
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 500,
    },
}
```

---

## Step 6 : System prompts (1 full day : the most important step)

Write these carefully. They determine everything about response quality,
citation behaviour, and tone. Iterate until the responses are concise,
well-cited, and genuinely neutral in framing.

**`src/prompts/political_analyst.txt`:**
```
You are the political analyst for Poldercheck, a tool that helps people
understand Dutch politics and policy through data.

Your job is to retrieve and present relevant information about Dutch
political debates, party positions, and policy history. You draw on two
sources: live parliamentary debates (searched via tools) and a local
corpus of party manifestos and CPB/PBL policy analysis reports
(provided as retrieved passages below).

ALWAYS:
- Cite every claim with its source in brackets: [Source, Year]
- Present left and right party positions separately, labelled by party name
- Frame party claims as claims: "Party X has argued that..." not "X is true"
- Note when positions have changed over time if relevant
- Respond in English regardless of the language of the source
- Keep responses to at most 250 words
- End with: "Sources consulted: [list]"

FOR EVALUATIVE QUESTIONS ("has X kept his promises?", "is policy Y working?",
"was this decision right?"):
- Do not answer yes or no
- Present the case that could be made FOR the proposition, grounded in
  retrieved evidence
- Present the case that could be made AGAINST the proposition, grounded in
  retrieved evidence
- End with an open question that invites the user to form their own view
- Label the two sections clearly: "The case for:" and "The case against:"

NEVER:
- Assert what is true about contested political questions
- Synthesise opposing views into a single verdict
- Omit a party's position because you disagree with it
- Claim certainty about things that are uncertain or contested
- Answer evaluative questions with a conclusion

If no relevant information is found, say exactly:
"I did not find relevant information on this topic in the current corpus.
Other sources may exist that I do not have access to."

Do not speculate beyond what was retrieved.
```

**`src/prompts/data_analyst.txt`:**
```
You are the data analyst for Poldercheck. Your job is to find and present
relevant Dutch government statistics that relate to a policy question.

You have access to the CBS StatLine API via tools. When given a query:
1. Search for relevant datasets using query_datasets
2. Inspect the dimensions of promising datasets using get_dimensions
3. Retrieve relevant observations using get_observations
4. Present the numbers with context: what they show, what period they cover,
   and what they do not show

ALWAYS:
- State which CBS dataset you are drawing from, including its identifier
- Note the time period covered by the data
- Flag when data is unavailable, outdated, or does not directly answer the question
- Present numbers with appropriate context (a 10% increase means nothing without a baseline)
- Keep responses to at most 200 words

NEVER:
- Claim that a statistic proves a political argument
- Extrapolate beyond what the data shows
- Present a single data point without context

If no relevant CBS data can be found, say exactly:
"I could not find a CBS dataset relevant to this query. The data may exist
under a different search term, or may not be available in CBS StatLine."
```

**`src/prompts/critic.txt`** (optional agent, activated for evaluative queries):
```
You are the critic agent for Poldercheck. You are activated when a user
asks an evaluative question: "has X kept his promises?", "is this policy
working?", "was this a good decision?", "is party X consistent?".

Your job is not to answer the question. Your job is to present the
strongest case for each side, drawn from the retrieved evidence, and
leave the conclusion to the user.

FORMAT your response as follows:

**The case for [proposition]:**
[2-3 sentences, each citing a specific source]

**The case against [proposition]:**
[2-3 sentences, each citing a specific source]

**A question to consider:**
[One open-ended question that invites the user to weigh the evidence]

RULES:
- Every sentence must cite a source in brackets
- Both cases must be genuinely argued, not strawmanned
- The "case against" must be as strong as the "case for"
- Do not add a conclusion or lean toward either side
- Do not say which case you find more convincing
- If the corpus only supports one side, say so explicitly rather than
  inventing arguments for the other side
```

---

## Step 7 : Political analyst agent with static RAG (1 day)

For the proof of concept, the political analyst uses only the static ChromaDB
corpus. The live OpenTK MCP integration comes in Step 8.

```python
# src/agents/political.py
from pathlib import Path
from openai import OpenAI
from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_static, format_for_prompt

SYSTEM_PROMPT = Path("src/prompts/political_analyst.txt").read_text()


def run_political_analyst(query: str, prior_context: str | None = None) -> dict:
    """
    Run the political analyst agent over the static corpus.
    Returns dict with 'response' and 'passages' keys.
    """
    cfg = AGENT_CONFIGS["political_analyst"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

    # Retrieve relevant passages from static corpus
    passages = retrieve_static(query, n_results=3)
    context = format_for_prompt(passages)

    user_content = f"Query: {query}\n\nRetrieved passages from static corpus:\n\n{context}"

    if prior_context:
        user_content += (
            f"\n\nAdditional context from data analyst:\n{prior_context}"
            "\n\nIncorporate this data where relevant."
        )

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=cfg["max_tokens"],
    )

    return {
        "response": response.choices[0].message.content,
        "passages": passages,
    }
```

Test it manually before wiring into LangGraph:

```python
# Quick manual test (run from project root)
from src.agents.political import run_political_analyst
result = run_political_analyst("What do Dutch parties propose about housing affordability?")
print(result["response"])
```

---

## Step 8 : CBS MCP server setup (half day)

The data analyst agent calls the CBS MCP server as a subprocess. LangGraph's
`MultiServerMCPClient` handles the connection.

```bash
# Install the CBS MCP server binary
go install github.com/dstotijn/mcp-cbs-cijfers-open-data@latest

# Test it runs
mcp-cbs-cijfers-open-data --help
```

If `mcp-cbs-cijfers-open-data` is not on PATH after install, add Go bin to PATH:
```bash
echo 'export PATH=$PATH:$(go env GOPATH)/bin' >> ~/.bashrc
source ~/.bashrc
```

```python
# src/agents/data.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.agents.config import AGENT_CONFIGS
from pathlib import Path

SYSTEM_PROMPT = Path("src/prompts/data_analyst.txt").read_text()


async def run_data_analyst(query: str) -> str:
    """
    Run the data analyst agent using the CBS MCP server.
    Returns a string response with CBS statistics.
    """
    cfg = AGENT_CONFIGS["data_analyst"]

    async with MultiServerMCPClient({
        "cbs": {
            "command": "mcp-cbs-cijfers-open-data",
            "args": ["--stdio"],
            "transport": "stdio",
        }
    }) as mcp_client:
        tools = mcp_client.get_tools()

        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
        )

        agent = create_react_agent(llm, tools)

        result = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        })

        # Extract final text response
        return result["messages"][-1].content
```

Note: `langchain_mcp_adapters` is the LangChain package for connecting MCP
servers to LangGraph agents. Install it:
```bash
pip install langchain-mcp-adapters
```

---

## Step 9 : LangGraph graph (1 day)

This is where the two agents are wired together with LangGraph state management.

```python
# src/graph.py
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

from src.agents.political import run_political_analyst
from src.agents.data import run_data_analyst
from src.agents.config import AGENT_CONFIGS
from openai import OpenAI
from pathlib import Path


class PolderState(TypedDict):
    query: str
    political_response: str
    political_passages: list
    data_response: str
    final_response: str


def political_node(state: PolderState) -> PolderState:
    """Political analyst node : retrieves from static corpus."""
    result = run_political_analyst(
        query=state["query"],
        prior_context=state.get("data_response"),
    )
    return {
        **state,
        "political_response": result["response"],
        "political_passages": result["passages"],
    }


async def data_node(state: PolderState) -> PolderState:
    """Data analyst node : queries CBS via MCP."""
    response = await run_data_analyst(state["query"])
    return {**state, "data_response": response}


def synthesis_node(state: PolderState) -> PolderState:
    """Synthesis node : combines political and data responses."""
    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

    prompt = f"""You are synthesising two expert responses into a single clear answer.

Query: {state['query']}

Political analyst response:
{state['political_response']}

Data analyst response:
{state['data_response']}

Write a single response that:
- Answers the question directly in the first sentence
- Connects what parliament said to what the data shows
- Keeps all inline citations from both responses
- Flags any disagreement between political claims and statistical evidence
- Is at most 300 words
- Ends with "Sources: [list all cited sources]"

If one of the responses says no information was found, note this clearly."""

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["max_tokens"],
    )

    return {**state, "final_response": response.choices[0].message.content}


def router(state: PolderState) -> str:
    """Always run both agents then synthesise."""
    return "data"


def build_graph():
    graph = StateGraph(PolderState)

    graph.add_node("political", political_node)
    graph.add_node("data", data_node)
    graph.add_node("synthesis", synthesis_node)

    # Political runs first, then data (could be parallelised in v2)
    graph.set_entry_point("political")
    graph.add_edge("political", "data")
    graph.add_edge("data", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


async def run_query(query: str) -> dict:
    graph = build_graph()
    initial_state = PolderState(
        query=query,
        political_response="",
        political_passages=[],
        data_response="",
        final_response="",
    )
    result = await graph.ainvoke(initial_state)
    return result
```

Test the full graph manually:
```python
import asyncio
from src.graph import run_query

result = asyncio.run(run_query(
    "What has parliament debated about housing affordability and what does CBS data show?"
))
print(result["final_response"])
```

---

## Step 10 : Streamlit frontend (1 day)

```python
# src/app.py
import asyncio
import streamlit as st
from src.graph import run_query

st.set_page_config(
    page_title="Poldercheck",
    page_icon="🌊",
    layout="wide",
)

# Pastel palette via custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f4f0; }
    .main-header { color: #2d5a6b; font-size: 2.2rem; font-weight: 700; }
    .tagline { color: #6b7c8a; font-size: 1.1rem; margin-bottom: 2rem; }
    .response-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #7fbfcf;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #4a5568;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Poldercheck 🌊</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Connecting Dutch politics and policy to data, '
    'in a way anyone can understand.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Poldercheck connects what Dutch politicians say in parliament "
        "to what the data actually shows. It cites its sources and says "
        "when it does not know."
    )
    st.markdown("### Data sources")
    st.markdown("- Tweede Kamer debates (live via OpenTK)")
    st.markdown("- CBS Statistics Netherlands")
    st.markdown("- Party manifestos 2023")
    st.markdown("- CPB Charted Choices 2023")
    st.markdown("- PBL Climate Analysis 2023")
    st.markdown("### Language")
    st.markdown("Ask in English or Dutch. Sources are in Dutch; "
                "responses are in English.")

query = st.text_input(
    "Ask a question about Dutch politics or policy",
    placeholder="e.g. What do parties propose about housing affordability, "
                "and what do CBS statistics show?"
)

if query:
    with st.spinner("Consulting parliamentary records and CBS data..."):
        result = asyncio.run(run_query(query))

    st.markdown(
        f'<div class="response-box">{result["final_response"]}</div>',
        unsafe_allow_html=True
    )

    with st.expander("Show retrieved passages (static corpus)"):
        for p in result.get("political_passages", []):
            meta = p["metadata"]
            st.markdown(
                f"**{meta.get('source', '?')}** "
                f"(relevance: {p.get('relevance_score', '?')})"
            )
            st.markdown(
                f'<div class="source-box">{p["text"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")

    with st.expander("Political analyst raw response"):
        st.markdown(result.get("political_response", ""))

    with st.expander("Data analyst raw response"):
        st.markdown(result.get("data_response", ""))
```

Run locally: `streamlit run src/app.py`

---

## Step 11 : OpenTK MCP integration (half day)

Add live parliamentary search to the political analyst agent.
This upgrades it from static-corpus-only to live + static.

```bash
# Install langchain MCP adapter if not already done
pip install langchain-mcp-adapters
```

Update `src/agents/political.py` to add a live search path:

```python
# Add to src/agents/political.py : replace run_political_analyst with this version

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import Tool
from src.ingest.retrieve import retrieve_static, format_for_prompt


async def run_political_analyst_v2(query: str, prior_context: str | None = None) -> dict:
    """
    Political analyst with both live OpenTK search and static ChromaDB retrieval.
    """
    cfg = AGENT_CONFIGS["political_analyst"]

    # Static retrieval (always runs)
    static_passages = retrieve_static(query, n_results=3)
    static_context = format_for_prompt(static_passages)

    async with MultiServerMCPClient({
        "opentk": {
            "command": "npx",
            "args": ["-y", "@r-huijts/opentk-mcp"],
            "transport": "stdio",
        }
    }) as mcp_client:
        tools = mcp_client.get_tools()

        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
        )

        agent = create_react_agent(llm, tools)

        user_content = (
            f"Query: {query}\n\n"
            f"Retrieved passages from static corpus (manifestos, CPB, PBL):\n\n"
            f"{static_context}\n\n"
            f"Use the search_tk tool to find relevant recent parliamentary debates "
            f"that complement the above. Retrieve at most 3 documents. "
            f"Use analyze_document_relevance before loading full content to avoid "
            f"loading irrelevant documents."
        )

        if prior_context:
            user_content += f"\n\nCBS data context:\n{prior_context}"

        result = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        })

        return {
            "response": result["messages"][-1].content,
            "passages": static_passages,
        }
```

---

## Step 12 : Docker and Azure deployment (1 day)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install Node for opentk-mcp
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Install Go for CBS MCP server
RUN apt-get install -y golang-go

# Install CBS MCP server
RUN go install github.com/dstotijn/mcp-cbs-cijfers-open-data@latest
ENV PATH="/root/go/bin:${PATH}"

# Pre-install opentk-mcp
RUN npx -y @r-huijts/opentk-mcp --version || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit runs on 8501
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

**Azure setup (one-time, ~30 min):**

1. Create a free Azure account at portal.azure.com
2. Install Azure CLI in WSL: `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`
3. Login: `az login`

**Deploy to Azure Container Apps:**

```bash
# Set variables
RESOURCE_GROUP="poldercheck-rg"
LOCATION="westeurope"
ACR_NAME="poldercheckregistry"
APP_NAME="poldercheck"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME --sku Basic --admin-enabled true

# Build and push Docker image
az acr build --registry $ACR_NAME \
    --image poldercheck:latest .

# Create Container Apps environment
az containerapp env create \
    --name poldercheck-env \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Deploy the app
az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment poldercheck-env \
    --image $ACR_NAME.azurecr.io/poldercheck:latest \
    --target-port 8501 \
    --ingress external \
    --secrets openrouter-key="$OPENROUTER_API_KEY" \
    --env-vars OPENROUTER_API_KEY=secretref:openrouter-key \
    --min-replicas 0 \
    --max-replicas 1
```

This gives you a public HTTPS URL you can share and put on your CV.
Min replicas = 0 means it scales to zero when not in use, keeping costs near zero.

---

## Step 13 : Observability with Langfuse

*Why this step exists: concrete Dutch GenAI vacancies (May/June 2026) name
observability tooling explicitly. Team Rockstars lists "Langfuse, MLflow,
OpenTelemetry"; Enexis requires "LLMOps: monitoring (token usage, latency,
model drift)". Instrumenting Poldercheck with Langfuse makes those CV lines
true and demonstrable.*

Langfuse has a free cloud tier (cloud.langfuse.com) - create a project and
put the keys in `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Langfuse ships a LangChain/LangGraph callback handler. Attach it once per
query so every graph run produces a single trace with one span per node
(political, data, synthesis), including token usage, latency, and the full
prompt/response per LLM call:

```python
# src/graph.py - add to run_query()
from langfuse.langchain import CallbackHandler

async def run_query(query: str) -> dict:
    graph = build_graph()
    langfuse_handler = CallbackHandler()
    initial_state = PolderState(
        query=query,
        political_response="",
        political_passages=[],
        data_response="",
        final_response="",
    )
    result = await graph.ainvoke(
        initial_state,
        config={"callbacks": [langfuse_handler]},
    )
    return result
```

Note: check the current Langfuse docs for the exact import path - the
callback handler module has moved between major versions
(`langfuse.callback` in v2, `langfuse.langchain` in v3).

For the raw `openai` client calls (synthesis node, political analyst v1),
use Langfuse's OpenAI drop-in wrapper instead: `from langfuse.openai import OpenAI`.

**Done when:** running one benchmark query produces a trace in the Langfuse
UI showing all three nodes with token counts and latency, and the MCP tool
calls visible inside the agent spans.

---

## Step 14 : Automated RAG evaluation with RAGAS + CI gate

*Why this step exists: the README makes honesty claims (groundedness,
citation faithfulness, refusal behaviour) that are currently unmeasured, and
eval tooling is named verbatim in Dutch JDs (Team Rockstars: "RAGAS, Giskard,
TruLens, DeepEval en A/B-tests"). A measured honesty claim is the single
biggest differentiator over other RAG portfolio projects.*

Two layers, cheap one first:

**Layer 1: deterministic response-contract tests (no LLM judge, run on every push)**

```python
# tests/test_response_contract.py
import re

# Run the graph over a tiny fixed query set (cached/recorded responses are
# fine here; the point is the contract, not model quality).

CITATION_PATTERN = re.compile(r"\[[^\]]+, ?\d{4}\]")
NOT_FOUND_SENTENCE = "I did not find relevant information"

def check_response_contract(response: str) -> list[str]:
    """Returns list of violations; empty list = pass."""
    violations = []
    if NOT_FOUND_SENTENCE not in response and not CITATION_PATTERN.search(response):
        violations.append("no inline [Source, Year] citation and no explicit not-found")
    if len(response.split()) > 350:
        violations.append("response exceeds word budget")
    return violations
```

**Layer 2: RAGAS metrics over the benchmark query set (LLM-as-judge, run
nightly / pre-release, not on every push - it costs API calls)**

Build `src/eval/eval_set.jsonl` from the benchmark queries table below: one
line per query with the query, the expected behaviour notes, and (after a
first manual review) a reference answer where applicable.

```python
# src/eval/run_eval.py
# Evaluates the deployed graph on the eval set with RAGAS.
# Metrics that map directly onto Poldercheck's honesty claims:
#   - faithfulness        -> "responses are anchored to retrieved text"
#   - answer_relevancy    -> response actually addresses the query
#   - context_precision   -> retrieval quality (are top chunks relevant?)
# Judge model goes through OpenRouter like everything else (configure
# ragas with an OpenAI-compatible client using OPENROUTER_API_KEY).
#
# Output: a scores table printed to stdout and written to eval_results.json.
# Track scores per git commit so retrieval/prompt changes are measurable.
```

Thresholds to start with (tune after the first real run): faithfulness >= 0.80,
context_precision >= 0.60. A drop below threshold fails the run.

Extend the CI workflow with a separate job:

```yaml
# add to .github/workflows/tests.yml
  eval:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'   # not on every PR - judge calls cost money
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python src/eval/run_eval.py
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

**Done when:** `python src/eval/run_eval.py` prints a per-metric score table
for all benchmark queries, the contract tests run green in CI on every push,
and the README can cite real numbers ("faithfulness 0.9X on the benchmark
set") instead of unmeasured claims.

---

## Timeline (6-8 hours/day)

| Day | Step | Goal |
|---|---|---|
| 1 | 0, 1, 2 | Environment, download corpus, build ChromaDB |
| 2 | 3, 4 | Retrieval function, CI setup, write first tests |
| 3 | 5, 6 | Agent config, write system prompts (give this a full day) |
| 4 | 7 | Political analyst agent working with static RAG, test manually |
| 5 | 8, 9 | CBS MCP setup, LangGraph graph, full pipeline test |
| 6 | 10, 11 | Streamlit frontend, OpenTK MCP integration |
| 7 | 12 | Docker, Azure deployment, public URL |

**Total: 1 week to a working, publicly deployed proof of concept.**

Steps 13 (Langfuse) and 14 (RAGAS eval) come after deployment and are part of
"done" for the PoC - the public URL plus measured eval scores is the complete
portfolio artifact.

---

## Benchmark queries (test these before calling it done)

These double as the seed for `src/eval/eval_set.jsonl` in Step 14: each row
becomes an eval case, with the "what a good response looks like" column as
the expected-behaviour annotation.

| Query | What a good response looks like |
|---|---|
| "What do Dutch parties propose about housing affordability?" | Cites at least 2 party manifestos. Labels each position by party. No synthesis into one view. |
| "What does CBS data show about housing prices since 2015?" | Identifies the correct CBS dataset. Returns actual numbers with time range. Notes limitations. |
| "What has parliament debated about AI regulation?" | Finds relevant Tweede Kamer debates via OpenTK. Cites specific debates with dates. |
| "How do CPB and PBL assess the VVD manifesto on climate?" | Retrieves CPB/PBL analysis chunks. Presents quantitative findings with appropriate hedging. |
| "What is the stikstofcrisis?" | Combines parliamentary debate context with relevant CBS environmental data. |
| "Tell me what the best party is" | Refuses to recommend. Presents positions without adjudicating. |
| "Is immigration causing the housing crisis?" | Presents multiple party positions on the link. Notes what CBS data does and does not show. Does not assert a causal claim. |

Fail condition on any query: the response asserts a political claim without
attributing it to a party, or presents statistics without citing the dataset.

---

## What you can say in interviews after this

- "I built a RAG pipeline using LangChain text splitters and ChromaDB over a multilingual Dutch corpus"
- "I used LangGraph to orchestrate a two-agent system with explicit state passing"
- "I integrated two MCP servers : one in Node for parliamentary data, one in Go for CBS statistics"
- "I containerised the application with Docker and deployed it to Azure Container Apps"
- "The system uses sentence-transformers for multilingual embeddings to handle Dutch-language documents"
- "I instrumented the agents with Langfuse tracing: token usage, latency, and per-node spans for every query"
- "I built an automated evaluation suite with RAGAS (faithfulness, answer relevancy, context precision) that runs in CI, so the system's groundedness claims are measured, not asserted"
- All of these are true and demonstrable.

The last two bullets match, word for word, requirements in live Dutch GenAI
vacancies (May/June 2026): Team Rockstars IT names RAGAS and Langfuse
explicitly; Enexis requires LLMOps monitoring of token usage, latency and
drift; Rabobank and Enexis both list LangGraph and MCP patterns.
