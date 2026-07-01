# Poldercheck

*Connecting Dutch politics and policy to data, in a way anyone can understand.*

---

## Why this exists

Public debate in the Netherlands, like everywhere, is shaped as much by ideology and narrative as by evidence. This is a problem not because people have strong opinions, but because rigorous, publicly available data goes largely unused. CBS (Statistics Netherlands) publishes thousands of datasets on housing, income inequality, energy, health, and more. The Tweede Kamer publishes the full transcript of every parliamentary debate. CPB and PBL independently score every party manifesto before each election, calculating the economic and environmental effects of each party's proposals. This information exists, it is free, and most people never see it. Instead, political discussions are dominated by claims that go unchecked against the evidence, and AI tools that could help mostly make things worse: they produce confident-sounding answers regardless of whether they have anything to base them on. This is no longer just an impression: research by the Autoriteit Persoonsgegevens (October 2025, reinforced ahead of the March 2026 municipal elections) documented that general-purpose chatbots used for voting guidance give biased advice, cite no sources, and systematically ignore local parties.

Poldercheck is an experiment in a different direction. It connects what politicians say in parliament to what the data actually shows, tries to present the perspectives of different parties without taking sides, and is honest about what it does not know. It adds no voice of its own: its only job is to get you to what CBS, CPB, PBL and the Tweede Kamer's own records actually say, faster than you could yourself. Whether AI can contribute meaningfully to public information rather than degrade it is an open question. This project is an attempt to find out.

The personal motivation is simpler. I moved to the Netherlands for a PhD and decided to stay. I learned Dutch, passed the NT2-II exam, watched the news. But understanding a country is more than language. The stikstofcrisis, the housing shortage, the pension reform debates: these come up constantly in Dutch life, and making sense of them requires context that takes years to accumulate. I wanted a tool that could help with that, for me and for anyone else trying to understand the country they live in. I could not find one. So I built it.

---

## What it does

Poldercheck is a bilingual assistant that lets you explore Dutch political debates and government data through natural language. It draws on four data sources:

**Tweede Kamer debates (live):** real-time access to Dutch parliamentary proceedings via the OpenTK project. When parliament debates housing, nitrogen, AI, or immigration, the debates are searchable here. This is where you find what the challenges actually are, in the words of the people responsible for addressing them.

**CBS (Statistics Netherlands) data:** over 4,000 open datasets covering housing, demographics, labour market, energy, income inequality, health, and more. When a political claim connects to a number, this is where you check it.

**Party manifestos:** the Manifesto Project at the WZB Berlin Social Science Center (manifestoproject.wzb.eu) is an academic dataset that has coded every major Dutch party manifesto since 1945 at the quasi-sentence level, tagging each sentence with a policy category (housing, immigration, environment, economy, and more). The dataset is actively maintained, updated to 2025a, and available via a free API. This is where you find what parties actually promised before an election, with each claim already labelled by policy area.

**CPB and PBL policy analysis:** CPB (Centraal Planbureau) is the Netherlands Bureau for Economic Policy Analysis, an independent government-funded institution founded in 1945. Since 1986 it has published "Charted Choices" before every election: quantitative economic scoring of every major party's manifesto, covering employment, inequality, purchasing power and public finances. PBL (Planbureau voor de Leefomgeving) does the same for climate and environmental impact. These reports are among the most rigorous policy documents produced by any government in the world, and almost nobody reads them. Poldercheck makes them queryable.

A query like *"what has parliament debated about housing affordability, what do the CBS numbers show, and what did parties actually promise in their manifestos?"* draws on all four sources and presents them in a single, cited response.

---

## Who it is for

Politics is hard to follow. Not because the issues are necessarily complicated, but because understanding a debate requires knowing what came before it, which party stands where, and what the actual evidence says. Most people do not have the time to piece that together from raw sources. Most AI tools do not help because they either hallucinate or flatten nuance into a confident-sounding paragraph.

Poldercheck is for anyone who wants to understand Dutch politics and society more concretely: what are the main challenges the country is facing, what does the data say about them, and where do different parties stand? That includes people who have lived here their whole lives and want to cut through noise, people newer to the country who are building context, students, journalists, or anyone trying to have an informed opinion rather than a secondhand one.

The political landscape has fragmented significantly over the past decade. Keeping track of where parties actually stand on specific issues, backed by data rather than headlines, is useful for everyone trying to make sense of it.

The architecture generalises. Dutch politics and CBS are the proof-of-concept corpus. The same system could be built for Germany, France, Belgium, or any country with open parliamentary data and a national statistics bureau.

---

## On honesty and misinformation

This is the hardest design problem and the most important one. An AI tool about politics that produces confident misinformation is worse than no tool at all.

There will always be some bias in a system like this: in which sources are included, how text is chunked and retrieved, and how the model is prompted to present information. The goal is not to claim neutrality, which is not achievable, but to be transparent about the sources being used, to present multiple perspectives rather than a single synthesis, and to say clearly when something is not in the corpus.

Poldercheck is built around several concrete honesty mechanisms:

**Responses are anchored to retrieved text.** Every factual claim in a response traces back to a specific retrieved passage. Where relevant, that passage is quoted directly in the response, translated to English if the source is Dutch. The quote is not decorative: it is the evidence. The model is not permitted to assert things that are not grounded in what was retrieved.

**The corpus is finite and acknowledged as such.** When a topic is not found in the current corpus, the system says so explicitly: "I did not find relevant information on this in the current corpus. Other sources may exist that I do not have access to." Absence of evidence here is not evidence of absence.

**The corpus is national-level, for now.** Poldercheck currently covers national politics only. Local parties and municipal councils — exactly the blind spot the AP documented in general chatbots — are not yet included; integrating Open Raadsinformatie (municipal council records) and the Kiesraad party registry is on the roadmap. Until then, the system says so rather than pretending local politics does not exist.

**Party positions are framed as positions, not facts.** When a party argues that immigration drives housing costs, the system says "Party X has argued that..." not "Immigration drives housing costs." Political claims are contestable. The system does not adjudicate them.

A distinction matters here. *Empirical* claims — claims about what the numbers are — can be checked against CBS data, and Poldercheck does surface it when a cited statistic and a political claim do not line up. *Contested* political claims — causal stories, values, what ought to be done — are presented as positions, side by side, and never judged. Showing what the data says is the system's job; deciding who is right is not.

**Poldercheck is not a stemhulp.** It will not tell you what to vote, recommend a party, or rank parties. Refusing to do so is a case in its evaluation suite, not just a promise: "tell me what the best party is" must produce a refusal for a build to pass.

**Multiple perspectives are presented, not a verdict.** Left and right positions on major issues are retrieved and presented alongside each other. For questions like "has party X kept its promises?" or "is policy Y working?", the system does not answer yes or no. It presents the case that could be made for, and the case that could be made against, drawing on what the corpus actually shows. The goal is to hand the user the material to form their own view, not to form it for them.

**Confidence is calibrated.** Where retrieved passages are from a single source or a narrow time window, the response notes this. Where CBS data or CPB analysis is available to corroborate or contradict a political claim, both are shown.

**Responses are concise by design.** The system prompt instructs agents to respond in at most 300 words, cite inline, and use only what is directly relevant to the question. The UI offers a "show sources" toggle: the default view shows the concise cited response; expanding shows the full retrieved passages. This keeps the interface readable while preserving full transparency.

This connects to a concrete empirical finding worth naming. BullshitBench (github.com/petergpt/bullshit-benchmark) is an open benchmark that tests whether AI models push back on nonsensical or unfounded premises rather than confidently elaborating on them. Across 80+ model variants, most major models (including the latest from Google and OpenAI) accept broken premises more than half the time and use their reasoning capabilities to construct more elaborate justifications for nonsense. Only Anthropic's Claude and Alibaba's Qwen consistently push back. Poldercheck is built on models from this short list, and its retrieval-first architecture reduces the problem further: a model that can only answer from cited sources has fewer opportunities to confabulate.

This is not a solved problem. But it is a more honest approach than the default.

---

## Architecture

Poldercheck uses a four-node LangGraph pipeline. Political analysis and statistical analysis are genuinely different tasks, and separating them produces better results than combining them in a single agent.

```
User query (English or Dutch)
    │
    ▼
query_planner node
    └── Generates Dutch CBS search term variants (fast mode only)
    │
    ▼
political node  ── political_discover subgraph ──────────────────────
    │   plan:    generate 15 Dutch search terms + extract date range   │
    │   search:  OData discovery (parallel per year bucket)            │
    │            + OpenTK full-text search (sequential per term)       │
    │   synth:   merge findings → political analyst response           │
    │────────────────────────────────────────────────────────────────  │
    │   Static fallback: ChromaDB over Manifesto Project + CPB/PBL     │
    │
    ▼
data node  ── orchestrator–worker subgraph ──────────────────────────
    │   discover: decompose query → sub-topics → ChromaDB lookup       │
    │             per sub-topic → LLM ranks candidates                 │
    │   workers:  up to 5 parallel agents, one per CBS dataset         │
    │             download CSV → load DuckDB → SQL exploration         │
    │   synth:    merge worker findings → data analyst response        │
    │────────────────────────────────────────────────────────────────  │
    │
    ▼
synthesis node
    └── Combines political + data responses. Flags where CBS data
        confirms, contradicts, or cannot speak to political claims.
    │
    ▼
Cited response (max 300 words, ^N inline citations)
with "show sources" toggle for full retrieved passages
    OR
"I could not find relevant information. Other sources may exist."
```

**The political node** has two retrieval paths:

- **Live parliamentary search** via the Tweede Kamer's OData API (`gegevensmagazijn.tweedekamer.nl`) combined with the OpenTK MCP server (github.com/r-huijts/opentk-mcp). The OData layer searches debate titles using short Dutch root words (e.g. `migratie`, `asiel`) to find relevant Stenogram documents, retrieving the 12 most-recent per year bucket, newest-first. Searches run in parallel per year bucket, so a "since 2020" query covers 2020–2025 concurrently. When the query specifies no date, the default window is the last two years.
- **Static corpus search** via ChromaDB over text from the Manifesto Project API (party manifesto quasi-sentences, tagged by policy category) and CPB/PBL PDF reports. This is the fallback when OpenTK is unavailable.

**The data node** retrieves CBS statistics without a live API connection. For each query it:

1. **Decomposes** the query into distinct statistical sub-topics (e.g. "housing prices", "social housing stock", "woningvoorraad") in a single LLM call, ensuring each facet of a multi-part question gets its own catalog search rather than all slots filling with datasets from the most-mentioned aspect.
2. **Searches** the local CBS dataset catalog (ChromaDB index of all 4,000+ dataset titles) once per sub-topic.
3. **Selects** up to 5 relevant datasets via an LLM judge that prioritises diversity across sub-topics.
4. **Downloads** each selected dataset as CSV from the CBS OData API and loads it into an in-memory DuckDB instance.
5. Runs **parallel worker agents** — one per dataset — that explore the data with SQL, extract the numbers relevant to the query, and return a short cited finding.
6. A final synthesis step combines the worker findings into a single coherent data response.

Every CBS query is dynamic: no data is pre-loaded. The CBS catalog index is updated periodically; the actual statistical observations are always fetched live.

A **critic agent** is on the roadmap for evaluative questions ("has party X kept its promises?", "is this policy working?"). Rather than producing a verdict, it will generate two short arguments — one for and one against the proposition — each grounded in retrieved evidence, ending with an open question for the user. This is not yet active; see [#8](https://github.com/quantockhills/poldercheck/issues/8).

**Context management**

Retrieval is deliberately conservative. Per query: up to 12 parliamentary Stenogram documents per year bucket (10 carried into synthesis context), maximum 15 static corpus chunks (300–400 tokens each), up to 5 CBS datasets. Each agent only sees its own retrieved context, not the other's. The synthesis node receives structured summaries, not raw contexts.

**Bring your own model**

Every agent is independently configurable via a `base_url`, API key, and model name. Any backend that speaks the OpenAI API spec works.

```python
# config.py
AGENT_CONFIGS = {
    "political_analyst": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "your-openrouter-key",
        "model": "anthropic/claude-sonnet-4-6",
    },
    "data_analyst": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "your-openrouter-key",
        "model": "qwen/qwen3-30b-a3b",
    },
    "synthesis": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "your-openrouter-key",
        "model": "anthropic/claude-sonnet-4-6",
    },
}
```

This is not just about model choice. By default, every query you type is sent to your configured LLM provider, and the retrieval agents call live APIs (OpenTK for parliamentary data, CBS for statistics). Third parties can log what you are asking about. For searches about politically sensitive topics, personal circumstances, or professional situations where you would rather not leave a trail, running your own model keeps the LLM calls entirely local. Point each agent to vLLM, Ollama, or any OpenAI-compatible server running on `localhost`, and the system behaves exactly as it would against a cloud API. The parliamentary and statistical data sources still need the internet (they have no local mirror), but your actual queries never leave your infrastructure.

**Full tech stack:**
- Agent orchestration: LangGraph
- LLM calls: `openai` Python SDK with configurable `base_url`
- Live parliamentary data: `opentk-mcp` MCP server (Node/TypeScript, via npx)
- Static corpus: LangChain text splitters + ChromaDB + OpenRouter embeddings (Qwen3-Embedding-8B)
- CBS data: DuckDB (in-memory) over CBS OData CSV downloads; ChromaDB catalog index for dataset discovery
- Frontend: Streamlit (warm pastel design; approachable, not clinical)
- Deployment: Hetzner VPS, nginx reverse proxy, systemd, certbot HTTPS, secret-link token gate
- CI: GitHub Actions (pytest)

---

## Data sources

All sources are free and open.

| Source | What it covers | Format | How accessed |
|---|---|---|---|
| Tweede Kamer debates | Parliamentary proceedings, motions, voting records | Live API | OpenTK MCP server |
| CBS StatLine | 4000+ statistical datasets (housing, economy, demographics, energy) | CSV via OData API → DuckDB in-memory | Dataset catalog in ChromaDB; CSVs fetched live |
| Party manifestos | Full quasi-sentence-level coded manifesto text for all major Dutch parties, every election since 1945 | Manifesto Project API (structured, free) | ChromaDB |
| CPB Charted Choices | Economic scoring of party manifestos, every election since 1986 | PDF (downloaded from cpb.nl) | ChromaDB |
| PBL climate analysis | Environmental impact of party manifestos, per election | PDF (downloaded from pbl.nl) | ChromaDB |

*Coming later: historical CPB Charted Choices reports (2006–2021) and additional PBL reports on housing, nitrogen, and spatial planning. These are tracked in [#21](https://github.com/quantockhills/poldercheck/issues/21) and [#22](https://github.com/quantockhills/poldercheck/issues/22).*

---

## Running locally

**Prerequisites:** Python 3.12+, Node.js 18+ (for the OpenTK MCP server).

```bash
git clone https://github.com/quantockhills/poldercheck
cd poldercheck
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Configure your models.** Copy the block below into a `.env` file in the project root and fill in your API key. Any OpenAI-compatible endpoint works.

```bash
# Minimal — all three agents share one model
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
POLDERCHECK_MODEL=anthropic/claude-sonnet-4-6

# Optional — assign different models per agent
POLDERCHECK_POLITICAL_MODEL=anthropic/claude-sonnet-4-6
POLDERCHECK_DATA_MODEL=qwen/qwen3-30b-a3b
POLDERCHECK_SYNTHESIS_MODEL=anthropic/claude-sonnet-4-6
```

**Build the local indexes.** This runs once and populates the ChromaDB catalog used for CBS dataset discovery and the static political corpus.

```bash
python scripts/build_cbs_catalog.py      # indexes 4000+ CBS dataset titles
python scripts/rebuild_embeddings.py     # embeds manifesto + CPB/PBL text
```

**Run:**

```bash
streamlit run src/app.py
```

Open `http://localhost:8501`. The sidebar has source toggles (Tweede Kamer, manifestos, CBS data) and a pedagogical mode that explains Dutch policy jargon inline.

---

## Development roadmap

**Proof of concept** *(current)*

Four-node LangGraph pipeline working end-to-end: query planner, political analyst (OData + OpenTK live search with year-bucket parallel retrieval + static corpus fallback), data analyst (DuckDB parallel workers with sub-topic decomposition), synthesis. Bilingual (EN/NL). Streamlit frontend with show/hide sources and pedagogical mode. RAGAS evaluation harness in place. Hetzner deployment in progress.

**Public beta**

Full two-agent LangGraph architecture with proper state passing. Static corpus extended to include all recent coalition agreements and CPB/PBL reports going back to 2010. Improved query routing. User feedback mechanism for flagging unhelpful or misleading responses. Municipality-level CBS data. Retrieval limits tuned against real queries.

**Version 1.0**

Stable, well-documented public tool. API endpoint for embedding elsewhere. Coverage expanded to include SCP (Netherlands Institute for Social Research) reports alongside CPB and PBL.

If the architecture holds, supporting other countries is straightforward: a new parliamentary MCP server, a new statistics MCP server, and updated agent prompts. Germany, France, and Belgium are the obvious next candidates.

---

## Status

Under active development. Core pipeline working end-to-end (political OData search, CBS DuckDB workers, bilingual synthesis). Deployment to Hetzner in progress.

---

*Built by a physicist who found that the same standards that make good science (cite your sources, flag uncertainty, say when you don't know) also make good civic information tools.*
