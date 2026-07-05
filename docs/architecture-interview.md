# Poldercheck: How It Works

Imagine you ask: *"What have Dutch parties said about housing affordability since 2020?"*

To answer this properly, Poldercheck needs to do three things at once: check what parliament has debated, look up the relevant statistics, and consult what the parties promised in their manifestos. These are fundamentally different kinds of questions — reading debate transcripts is not the same as querying a statistical database, and neither is the same as searching through party platforms. So instead of one big "ask an AI" box, Poldercheck splits the work across **two specialised agents that search independently, then merges their findings.**

---

## The Flow, Step by Step

Think of it like a newsroom with two reporters and an editor:

### Step 1: The Parliamentary Reporter (Political Agent)

This agent's job is to find what was actually said in parliament. It doesn't search a general search engine — it goes directly to the Tweede Kamer's own document database.

**First, it plans.** One AI call thinks about the question and generates 15 different Dutch search terms. Why 15? Because political language varies — a debate about "housing affordability" might use terms like *woningtekort* (housing shortage), *huurprijzen* (rental prices), *hypotheekrente* (mortgage rates), or *volkshuisvesting* (public housing). It also extracts the time period from your question and breaks it into year-by-year chunks so it can search in parallel.

**Then, it searches.** For each year, it queries the parliamentary database directly, downloading the full text of relevant debates — not just summaries, but the actual transcripts. This is where it finds 30-60 potentially relevant debates.

**Then, it ranks them — in two passes.** Dutch is a tricky language for search because of compound words: a debate titled *woningbouwvereniging* (housing association) should still match a search for *woning* (home) or *bouw* (construction). So it first uses a clever text-matching trick that splits compounds, narrowing 60 debates down to the 40 best-looking chunks. Then one AI call reads those 40 chunks and scores each one — essentially saying "this debate is an 8/10 for relevance, this one's a 3/10." The top 15 survive.

**Finally, it extracts party positions.** From those top debates, it finds every mention of party names — VVD, GroenLinks, PVV, etc. — and pulls out the surrounding paragraphs. This means the final output can say "the VVD argued X, while the PvdA argued Y," backed by actual parliamentary transcript.

**Then it writes its report.** One final AI call reads all these excerpts — the relevant debate passages plus matching sections from party manifestos — and produces a concise summary with citations.

### Step 2: The Data Analyst (CBS Agent)

While the political reporter works, a second agent handles the numbers. If your question needs statistics (and housing affordability definitely does), it needs to find the right CBS dataset and write the right SQL query — not something you'd expect a general chatbot to do.

**First, it discovers.** It breaks your question into sub-questions, like "what happened to house prices?", "what about rental costs?", "how many homes were built?" For each sub-question, it searches a catalogue of 1,300 Dutch statistical datasets — not by keyword matching, but by semantic meaning, using a vector database that understands that "affordable housing" relates to datasets about *betaalbaarheid woningmarkt* rather than just matching the word "housing."

**Then, it fans out.** It picks the 5 most relevant datasets and spawns 5 parallel workers — each one downloads the actual CBS data (a CSV file), loads it into an in-memory database, and explores it by writing SQL queries. These workers are **mini AI agents**: they think, run a query, look at the results, think again, run another query, until they've found something meaningful. They don't need to be told which columns or tables exist — they figure that out by inspecting the data.

**Then it synthesises.** All 5 workers return their findings, and one AI call weaves them into a coherent statistical narrative with proper citations to specific CBS datasets.

### Step 3: The Editor (Synthesis Node)

Now we have two reports: one from the parliamentary reporter (what politicians said) and one from the data analyst (what the numbers show). The editor's job is to merge them into a single, readable answer — no more than 300 words, every claim cited, never taking sides.

This is where the system enforces its **honesty rules**: every factual statement must have a `^1` or `^2` pointing to a specific source. If the system can't find relevant information, it must say so explicitly rather than inventing an answer. It must attribute causal claims to the politicians who made them rather than presenting them as facts. And it must never, ever tell you which party to vote for — it presents positions neutrally and lets the reader decide.

---

## Key Design Choices (The "Why" Behind the Architecture)

### Why separate agents?

A single AI trying to simultaneously search debates, run SQL queries, and consult manifestos gets confused. Its context window fills with mixed information, and the prompts needed for political nuance ("VVD argued X, while noting Y") clash with those needed for statistical precision ("the youth unemployment rate fell from 10.6% to 7.6% between Q1 2020 and Q3 2022"). By keeping them separate, each agent can use a different AI model, different instructions, and different tools — and they never pollute each other's context.

### Why DuckDB?

CBS datasets are CSV files updated quarterly. Downloading them fresh each time means the data is always current — no stale database to maintain. DuckDB is an analytical database that runs entirely in memory: no server, no setup, no disk. Think of it as "SQLite designed for statistics." It reads CSV files directly and executes complex analytical queries instantly. For a system that needs to explore unfamiliar datasets on the fly, it's the perfect fit.

### Why two-stage ranking?

Searching through years of parliamentary transcripts is expensive if you use AI for every step. BM25 — a classic text-matching algorithm from the 1990s — is extremely fast and handles Dutch compound words surprisingly well with a small custom tweak. It narrows the field from potentially hundreds of debates to 40 candidates. Then a single AI call ranks those 40, costing a fraction of what it would take to have AI read everything from scratch.

### Why vector search + catalogue?

The CBS has 1,300 datasets with names like "Arbeidsdeelname; kerncijfers" (Labour participation; key figures). A keyword search for "youth unemployment" would miss "Arbeidsdeelname." A vector database — which stores text as numerical representations of meaning, not words — catches these semantic relationships. The same search that finds "youth unemployment" also finds "jeugdwerkloosheid" and "labour participation by age group."

### What happens when things break?

Live parliamentary data goes down, APIs time out, debate PDFs fail to download. Poldercheck is designed to degrade gracefully: if the Tweede Kamer search fails, it falls back to searching only the party manifestos and tells you honestly what was and wasn't searched. The system never silently drops a source — it's explicit about coverage.

---

## Technologies in Plain Language

| Technology | What it does | Why it matters |
|---|---|---|
| **Vector database (ChromaDB)** | Stores text as numerical "meaning vectors"; finds conceptually similar content even with different wording | Bridges the gap between your English question and Dutch dataset names |
| **DuckDB** | An in-memory analytical database that reads CSV files directly | Lets the system explore fresh CBS data without maintaining a data warehouse |
| **BM25** | A fast text-matching algorithm that understands word frequency | Narrows thousands of debate pages to a shortlist cheaply, before involving AI |
| **OData API** | A standard way to query government databases over the web | Direct, authoritative access to parliamentary records — no third-party search engine |
| **LangGraph** | A framework for wiring AI calls and tools into a flowchart-like pipeline | Makes the multi-step process reliable: each step completes before the next begins, and failures are handled gracefully |
| **Langfuse** | An LLM observability platform — traces every AI call, tool invocation, and node automatically | Turns a black-box pipeline into something you can debug: see exactly what each agent retrieved, what it thought, and where it was slow |

---

## Debugging a Complex AI System

When you're building something like Poldercheck, you can't just `print()` debug. The AI outputs are probabilistic — the same query can produce different results. Latency varies. Costs add up. And when something goes wrong — a hallucinated statistic, a missed debate, a 5-minute query that should've taken 2 — you need to see inside the black box.

**Langfuse** provides that visibility. It hooks into LangChain's callback system with a single line of configuration — no code changes inside the agents themselves. Every LLM call, every tool invocation (downloading a CBS dataset, running a SQL query, fetching a debate transcript), and every node transition gets logged as a timestamped trace. You open the Langfuse dashboard and see a visual timeline:

```
[query_planner] ── 2.1s ──► [political agent] ── 97.6s ──► [data agent] ── 145s ──► [synthesis] ── 6.9s
                                    │                         │
                              plan: 8.2s              discover: 12.3s
                              search: 22.1s           worker 1: 8.4s
                              synthesize: 67.3s       worker 2: 11.2s
                                                      worker 3: 6.1s
                                                      ...
```

Each step expandable: you can see the exact prompt sent to the LLM, the full response, the token count, the cost, and — crucially — the retrieved evidence that the agent used. When a RAGAS evaluation says the `context_precision` score is `0.087`, you can trace backwards to see that 95 of the 101 tool-call contexts were intermediate SQL exploration queries, not the final results that informed the answer.

It's opt-in: if the environment variable isn't set, nothing happens — no overhead, no dependency. But when it's on, it turns debugging from guesswork into inspection.

### What is observability?

"Observability" sounds like jargon, but the idea is simple: if you can't see inside a system, you can't fix it. In traditional software, you'd use logs and error messages. But with AI systems, the problems are harder to spot — a wrong answer can look perfectly fine at first glance. Observability means instrumenting your system so that when something goes wrong, you can trace backwards from the symptom to find *exactly* which step failed. It's the difference between "the answer is wrong, no idea why" and "the data agent queried the wrong CBS dataset because the catalogue search returned irrelevant results."

### Langfuse: Why it matters

Poldercheck runs a complex pipeline — two parallel agents, each making multiple AI calls, downloading data, running SQL, merging results. When something goes wrong (a weird answer, a slow response, a hallucinated statistic), you need to see *exactly* what happened inside.

**What it does:** Every single AI call, every tool invocation, every database query gets logged as a timestamped "trace" — think of it like a flight recorder for your AI system. You open the Langfuse dashboard and see a visual timeline of your entire pipeline: which node took how long, what the AI was thinking at each step, what data it retrieved, and what it ultimately said.

**Why it matters:**

1. **LLM observability is a real engineering concern.** When you're building with AI, you can't just `print()` debug — the outputs are probabilistic, latency varies, and costs add up. Langfuse gives you production-grade visibility: latency per node, token usage, cost per query, and full prompt/response logging. You can answer questions like "why did this query take 4 minutes instead of 2?" or "why did the synthesis node produce a hallucinated number?"

2. **It's opt-in and zero-cost to disable.** The integration is a one-liner — LangChain's callback system means you just pass a `CallbackHandler()` and every LLM call, tool call, and node transition gets traced automatically. No code changes needed inside the agents. If you don't set the environment variable, nothing happens — no overhead, no dependency.

3. **It enables systematic evaluation.** When you're tuning RAGAS scores (faithfulness, context precision, etc.), you need to spot-check individual verdicts — "did the judge correctly mark this claim as unfaithful?" Langfuse lets you see the exact context that was retrieved, the exact LLM response, and the judge's reasoning, all linked in one trace. Without it, debugging an eval run is guesswork.

**The one-sentence version for an interview:** "Langfuse is the observability layer — it traces every LLM call, tool invocation, and node in the pipeline so I can debug slow queries, spot hallucinations, and verify that the RAGAS evaluation scores are actually fair."

---

## The Core Philosophy

Poldercheck isn't trying to be smarter than the user. It's trying to be **more diligent**: searching the actual sources, citing them, separating evidence from interpretation, and being honest about what it couldn't find. The complexity exists in service of that diligence — not to show off technology, but to produce answers you can actually verify.
