# Parliamentary retrieval: design and rationale

How Poldercheck decides which Tweede Kamer debates reach the synthesis LLM, and why it works the way it does. The design came out of an empirical investigation (July 2026) triggered by a real failure: asked about women's representation in industry and government, the system answered "politicians have not discussed this," while a May 2024 plenary debate containing a motion to extend the vrouwenquotum sat in the corpus, fetched but discarded by the ranking layer.

## The pipeline

```
User question (EN or NL)
    |
    v
Plan node: LLM generates 15 Dutch search terms + 3-5 title root words
    |
    v
Title search (OData): contains() match on debate titles, per year bucket
    - five transcript types: Stenogram, Verslag van een commissiedebat,
      Verslag van een algemeen overleg (pre-2021 name), Verslag van een
      wetgevingsoverleg, Verslag van een notaoverleg
    - up to 30 docs per year, buckets fetched in parallel, coverage 2018+
    |
    v
Full-text download per candidate (PDF/DOCX extraction)
    |
    v
BM25 champion selection (free, milliseconds)
    - documents split into 1500-char chunks
    - all chunks scored against the search-term tokens on one scale
    - each debate's best chunk becomes its "champion"
    - decompound-lite matching: a document token also matches any query
      token it starts or ends with, so 'vrouwenquotum' matches 'quotum'
    |
    v
LLM triage (one call, ~1 cent, seconds)
    - top 40 candidates by BM25, each represented by 900 chars of its champion
    - scored 0-10 against the question; keep score >= 3, max 15 debates
    - fallback: if the call fails, rank by BM25 champion score alone
    |
    v
Synthesis: receives each kept debate's champion passage + locally
extracted party excerpts
```

## Why this design: what was tested

All experiments used the same real failure case: 18 candidate stenograms for the women's representation question, with the known-correct target being the 2024-05-21 Emancipatie debate (vrouwenquotum motions).

| Method | Target's rank | Verdict |
|---|---|---|
| Original scorer (OpenTK MCP `analyze_document_relevance`) | dropped | binary 40/0 scores; selection was list-order lottery |
| FlashRank ms-marco-MultiBERT-L-12 (2021, 170M, local CPU) | 14 of 18 | scores saturated at 0.99; ranked on passage length; cross-lingual pairs scored near zero |
| Embeddings (Qwen3-Embedding-8B via API, chunk max-pooling) | 1 | clean gradient, right passage for the right reason |
| BM25 champion chunks (free, local) | 1 | clear margin; champion was the quota motion itself |
| gte-multilingual-reranker-base (2024, 306M, local CPU) | 8 (EN query) / 1 (NL query) | better than 2021 model but mushy mid-ranking; degrades cross-lingually |
| BM25 champions + LLM triage (implemented) | 1, score 90/100 | triage also zeroed all topically-adjacent-but-irrelevant debates |

A second question (nuclear power positions since 2020) validated the ranking on fresh data and exposed the discovery gap: with the old Stenogram-only filter, 7 candidates; with all five transcript types, 17 candidates, and the new number one was a Kernenergie committee debate containing an MP describing exactly how party positions had shifted, a document type that was previously invisible.

## Division of labor

- The plan node's LLM term generation acts as query expansion: it bridges the
  vocabulary gap (English question to Dutch parliamentary jargon) before any
  matching happens. BM25's strong results depend on it.
- BM25 finds where the query words concentrate. It is free and precise but
  blind to meaning: it cannot tell a femicide debate from a boardroom-quota
  debate when both are saturated with the word "vrouwen".
- The triage is the semantic filter. It reads each debate's champion passage
  against the actual question and drops word-rich but substance-poor
  candidates. On the test case it zeroed every violence/safety/question-hour
  document that BM25 had scored 5-10.

## Rejected alternatives

- **Pre-indexed embeddings (ChromaDB over all transcripts)**: best ranking
  quality in tests, but 16k+ documents means gigabytes of vectors and a
  RAM-resident index; the production server has 4 GB. Viable later as
  cache-as-you-go (embed candidates on first contact, store vectors on disk,
  compare in plain Python).
- **Local neural rerankers**: the 2021-era model failed outright; the 2024
  306M encoder was mediocre on this corpus and degrades on cross-lingual
  input. Small LLM-based rerankers (Qwen3-Reranker-0.6B) remain untested and
  plausible, but weights plus runtime strain a 4 GB server.
- **Inject everything into synthesis**: works at 18 candidates, breaks at
  17+ committee debates of 200k chars each; also degrades citation quality
  (needle-in-haystack attention).

## Known limitations and open items

- Discovery is title-only: a debate discussing the topic under an unrelated
  title is invisible. The OpenTK full-text channel partially compensates.
- Motions (`Soort eq 'Motie'`) are not yet searched, though they are the most
  position-dense document type. Candidate for the next widening.
- Some document resources are ~450-char stubs (e.g. the 2024 Kernenergiewet
  plenary texts); extraction yields nothing and the debate effectively
  vanishes. Needs investigation of alternate resource variants.
- BM25 inherits the plan node's terms: if term generation misses a document's
  vocabulary, BM25 is blind there and only the triage excerpt can save it.
- Proximity-aware scoring within chunks is a possible refinement
  ([#53](https://github.com/quantockhills/poldercheck/issues/53)).
