# Langfuse

Open-source AI engineering platform for observability, prompt management, and evaluation of LLM applications.

## Core Features

### Observability
- Comprehensive tracing: LLM calls, retrieval, embeddings, API interactions
- Session tracking for multi-turn conversations
- Agent workflow visualisation as graphs
- Capture via native SDKs, framework integrations, OpenTelemetry, or LLM gateways

### Prompt Management
- Create and version prompts via UI, SDKs, or APIs
- Interactive testing via LLM Playground
- Deploy using labels without code changes
- Compare performance across prompt versions

### Evaluation
- LLM-as-a-judge
- Code-based evaluators
- User feedback collection
- Manual annotation workflows
- Production monitoring

## Usage in poldercheck

`langfuse` is wired into `src/graph.py` via `CallbackHandler` (v3 import path). Active when `LANGFUSE_PUBLIC_KEY` is set in `.env`. Traces every LangGraph node invocation.

## Links
- Docs: https://langfuse.com/docs
- GitHub: https://github.com/langfuse/langfuse
