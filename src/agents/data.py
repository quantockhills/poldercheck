"""Data analyst agent: queries CBS StatLine via the CBS MCP server (Go)."""
import asyncio
import json
import re
from pathlib import Path

import mcp.types
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.agents.config import AGENT_CONFIGS
from src.ingest.retrieve import retrieve_cbs_datasets

# The prebuilt CBS MCP server (v0.2.1, built on an early-2025 mcp-go)
# hard-rejects protocol versions it doesn't know instead of negotiating
# down, so offer the one version it supports. The Python client accepts
# 2024-11-05 on the response side, and per spec a modern server offered an
# old version negotiates down, so this is safe process-wide. Remove if the
# CBS server ships a newer release.
mcp.types.LATEST_PROTOCOL_VERSION = "2024-11-05"

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "data_analyst.txt").read_text()

CBS_NOT_FOUND = (
    "I could not find a CBS dataset relevant to this query. The data may exist "
    "under a different search term, or may not be available in CBS StatLine."
)

_MCP_CONFIG = {
    "cbs": {
        "command": "mcp-cbs-cijfers-open-data",
        "args": ["--stdio"],
        "transport": "stdio",
    }
}


def _tool_text(result) -> str:
    return result[0]["text"] if isinstance(result, list) else str(result)


async def run_data_analyst(query: str, cbs_queries: list[str] | None = None) -> str:
    """
    Parallel fixed pipeline replacing the sequential React agent:

    1. ChromaDB → top 12 candidates (local, ~50ms)
    2. asyncio.gather → get_dimensions on ALL candidates simultaneously
    3. Single LLM call picks top 3 datasets based on dimension structures
    4. asyncio.gather → get_observations on top 3 simultaneously
    5. Single LLM synthesis with transparency footer
    """
    cfg = AGENT_CONFIGS["data_analyst"]
    DEBUG_LOG = print

    queries = cbs_queries if cbs_queries else [query]
    candidates = retrieve_cbs_datasets(queries, n_results=12)
    DEBUG_LOG(f"DEBUG_LOG: catalog found {len(candidates)} candidates: "
              f"{[c['identifier'] for c in candidates]}")

    if not candidates:
        return CBS_NOT_FOUND

    try:
        mcp_client = MultiServerMCPClient(_MCP_CONFIG)
        llm = ChatOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            timeout=60,
            max_retries=1,
        )
        # Use sync OpenAI client for the fast selection call (no thinking, low tokens)
        fast_client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=15)

        async with mcp_client.session("cbs") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}

            # Step 2: parallel get_dimensions on all candidates
            dim_results = await asyncio.gather(*[
                tools["get_dimensions"].ainvoke({"catalog": "CBS", "dataset": c["identifier"]})
                for c in candidates
            ], return_exceptions=True)

            dim_summaries = []
            for c, result in zip(candidates, dim_results):
                if isinstance(result, Exception):
                    continue
                dim_summaries.append(
                    f"{c['identifier']} — {c['title']}:\n{_tool_text(result)[:400]}"
                )

            DEBUG_LOG(f"DEBUG_LOG: got dimensions for {len(dim_summaries)}/{len(candidates)} datasets")

            # Step 3: LLM picks top 3
            selection_prompt = (
                f"Query: {query}\n\n"
                f"CBS datasets with their dimensions:\n\n"
                + "\n\n".join(dim_summaries)
                + "\n\nReturn JSON only — pick the 3 most relevant dataset identifiers:\n"
                  '{"selected": ["XXXXNED", "XXXXNED", "XXXXNED"]}'
            )
            sel_response = fast_client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": selection_prompt}],
                max_tokens=60,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw_sel = sel_response.choices[0].message.content.strip()
            try:
                m = re.search(r'\{.*\}', raw_sel, re.DOTALL)
                selected_ids = json.loads(m.group())["selected"][:3]
            except Exception:
                selected_ids = [c["identifier"] for c in candidates[:3]]

            DEBUG_LOG(f"DEBUG_LOG: selected datasets: {selected_ids}")

            not_queried = [
                f"{c['identifier']} ({c['title']})"
                for c in candidates if c["identifier"] not in set(selected_ids)
            ]

            # Step 4: parallel get_observations on top 3
            obs_results = await asyncio.gather(*[
                tools["get_observations"].ainvoke({"catalog": "CBS", "dataset": did, "limit": 50})
                for did in selected_ids
            ], return_exceptions=True)

            data_blocks = []
            used_labels = []
            for did, result in zip(selected_ids, obs_results):
                if isinstance(result, Exception):
                    DEBUG_LOG(f"DEBUG_LOG: get_observations failed for {did}: {result}")
                    continue
                c_meta = next((c for c in candidates if c["identifier"] == did), {})
                title = c_meta.get("title", did)
                data_blocks.append(f"**{did} — {title}**\n{_tool_text(result)[:1200]}")
                used_labels.append(f"{did} ({title})")

            if not data_blocks:
                return CBS_NOT_FOUND

            # Step 5: synthesis
            transparency = (
                f"\n\n**CBS datasets queried:** {'; '.join(used_labels)}"
                + (f"\n**Also in catalog, not queried:** {'; '.join(not_queried[:4])}" if not_queried else "")
            )

            result = await llm.ainvoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{query}\n\nCBS data:\n\n" + "\n\n---\n\n".join(data_blocks)
                    + "\n\nEnd your response with:\n" + transparency
                )},
            ])
            return result.content

    except (FileNotFoundError, OSError) as exc:
        DEBUG_LOG(f"DEBUG_LOG: CBS MCP server unavailable: {exc}")
        return CBS_NOT_FOUND
