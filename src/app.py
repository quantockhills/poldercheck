import asyncio
import json
import os
import threading
import time

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from src.graph import run_query
from src.storage import delete_conversation, load_history, save_conversation
from src.ui import inject_page_css
from src.ui import render_result as _render_result


class _StatusCallback(BaseCallbackHandler):
    """Pipes LangGraph node and tool events to a Streamlit st.status container."""

    _NODE_LABELS = {
        "query_planner": "Generating search terms...",
        "political": "Searching Tweede Kamer debates...",
        "data": "Fetching CBS data...",
        "synthesis": "Synthesizing...",
    }
    _TOOL_LABELS = {
        "search_manifesto_corpus": "Searching manifestos & policy reports",
        "search_cbs_catalog": "Searching CBS catalog",
        "fetch_cbs_dataset": "Fetching CBS dataset",
        "query_datasets": "Querying CBS catalog",
        "get_dimensions": "Inspecting CBS dataset",
        "get_dimension_values": "Loading dimension values",
        "get_observations": "Fetching CBS data",
        "query_observations": "Fetching CBS data",
    }

    def __init__(self, write_fn):
        super().__init__()
        self._write = write_fn

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("name", "")
        if name in self._NODE_LABELS:
            self._write(self._NODE_LABELS[name])

    def on_chain_end(self, outputs, **kwargs):
        if isinstance(outputs, dict) and outputs.get("cbs_queries"):
            self._write(f"CBS search terms: *{', '.join(outputs['cbs_queries'])}*")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "")
        if name not in self._TOOL_LABELS:
            return
        args = {}
        if isinstance(input_str, dict):
            args = input_str
        elif isinstance(input_str, str):
            try:
                args = json.loads(input_str)
            except Exception:
                import ast

                try:
                    args = ast.literal_eval(input_str)
                except Exception:
                    pass
        if name == "search_manifesto_corpus":
            q = args.get("query", "")
            if q:
                self._write(f"Searching manifestos/CPB/PBL: *{q[:60]}*")
        elif name in ("search_cbs_catalog",):
            q = args.get("query", "")
            if q:
                self._write(f"Searching CBS catalog: *{q[:60]}*")
        elif name == "fetch_cbs_dataset":
            did = args.get("dataset_id", "?")
            self._write(f"Fetching CBS dataset: *{did}*")
        elif name == "query_datasets":
            q = args.get("query", args.get("term", args.get("q", "")))
            if q:
                self._write(f"Searching CBS catalog: *{q[:60]}*")
        elif name == "get_dimensions":
            did = args.get("dataset", args.get("datasetId", "?"))
            self._write(f"Inspecting CBS dataset: *{did}*")
        elif name == "get_dimension_values":
            did = args.get("dataset", args.get("datasetId", "?"))
            dim = args.get("dimension", args.get("dimensionName", ""))
            self._write(f"Loading {dim} values for *{did}*")
        elif name in ("get_observations", "query_observations"):
            did = args.get("dataset", args.get("datasetId", "?"))
            self._write(f"Fetching CBS data: *{did}*")


def _translate(text: str, target_lang: str) -> str:
    from openai import OpenAI

    from src.agents.config import AGENT_CONFIGS

    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=600)
    lang_name = "English" if target_lang == "en" else "Dutch"
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Translate the following to {lang_name}. "
                    "Preserve all superscript citations (^1, ^2, etc.), source IDs, document identifiers, "
                    "and the ## Sources section structure exactly — only translate the prose.\n\n" + text
                ),
            }
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Poldercheck", page_icon="🌊", layout="wide")

_TOKEN = os.environ.get("ACCESS_TOKEN", "")
if _TOKEN and st.query_params.get("token") != _TOKEN:
    st.stop()

# Presentation mode: set PRESENTATION_MODE=1 in .env when hosting for visitors.
# Disables the manifesto/CPB pipeline entirely (server RAM cannot hold ChromaDB
# yet) and shows an expectation-setting notice on the search screen that links
# to the curated Examples page.
_PRESENTATION = os.environ.get("PRESENTATION_MODE", "").lower() in ("1", "true", "yes")

@st.cache_resource
def _static_corpus_available() -> bool:
    try:
        import chromadb
        _path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        client = chromadb.PersistentClient(path=_path)
        client.get_collection("poldercheck_static")
        return True
    except Exception:
        return False


inject_page_css()

st.markdown(
    "<div style='text-align:center;margin-bottom:0.5rem'>"
    "<span class='pc-box'><h1 style='margin:0;padding:0'>POLDERCHECK</h1></span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    language = "en" if st.radio("Language", ["Dutch (NL)", "English (EN)"]) == "English (EN)" else "nl"
    st.radio("Mode", ["Deep (thorough)"], index=0)
    st.caption("Fast mode: coming soon")
    mode = "deep"
    pedagogical = st.checkbox(
        "Pedagogical mode",
        value=True,
        help="Explains Dutch terms, abbreviations, and policy names inline.",
    )
    st.divider()
    st.markdown("**Sources**")
    # Short-circuit so presentation mode never even loads the ChromaDB corpus
    _manifesto_ready = (not _PRESENTATION) and _static_corpus_available()
    include_manifestos = st.checkbox(
        "Party manifestos & CPB/PBL",
        value=_manifesto_ready,
        disabled=not _manifesto_ready,
        help="Search party PDFs, CPB Charted Choices, and PBL Climate Analysis.",
    )
    if not _manifesto_ready:
        st.caption("Manifesto corpus: coming soon")
    if _PRESENTATION:
        include_manifestos = False
    include_tk = st.checkbox("Tweede Kamer debates", value=True, help="Search live parliamentary debates via the Tweede Kamer OData API (2018 onwards).")
    include_cbs = st.checkbox(
        "CBS statistical data", value=True, help="Fetch CBS StatLine data to support or challenge political claims.",
    )
    st.radio(
        "CBS query mode",
        ["DuckDB (local SQL)"],
        index=0,
        disabled=not include_cbs,
        help="DuckDB: download CSV and query with SQL (faster, more reliable).",
    )
    st.caption("MCP mode: coming soon")
    cbs_mode = "duckdb"
    num_datasets = st.number_input(
        "CBS datasets to query",
        min_value=1, max_value=10, value=5,
        help="Number of CBS datasets the agent will search and present.",
        disabled=not include_cbs,
    )
    st.divider()
    st.caption("National-level politics only. Not a stemhulp.")
    st.caption("Fully open source · [github.com/quantockhills/poldercheck](https://github.com/quantockhills/poldercheck)")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_history = st.tabs(["Search", "History"])

# ── session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("app_state", "idle"),
    ("search_out", {}),
    ("status_msgs", []),
    ("stop_event", None),
    ("search_thread", None),
    ("translations", {}),
    ("search_language", None),
    ("last_query", ""),
]:
    st.session_state.setdefault(_k, _v)


def _search_thread(
    query, language, mode, pedagogical, include_manifestos, include_tk, include_cbs, cbs_mode, num_datasets, stop_event, msgs, out
):
    async def _cancellable():
        # Watch stop_event and cancel the whole graph run; cancellation propagates
        # through every awaited LLM/HTTP call, actually aborting in-flight work.
        task = asyncio.ensure_future(
            run_query(
                query,
                language=language,
                mode=mode,
                pedagogical=pedagogical,
                include_manifestos=include_manifestos,
                include_tk=include_tk,
                include_cbs=include_cbs,
                cbs_mode=cbs_mode,
                num_datasets=num_datasets,
                on_status=msgs.append,
                extra_callbacks=[_StatusCallback(msgs.append)],
                debug=False,
            )
        )
        while not task.done():
            if stop_event.is_set():
                task.cancel()
                break
            await asyncio.sleep(0.5)
        return await task

    def _go():
        try:
            import time as _time
            _t0 = _time.time()
            out["result"] = asyncio.run(_cancellable())
            out["elapsed"] = int(_time.time() - _t0)
        except asyncio.CancelledError:
            out["cancelled"] = True
        except Exception as exc:
            out["error"] = str(exc)
        finally:
            out["done"] = True

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


# ── search tab ────────────────────────────────────────────────────────────────
_SEARCH_LABEL = "Ask a question about Dutch politics or CBS data, or [learn more about the project](/about)"
with tab_search:
    with st.form("search_form", clear_on_submit=False):
        if _PRESENTATION:
            st.markdown(_SEARCH_LABEL)
            st.markdown("Or [click here to see examples of real questions we ran ourselves](/examples).")
            st.caption(
                "Because every search runs live through more than 16,500 Tweede Kamer debate "
                "transcripts (2018 onwards, plenary and committee) and analyses statistics from "
                "nearly 1,300 CBS datasets, the process can take around five minutes. A fast mode "
                "is coming soon."
            )
            query = st.text_area(
                _SEARCH_LABEL,
                label_visibility="collapsed",
                placeholder="e.g. What do parties propose about housing affordability, and what does CBS show?",
                height=120,
            )
        else:
            query = st.text_area(
                _SEARCH_LABEL,
                placeholder="e.g. What do parties propose about housing affordability, and what does CBS show?",
                height=120,
            )
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    # start a new search
    if submitted and query and st.session_state.app_state in ("idle", "done"):
        stop_event = threading.Event()
        msgs: list = []
        out: dict = {"done": False}
        st.session_state.stop_event = stop_event
        st.session_state.status_msgs = msgs
        st.session_state.search_out = out
        st.session_state.translations = {}
        st.session_state.search_language = language
        st.session_state.last_query = query
        st.session_state.pop("current_conv_id", None)
        st.session_state.search_thread = _search_thread(
            query, language, mode, pedagogical, include_manifestos, include_tk, include_cbs,
            cbs_mode, num_datasets, stop_event, msgs, out
        )
        st.session_state.app_state = "searching"
        st.rerun()

    # poll while searching
    if st.session_state.app_state == "searching":
        out = st.session_state.search_out
        msgs = st.session_state.status_msgs

        with st.status("Searching...", expanded=True) as _status:
            for m in msgs:
                _status.write(m)
            if st.button("Stop", type="secondary"):
                st.session_state.stop_event.set()
                st.session_state.app_state = "idle"
                _status.update(label="Stopped", state="error", expanded=False)
                st.rerun()

        if out.get("done"):
            st.session_state.app_state = "done"
            st.rerun()
        else:
            time.sleep(0.5)
            st.rerun()

    # render result
    if st.session_state.app_state == "done":
        out = st.session_state.search_out
        if out.get("error"):
            st.error(f"Error: {out['error']}")
            st.stop()
        if out.get("cancelled") or not out.get("result"):
            st.info("Search stopped.")
            st.stop()

        result = out["result"]
        if "elapsed" in out:
            result["_elapsed"] = out["elapsed"]

        # Seed translation cache
        search_lang = st.session_state.search_language or language
        if search_lang not in st.session_state.translations:
            st.session_state.translations[search_lang] = result["final_response"]

        # Translate on toggle
        if language not in st.session_state.translations:
            with st.spinner(f"Translating to {'English' if language == 'en' else 'Dutch'}..."):
                st.session_state.translations[language] = _translate(
                    st.session_state.translations[search_lang], language
                )

        # Save to history (once, keyed by conv_id in session state)
        if "current_conv_id" not in st.session_state:
            settings = {
                "language": search_lang,
                "mode": mode,
                "pedagogical": pedagogical,
                "include_manifestos": include_manifestos,
                "include_tk": include_tk,
                "include_cbs": include_cbs,
            }
            st.session_state.current_conv_id = save_conversation(st.session_state.last_query, result, settings)

        _render_result(result, language, translations=st.session_state.translations)


# ── history tab ───────────────────────────────────────────────────────────────
with tab_history:
    convos = load_history()
    if not convos:
        st.info("No past searches yet. Run a query and it will appear here.")
    else:
        st.caption(f"{len(convos)} saved {'search' if len(convos) == 1 else 'searches'}")
        for c in convos:
            ts = c.get("timestamp", "")[:16].replace("T", " ")
            q = c.get("query", "")
            settings = c.get("settings", {})
            lang_label = "EN" if settings.get("language") == "en" else "NL"
            header = f"{ts} · {lang_label} · {q[:70]}"

            with st.expander(header):
                st.markdown(f"**{q}**")
                col_info, col_del = st.columns([9, 1])
                with col_info:
                    tags = []
                    if settings.get("include_manifestos"):
                        tags.append("manifestos")
                    if settings.get("include_tk"):
                        tags.append("TK")
                    if settings.get("include_cbs"):
                        tags.append("CBS")
                    if tags:
                        st.caption(f"Sources: {', '.join(tags)} · {settings.get('mode', 'deep')} mode")
                with col_del:
                    if st.button("Delete", key=f"del_{c['id']}"):
                        delete_conversation(c["id"])
                        st.rerun()

                _render_result(c, settings.get("language", "nl"))


