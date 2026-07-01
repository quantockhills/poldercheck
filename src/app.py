import asyncio
import json
import os
import re
import threading
import time

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from src.graph import run_query
from src.storage import delete_conversation, load_history, save_conversation


class _CancelCallback(BaseCallbackHandler):
    """Raises StopIteration between nodes when the stop event is set."""

    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self._stop = stop_event

    def on_chain_start(self, serialized, inputs, **kwargs):
        if self._stop.is_set():
            raise StopIteration("Search cancelled by user.")


class _StatusCallback(BaseCallbackHandler):
    """Pipes LangGraph node and tool events to a Streamlit st.status container."""

    _NODE_LABELS = {
        "query_planner": "Generating search terms...",
        "political": "Searching Tweede Kamer debates...",
        "data": "Fetching CBS data...",
        "synthesis": "Synthesizing...",
    }
    _TOOL_LABELS = {
        "search_tk": "Searching debates",
        "search_by_category": "Searching debates",
        "analyze_document_relevance": "Checking document",
        "get_document_content": "Loading document",
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
        if name in ("search_tk", "search_by_category"):
            q = args.get("query", "")
            if q:
                self._write(f"Searching for: *{q[:80]}*")
        elif name == "analyze_document_relevance":
            self._write(f"Checking relevance: {args.get('docId', '?')}")
        elif name == "get_document_content":
            self._write(f"Loading: {args.get('docId', '?')}")
        elif name == "search_manifesto_corpus":
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


PARTY_COLORS = {
    "VVD": "#FF6600",
    "PVV": "#003082",
    "CDA": "#007B5F",
    "D66": "#00A950",
    "GL": "#5FAD41",
    "GROENLINKS": "#5FAD41",
    "GROENLINKS-PVDA": "#C4332A",
    "SP": "#E1000F",
    "PVDA": "#E1000F",
    "CU": "#00699A",
    "SGP": "#F4821F",
    "PVDD": "#2E7D32",
    "JA21": "#004B87",
    "NSC": "#2196F3",
    "FVD": "#8B0000",
    "BBB": "#7CB342",
    "VOLT": "#5E1D76",
    "DENK": "#00BCD4",
    "50PLUS": "#8E44AD",
    "ARTIKEL1": "#E91E63",
}


def _party_color(party_name: str) -> str:
    key = re.sub(r"[\-_ ]", "", party_name.upper())
    for k, v in PARTY_COLORS.items():
        if re.sub(r"[\-_ ]", "", k) == key:
            return v
    return "#7fbfcf"


def _split_sources(text: str) -> tuple[str, str]:
    match = re.search(r"\n\n((?:##\s*)?Sources[^:]*:?.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), match.group(1).strip()
    return text, ""


def _render_footnotes(text: str) -> str:
    def _to_sup(m):
        nums = re.findall(r"\^(\d+)", m.group(0))
        return f'<sup style="color:#4a90d9;font-weight:600">{",".join(nums)}</sup>'

    return re.sub(r"(\^\d+)+", _to_sup, text)


def _linkify_ids(text: str) -> str:
    text = re.sub(
        r"\b(\d{5,6}[A-Z]{2,3})\b",
        r'<a href="https://www.cbs.nl/nl-nl/cijfers/detail/\1" target="_blank">\1</a>',
        text,
    )
    text = re.sub(
        r"\b(\d{4}D\d+)\b",
        r'<a href="https://www.tweedekamer.nl/kamerstukken/kamerstuk?id=\1" target="_blank">\1</a>',
        text,
    )
    return text


def _render_source_nums(text: str) -> str:
    text = re.sub(r"\^(\d+)", r"<sup>\1</sup>", text)
    return _linkify_ids(text)


def _translate(text: str, target_lang: str) -> str:
    from openai import OpenAI

    from src.agents.config import AGENT_CONFIGS

    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)
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
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def _render_result(result: dict, display_language: str, translations: dict | None = None, show_trace: bool = False):
    """Render a result dict (final_response + passages + sub-responses) into the current container."""
    # Debug trace expander — only shown when debug mode is active
    if show_trace:
        political_trace = result.get("political_trace", {})
        if political_trace:
            from src.agents.political import format_political_trace
            with st.expander("Pipeline trace", expanded=True):
                st.code(format_political_trace(political_trace), language=None)

    final_text = result.get("final_response", "")
    if not final_text:
        st.info("No response available.")
        return

    elapsed = result.get("_elapsed")
    if elapsed:
        st.caption(f"Completed in {elapsed // 60}:{elapsed % 60:02d}")

    # Use translations cache if provided (live search), otherwise render as-is
    if translations is not None:
        text_to_show = translations.get(display_language, final_text)
    else:
        text_to_show = final_text

    main_text, sources_text = _split_sources(text_to_show)

    with st.container(border=True):
        st.markdown(_render_footnotes(main_text), unsafe_allow_html=True)

    if sources_text:
        st.markdown(
            f'<div class="source-footer">{_render_source_nums(sources_text)}</div>',
            unsafe_allow_html=True,
        )

    passages = result.get("political_passages", [])
    pol_raw = result.get("political_response", "")
    data_raw = result.get("data_response", "")

    tab_labels = (
        (["Parliamentary passages"] if passages else [])
        + (["Political analyst"] if pol_raw else [])
        + (["Data analyst"] if data_raw else [])
    )

    if tab_labels:
        tabs = st.tabs(tab_labels)
        tab_idx = 0

        if passages:
            with tabs[tab_idx]:
                for p in passages:
                    meta = p["metadata"]
                    party = meta.get("party_name", "")
                    color = _party_color(party)
                    score = p.get("relevance_score", "")
                    score_str = f" &nbsp;·&nbsp; relevance {score}" if score else ""
                    st.markdown(
                        f'<div class="party-passage" style="border-left: 4px solid {color};">'
                        f"<strong>{meta.get('source', '?')} ({meta.get('year', '?')})</strong>"
                        f'<span style="color:#6b7c8a; font-size:0.8rem;">{score_str}</span><br/><br/>'
                        f"{p['text']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            tab_idx += 1

        if pol_raw:
            with tabs[tab_idx]:
                st.markdown(pol_raw)
            tab_idx += 1

        if data_raw:
            with tabs[tab_idx]:
                st.markdown(data_raw)


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Poldercheck", page_icon="🌊", layout="wide")

_TOKEN = os.environ.get("ACCESS_TOKEN", "")
if _TOKEN and st.query_params.get("token") != _TOKEN:
    st.stop()

st.markdown(
    """
<style>
    .stApp { background-color: #f8f4f0; color: #1a1a2e; }
    .party-passage {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        background-color: #ffffff;
        color: #1a1a2e;
    }
    .party-passage * { color: #1a1a2e; }
    .source-footer {
        font-size: 0.85rem;
        color: #2d3748;
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
    * { border-radius: 0 !important; }

    /* Grey out last radio option in Mode and CBS query mode radios */
    div[data-testid="stRadio"] div[role="radiogroup"] label:last-of-type {
        opacity: 0.38 !important;
        pointer-events: none !important;
        cursor: not-allowed !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:last-of-type input {
        pointer-events: none !important;
    }
    /* Exception: Language radio (first stRadio) — "English (EN)" is a real option, not coming soon */
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] label:last-of-type {
        opacity: 1 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] label:last-of-type input {
        pointer-events: auto !important;
    }

    /* Search button — glassy grey, black text */
    div[data-testid="stButton"] button,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {
        background: rgba(200, 200, 200, 0.45) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(160, 160, 160, 0.5) !important;
        color: #1a1a2e !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] button:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: rgba(180, 180, 180, 0.6) !important;
        color: #1a1a2e !important;
        border: 1px solid rgba(140, 140, 140, 0.6) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center'>POLDERCHECK</h1>", unsafe_allow_html=True)
st.caption("Connecting Dutch politics and policy to data, in a way anyone can understand.")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**About**")
    st.markdown(
        "Poldercheck connects what Dutch politicians say in parliament "
        "to what the data actually shows. It cites its sources and says "
        "when it does not know."
    )
    st.divider()
    st.markdown("**Data sources**")
    st.markdown(
        "- [Tweede Kamer debates](https://www.tweedekamer.nl) (live)\n"
        "- [CBS Statistics Netherlands](https://www.cbs.nl)\n"
        "- [Party manifestos](https://dnpp.nl) (Manifesto Project)\n"
        "- [CPB Charted Choices](https://www.cpb.nl)\n"
        "- [PBL Climate Analysis](https://www.pbl.nl)"
    )
    st.divider()
    st.caption("National-level politics only. Not a stemhulp.")
    st.divider()
    language = "en" if st.radio("Language", ["Dutch (NL)", "English (EN)"]) == "English (EN)" else "nl"
    st.radio("Mode", ["Deep (thorough)", "Fast (coming soon)"], index=0)
    mode = "deep"
    pedagogical = st.checkbox(
        "Pedagogical mode",
        value=True,
        help="Explains Dutch terms, abbreviations, and policy names inline.",
    )
    st.divider()
    st.markdown("**Sources**")
    include_manifestos = st.checkbox(
        "Party manifestos & CPB/PBL",
        value=True,
        help="Search party PDFs, CPB Charted Choices, and PBL Climate Analysis.",
    )
    include_tk = st.checkbox("Tweede Kamer debates", value=True, help="Search live parliamentary debates via OpenTK.")
    include_cbs = st.checkbox(
        "CBS statistical data", value=True, help="Fetch CBS StatLine data to support or challenge political claims.",
    )
    st.radio(
        "CBS query mode",
        ["DuckDB (local SQL)", "MCP (coming soon)"],
        index=0,
        disabled=not include_cbs,
        help="DuckDB: download CSV and query with SQL (faster, more reliable).",
    )
    cbs_mode = "duckdb"
    num_datasets = st.number_input(
        "CBS datasets to query",
        min_value=1, max_value=10, value=3,
        help="Number of CBS datasets the agent will search and present.",
        disabled=not include_cbs,
    )
    st.divider()
    debug = st.checkbox(
        "Debug mode",
        value=False,
        help="Show pipeline trace after each run: OData keywords, per-year doc counts, node timings.",
    )

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
    query, language, mode, pedagogical, include_manifestos, include_tk, include_cbs, cbs_mode, num_datasets, debug, stop_event, msgs, out
):
    def _go():
        try:
            import time as _time
            _t0 = _time.time()
            cb = _StatusCallback(msgs.append)
            out["result"] = asyncio.run(
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
                    extra_callbacks=[_CancelCallback(stop_event), cb],
                    debug=debug,
                )
            )
            out["elapsed"] = int(_time.time() - _t0)
        except StopIteration:
            out["cancelled"] = True
        except Exception as exc:
            out["error"] = str(exc)
        finally:
            out["done"] = True

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


# ── search tab ────────────────────────────────────────────────────────────────
with tab_search:
    with st.form("search_form", clear_on_submit=False):
        query = st.text_area(
            "Ask a question about Dutch politics or policy",
            placeholder="e.g. What do parties propose about housing affordability, and what does CBS show?",
            height=120,
        )
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    # start a new search
    if submitted and query and st.session_state.app_state == "idle":
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
            cbs_mode, num_datasets, debug, stop_event, msgs, out
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

        _render_result(result, language, translations=st.session_state.translations, show_trace=debug)


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
