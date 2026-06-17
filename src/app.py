import asyncio
import json
import re
import threading

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from src.graph import run_query


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
        "political":     "Searching Tweede Kamer debates...",
        "data":          "Fetching CBS data...",
        "synthesis":     "Synthesizing...",
    }
    _TOOL_LABELS = {
        "search_tk":                  "Searching debates",
        "search_by_category":         "Searching debates",
        "analyze_document_relevance": "Checking document",
        "get_document_content":       "Loading document",
        "get_observations":           "Fetching CBS observations",
    }

    def __init__(self, write_fn):
        super().__init__()
        self._write = write_fn

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("name", "")
        if name in self._NODE_LABELS:
            self._write(self._NODE_LABELS[name])

    def on_chain_end(self, outputs, **kwargs):
        # Surface CBS search terms after query_planner finishes
        if isinstance(outputs, dict) and "cbs_queries" in outputs:
            queries = outputs["cbs_queries"]
            if queries:
                self._write(f"CBS search terms: *{', '.join(queries)}*")

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
    """Split response into body and Sources section (handles ## Sources or Sources:)."""
    match = re.search(r"\n\n((?:##\s*)?Sources[^:]*:?.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), match.group(1).strip()
    return text, ""


def _render_footnotes(text: str) -> str:
    """Convert ^1, ^2, consecutive ^1^2 → blue HTML superscripts."""
    def _to_sup(m):
        nums = re.findall(r"\^(\d+)", m.group(0))
        return f'<sup style="color:#4a90d9;font-weight:600">{",".join(nums)}</sup>'
    return re.sub(r"(\^\d+)+", _to_sup, text)


def _render_source_nums(text: str) -> str:
    """Convert leading ^N in source list entries to plain superscripts."""
    return re.sub(r"\^(\d+)", r"<sup>\1</sup>", text)


def _translate(text: str, target_lang: str) -> str:
    """Translate a final_response to the target language, preserving footnotes."""
    from openai import OpenAI
    from src.agents.config import AGENT_CONFIGS
    cfg = AGENT_CONFIGS["synthesis"]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=60)
    lang_name = "English" if target_lang == "en" else "Dutch"
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": (
            f"Translate the following to {lang_name}. "
            "Preserve all superscript citations (^1, ^2, etc.), source IDs, document identifiers, "
            "and the ## Sources section structure exactly — only translate the prose.\n\n" + text
        )}],
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Poldercheck", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f4f0; }
    .party-passage {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        background-color: #ffffff;
    }
    .source-footer {
        font-size: 0.85rem;
        color: #4a5568;
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## Poldercheck 🌊")
st.caption("Connecting Dutch politics and policy to data, in a way anyone can understand.")

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
        "- Tweede Kamer debates (live)\n"
        "- CBS Statistics Netherlands\n"
        "- Party manifestos (Manifesto Project)\n"
        "- CPB Charted Choices\n"
        "- PBL Climate Analysis"
    )
    st.divider()
    st.caption("National-level politics only. Not a stemhulp.")
    st.divider()
    language = "en" if st.radio("Language", ["Dutch (NL)", "English (EN)"]) == "English (EN)" else "nl"
    mode = "fast" if st.radio("Mode", ["Deep (thorough)", "Fast"]) == "Fast" else "deep"
    pedagogical = st.checkbox(
        "Pedagogical mode",
        help="Explains Dutch terms, abbreviations, and policy names inline.",
    )

query = st.text_input(
    "Ask a question about Dutch politics or policy",
    placeholder="e.g. What do parties propose about housing affordability, and what does CBS show?",
)

# ── session state ────────────────────────────────────────────────────────────
for _k, _v in [("app_state", "idle"), ("search_out", {}), ("status_msgs", []),
               ("stop_event", None), ("search_thread", None),
               ("translations", {}), ("search_language", None)]:
    st.session_state.setdefault(_k, _v)


def _search_thread(query, language, mode, pedagogical, stop_event, msgs, out):
    """Runs run_query in a daemon thread; stores result in `out`."""
    def _go():
        try:
            cb = _StatusCallback(msgs.append)
            out["result"] = asyncio.run(
                run_query(query, language=language, mode=mode,
                          pedagogical=pedagogical,
                          extra_callbacks=[_CancelCallback(stop_event), cb])
            )
        except StopIteration:
            out["cancelled"] = True
        except Exception as exc:
            out["error"] = str(exc)
        finally:
            out["done"] = True
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


# ── start a new search ───────────────────────────────────────────────────────
if query and st.session_state.app_state == "idle":
    stop_event = threading.Event()
    msgs: list = []
    out: dict = {"done": False}
    st.session_state.stop_event = stop_event
    st.session_state.status_msgs = msgs
    st.session_state.search_out = out
    st.session_state.translations = {}
    st.session_state.search_language = language
    st.session_state.search_thread = _search_thread(
        query, language, mode, pedagogical, stop_event, msgs, out
    )
    st.session_state.app_state = "searching"
    st.rerun()

# ── poll while searching ─────────────────────────────────────────────────────
if st.session_state.app_state == "searching":
    out = st.session_state.search_out
    msgs = st.session_state.status_msgs

    with st.status("Consulting parliamentary records and CBS data...", expanded=True) as _status:
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
        import time; time.sleep(0.5)
        st.rerun()

# ── render result ────────────────────────────────────────────────────────────
if st.session_state.app_state == "done":
    out = st.session_state.search_out
    if out.get("error"):
        st.error(f"Error: {out['error']}")
        st.stop()
    if out.get("cancelled") or not out.get("result"):
        st.info("Search stopped.")
        st.stop()

    result = out["result"]

    # Seed the translation cache with the language the search was run in
    search_lang = st.session_state.search_language or language
    if search_lang not in st.session_state.translations:
        st.session_state.translations[search_lang] = result["final_response"]

    # Translate on toggle if we don't have this language cached yet
    if language not in st.session_state.translations:
        with st.spinner(f"Translating to {'English' if language == 'en' else 'Dutch'}..."):
            st.session_state.translations[language] = _translate(
                st.session_state.translations[search_lang], language
            )

    main_text, sources_text = _split_sources(st.session_state.translations[language])

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

    tab_labels = (["Parliamentary passages"] if passages else []) + (
        ["Political analyst"] if pol_raw else []
    ) + (["Data analyst"] if data_raw else [])

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
                        f'<strong>{meta.get("source", "?")} ({meta.get("year", "?")})</strong>'
                        f'<span style="color:#6b7c8a; font-size:0.8rem;">{score_str}</span><br/><br/>'
                        f'{p["text"]}'
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
