"""Shared Streamlit UI helpers: page CSS and result rendering.

Used by the main app (src/app.py) and the standalone pages (src/pages/),
so that saved conversations render identically everywhere.
"""

import base64
import os
import re

import streamlit as st

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


def party_color(party_name: str) -> str:
    key = re.sub(r"[\-_ ]", "", party_name.upper())
    for k, v in PARTY_COLORS.items():
        if re.sub(r"[\-_ ]", "", k) == key:
            return v
    return "#7fbfcf"


def split_sources(text: str) -> tuple[str, str]:
    match = re.search(r"\n\n((?:##\s*)?Sources[^:]*:?.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), match.group(1).strip()
    return text, ""


def render_footnotes(text: str) -> str:
    def _to_sup(m):
        nums = re.findall(r"\^(\d+)", m.group(0))
        return f'<sup style="color:#4a90d9;font-weight:600">{",".join(nums)}</sup>'

    return re.sub(r"(\^\d+)+", _to_sup, text)


def linkify_ids(text: str) -> str:
    text = re.sub(
        r"\b(\d{5,6}[A-Z]{2,3})\b",
        r'<a href="https://www.cbs.nl/nl-nl/cijfers/detail/\1" target="_blank">\1</a>',
        text,
    )
    text = re.sub(
        r"\b(\d{4}D\d+)\b",
        r'<a href="https://www.tweedekamer.nl/kamerstukken/detail?did=\1&id=\1" target="_blank">\1</a>',
        text,
    )
    return text


def render_source_nums(text: str) -> str:
    text = re.sub(r"\^(\d+)", r"<sup>\1</sup>", text)
    return linkify_ids(text)


def sources_to_html(text: str) -> str:
    """Convert the ## Sources markdown block to HTML for embedding inside a styled div."""
    lines = linkify_ids(text).split("\n")
    parts = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.match(r"^#{1,3}\s+", s):
            heading = re.sub(r"^#{1,3}\s+", "", s)
            parts.append(f"<p style='margin:0 0 0.4rem 0;font-weight:600'>{heading}</p>")
        else:
            s = re.sub(r"\^(\d+)", r"<sup>\1</sup>", s)
            parts.append(f"<p style='margin:0.15rem 0'>{s}</p>")
    return "\n".join(parts)


def render_result(result: dict, display_language: str, translations: dict | None = None, show_trace: bool = False):
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

    main_text, sources_text = split_sources(text_to_show)

    with st.container(border=True):
        st.markdown(render_footnotes(main_text), unsafe_allow_html=True)

    if sources_text:
        st.markdown(
            f'<div class="source-footer">{sources_to_html(sources_text)}</div>',
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
                    color = party_color(party)
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


def inject_page_css():
    """Inject the shared background image and frosted-glass styling into the page."""
    bg_path = os.path.join(os.path.dirname(__file__), "..", "data", "bg.jpg")
    try:
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        bg_css = f"url('data:image/jpeg;base64,{bg_b64}')"
    except FileNotFoundError:
        bg_css = "none"

    st.markdown(
        f"""
<style>
    .stApp {{
        background-image: {bg_css};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #1a1a2e;
    }}
    .party-passage {{
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        background-color: rgba(255,255,255,0.88);
        color: #1a1a2e;
    }}
    .party-passage * {{ color: #1a1a2e; }}
    .source-footer {{
        font-size: 0.85rem;
        color: #2d3748;
        background-color: rgba(240,244,248,0.88);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }}
    * {{ border-radius: 0 !important; }}

    /* Hide auto-generated multipage nav from sidebar */
    [data-testid="stSidebarNav"] {{ display: none !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(248, 244, 240, 0.90) !important;
        backdrop-filter: blur(6px) !important;
        -webkit-backdrop-filter: blur(6px) !important;
        color: #1a1a2e !important;
    }}

    /* Reusable frosted box for individual text elements */
    .pc-box {{
        display: inline-block;
        background-color: rgba(248, 244, 240, 0.88);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 0.35rem 1rem;
        color: #1a1a2e;
    }}

    /* Equal-width tabs + frosted tab bar */
    div[data-testid="stTabs"] div[role="tablist"],
    [data-baseweb="tab-list"] {{
        display: flex !important;
        background-color: rgba(248, 244, 240, 0.88) !important;
        backdrop-filter: blur(6px) !important;
        -webkit-backdrop-filter: blur(6px) !important;
    }}
    div[data-testid="stTabs"] div[role="tablist"] button[role="tab"],
    [data-baseweb="tab-list"] button[role="tab"] {{
        flex: 1 !important;
        justify-content: center !important;
    }}

    /* Frosted box for each tab's content */
    div[role="tabpanel"],
    [data-baseweb="tab-panel"],
    div[data-testid="stTabPanel"],
    div[data-testid="stTabsTabPanel"],
    div[data-baseweb="tab-panel"][role="tabpanel"] {{
        background-color: rgba(248, 244, 240, 0.88) !important;
        backdrop-filter: blur(6px) !important;
        -webkit-backdrop-filter: blur(6px) !important;
        padding: 1.5rem !important;
        color: #1a1a2e !important;
    }}

    /* Search bar — suppress red focus ring */
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextArea"] textarea:focus-visible {{
        border-color: rgba(160, 160, 160, 0.6) !important;
        box-shadow: 0 0 0 1px rgba(160, 160, 160, 0.6) !important;
        outline: none !important;
    }}

    /* Search button — glassy grey, black text */
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {{
        background: rgba(200, 200, 200, 0.45) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(160, 160, 160, 0.5) !important;
        color: #1a1a2e !important;
        box-shadow: none !important;
    }}
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    button[data-testid="baseButton-primary"]:hover {{
        background: rgba(180, 180, 180, 0.6) !important;
        color: #1a1a2e !important;
        border: 1px solid rgba(140, 140, 140, 0.6) !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
