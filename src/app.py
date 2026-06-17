import asyncio
import re

import streamlit as st

from src.graph import run_query

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
    """Split response into body and Sources section."""
    match = re.search(r"\n\n(Sources[^:]*:.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), match.group(1).strip()
    return text, ""


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

if query:
    with st.spinner("Consulting parliamentary records and CBS data..."):
        result = asyncio.run(run_query(query, language=language, mode=mode, pedagogical=pedagogical))

    main_text, sources_text = _split_sources(result["final_response"])

    with st.container(border=True):
        st.markdown(main_text)

    if sources_text:
        st.markdown(
            f'<div class="source-footer">{sources_text}</div>',
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
