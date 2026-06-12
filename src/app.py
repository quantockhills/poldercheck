import asyncio

import streamlit as st

from src.graph import run_query

st.set_page_config(
    page_title="Poldercheck",
    page_icon="🌊",
    layout="wide",
)

# Pastel palette via custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f4f0; }
    .main-header { color: #2d5a6b; font-size: 2.2rem; font-weight: 700; }
    .tagline { color: #6b7c8a; font-size: 1.1rem; margin-bottom: 2rem; }
    .response-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #7fbfcf;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #4a5568;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Poldercheck 🌊</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Connecting Dutch politics and policy to data, '
    'in a way anyone can understand.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Poldercheck connects what Dutch politicians say in parliament "
        "to what the data actually shows. It cites its sources and says "
        "when it does not know."
    )
    st.markdown("### Data sources")
    st.markdown("- Tweede Kamer debates (live via OpenTK)")
    st.markdown("- CBS Statistics Netherlands")
    st.markdown("- Party manifestos (Manifesto Project)")
    st.markdown("- CPB Charted Choices")
    st.markdown("- PBL Climate Analysis")
    st.markdown("### Language")
    st.markdown("Ask in English or Dutch. Sources are in Dutch; "
                "responses are in English.")

query = st.text_input(
    "Ask a question about Dutch politics or policy",
    placeholder="e.g. What do parties propose about housing affordability, "
                "and what do CBS statistics show?"
)

if query:
    with st.spinner("Consulting parliamentary records and CBS data..."):
        result = asyncio.run(run_query(query))

    st.markdown(
        f'<div class="response-box">{result["final_response"]}</div>',
        unsafe_allow_html=True
    )

    with st.expander("Show retrieved passages (static corpus)"):
        for p in result.get("political_passages", []):
            meta = p["metadata"]
            st.markdown(
                f"**{meta.get('source', '?')}** "
                f"(relevance: {p.get('relevance_score', '?')})"
            )
            st.markdown(
                f'<div class="source-box">{p["text"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")

    with st.expander("Political analyst raw response"):
        st.markdown(result.get("political_response", ""))

    with st.expander("Data analyst raw response"):
        st.markdown(result.get("data_response", ""))
