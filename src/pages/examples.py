"""Curated example searches, rendered exactly like the History tab.

Examples are JSON files in data/examples/ with the same schema as history
files: to publish one, copy its file from data/history/ into data/examples/.
"""

import streamlit as st

from src.storage import load_examples
from src.ui import inject_frosted_main, inject_page_css, render_result

st.set_page_config(page_title="Examples — Poldercheck", page_icon="🌊", layout="wide")

inject_page_css()
inject_frosted_main()

st.markdown(
    "<div style='text-align:center;margin-bottom:0.5rem'>"
    "<span class='pc-box'><h1 style='margin:0;padding:0'>EXAMPLES</h1></span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "These are real questions that we ran through Poldercheck ourselves, shown exactly as a "
    "live search would present them. Each one searched live Tweede Kamer debate transcripts "
    "and CBS statistical datasets. You can [go back to the search page](/) to run your own "
    "question, or [learn more about the project](/about)."
)

st.markdown(
    "Curious how we keep Poldercheck honest? Every search on this page is graded by an "
    "independent examiner, claim by claim. [See our benchmarks](/benchmarks) for how that "
    "works and to read the full reports."
)

examples = load_examples()

if not examples:
    st.info(
        "No examples have been published yet. They are on the way. In the meantime, you can "
        "run your own search from the [main page](/)."
    )
else:
    st.caption(f"{len(examples)} example {'search' if len(examples) == 1 else 'searches'}")
    for c in examples:
        ts = c.get("timestamp", "")[:16].replace("T", " ")
        q = c.get("query", "")
        settings = c.get("settings", {})
        lang_label = "EN" if settings.get("language") == "en" else "NL"
        header = f"{ts} · {lang_label} · {q[:70]}"

        with st.expander(header):
            st.markdown(f"**{q}**")
            tags = []
            if settings.get("include_manifestos"):
                tags.append("manifestos")
            if settings.get("include_tk"):
                tags.append("TK")
            if settings.get("include_cbs"):
                tags.append("CBS")
            if tags:
                st.caption(f"Sources: {', '.join(tags)} · {settings.get('mode', 'deep')} mode")

            render_result(c, settings.get("language", "nl"))
