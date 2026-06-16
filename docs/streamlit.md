# Streamlit

Python framework for building interactive data apps. Reruns the entire script top-to-bottom on every user interaction.

## Key Concepts

### Execution model
Every widget interaction or code change triggers a full script rerun. Callbacks (`on_change`, `on_click`) run before the rest of the script.

### Core API

```python
import streamlit as st

# Text and layout
st.title("Title")
st.write("Markdown or data")
st.sidebar.radio("Label", ["Option A", "Option B"])

# Widgets (return values)
choice = st.selectbox("Pick one", ["a", "b"])
query = st.text_input("Query")
submitted = st.button("Submit")

# Layout
col1, col2 = st.columns(2)
with col1:
    st.write("Left")

# Performance
@st.cache_data
def expensive_fn():
    ...
```

### Session State
Widgets with `key=` auto-integrate into `st.session_state`. Use for persisting values across reruns.

### Launching
```bash
streamlit run src/app.py
```

## Usage in poldercheck

`src/app.py` — the main UI. Sidebar has language toggle (NL/EN) and mode toggle (Deep/Fast). On submit, calls `asyncio.run(run_query(...))` and renders `result["final_response"]` as markdown.

## Links
- Docs: https://docs.streamlit.io
- GitHub: https://github.com/streamlit/streamlit
