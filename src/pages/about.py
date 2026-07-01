import streamlit as st

st.set_page_config(page_title="About — Poldercheck", page_icon="🌊", layout="wide")

st.markdown("# About Poldercheck")
st.markdown("""
Poldercheck has three uses. The first is to follow Dutch parliamentary debate: search what the Tweede Kamer has discussed on any topic, which parties took which positions, and what motions were submitted. The second is to query CBS data: Statistics Netherlands publishes over 4,000 datasets on housing, income, demographics, energy, and labour, and Poldercheck makes them searchable in plain language, including for questions that have nothing to do with politics. The third is to use both together: ask a political question and see whether the CBS numbers support or contradict what is being claimed in parliament.

---

### Why this exists

Public debate in the Netherlands, like everywhere, is shaped as much by ideology and narrative as by evidence. CBS publishes thousands of datasets on housing, income inequality, energy, health, and more. The Tweede Kamer publishes the full transcript of every parliamentary debate. CPB and PBL independently score every party manifesto before each election. This information exists, it is free, and most people never see it.

Research by the Autoriteit Persoonsgegevens (October 2025) documented that general-purpose chatbots used for voting guidance give biased advice, cite no sources, and systematically ignore local parties. Poldercheck is an experiment in a different direction: it connects what politicians say in parliament to what the data actually shows, tries to present the perspectives of different parties without taking sides, and is honest about what it does not know.

The personal motivation is simpler. I moved to the Netherlands for a PhD and decided to stay. The stikstofcrisis, the housing shortage, the pension reform debates — these come up constantly in Dutch life, and making sense of them requires context that takes years to accumulate. I wanted a tool that could help with that, for me and for anyone else trying to understand the country they live in.

---

### What it draws on

| Source | What it covers |
|---|---|
| Tweede Kamer debates | Parliamentary proceedings, motions, voting records (live) |
| CBS StatLine | 4,000+ statistical datasets: housing, economy, demographics, energy |
| Party manifestos | Coded quasi-sentence-level manifesto text, every major party since 1945 (Manifesto Project) |
| CPB Charted Choices | Economic scoring of party manifestos, every election since 1986 |
| PBL climate analysis | Environmental impact of party manifestos, per election |

---

### On honesty

An AI tool about politics that produces confident misinformation is worse than no tool at all.

**Responses are anchored to retrieved text.** Every factual claim traces back to a specific retrieved passage. The quote is not decorative: it is the evidence. The model is not permitted to assert things that are not grounded in what was retrieved.

**The corpus is finite and acknowledged as such.** When a topic is not in the corpus, the system says so explicitly. Absence of evidence here is not evidence of absence.

**Party positions are framed as positions, not facts.** When a party argues that immigration drives housing costs, the system says "Party X has argued that..." not "Immigration drives housing costs." Political claims are contestable. The system does not adjudicate them.

**Poldercheck is not a stemhulp.** It will not tell you what to vote, recommend a party, or rank parties.

**Multiple perspectives are presented, not a verdict.** For questions like "has party X kept its promises?" the system presents the case that could be made for, and the case against, drawing on what the corpus actually shows. The goal is to hand you the material to form your own view.

---

*Currently covers national politics only. Local parties and municipal councils are on the roadmap.*
*Fully open source: [github.com/quantockhills/poldercheck](https://github.com/quantockhills/poldercheck)*
""")
