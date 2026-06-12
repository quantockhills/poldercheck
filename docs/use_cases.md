# Use-case inventory

Example queries, organised by what makes them possible. For inspiration,
demos, and as a quarry for new eval cases (the benchmark set in
`src/eval/eval_set.jsonl` enforces correctness; this file collects ambition).
The strongest demos are the ones no general chatbot can do: they need this
tool's specific data joints.

## 1. Stance shifts over time — the killer category

Powered by the Manifesto Project's consistent coding across elections
(same `cmp_code` scheme in 2017 and 2021, later 2023+).

- "How did VVD's position on housing change between 2017 and 2021?"
- "Which party shifted most on immigration between the last two coded elections?"
- "Did GroenLinks talk more or less about climate in 2021 than 2017?"
- "Which parties discovered housing as a theme only recently?"

Note: "how much more" questions are *quantitative* — share of manifesto
devoted to a policy code per election. That is better answered by
aggregating the coded CSV (pandas over cmp_code counts) than by retrieval.
Candidate feature: a `stance_shift` tool the political analyst can call,
returning emphasis percentages per party/topic/election. Retrieval then
supplies the qualitative "what did they actually say" on top. This combo
(numbers + quotes) is the single most differentiating demo the tool has.

## 2. Promise vs. behaviour

Manifesto layer (what they said) × OpenTK live debates/motions (what they do).

- "PVV campaigned on X — what have they actually said about it in parliament since?"
- "Which parties that promised more social housing voted for motions about it?"
- "What happened in parliament to the nitrogen targets parties promised in 2021?"

## 3. Claim vs. data

Political claim × CBS statistics — the project's namesake move.

- "Parties say homeownership is becoming impossible for starters — what do CBS figures show since 2015?"
- "Is it true that energy poverty is rising?"
- "X claims crime is exploding — what does CBS actually measure?"

(Contract: empirical mismatch is surfaced; causal/contested framing is
presented as positions. See README honesty section.)

## 4. The unread experts

CPB/PBL doorrekeningen — rigorous, public, read by almost nobody.

- "What did CPB calculate the GL-PvdA programme would do to purchasing power?"
- "Which party's 2025 programme scored best on emissions according to the analyses?"
- "What does the CPB say the budgetary cost of party X's tax plan is?"

## 5. Cross-party orientation

Bread-and-butter comparisons, properly cited and never synthesised into a verdict.

- "Where do the parties stand on nuclear energy?"
- "Compare what coalition parties promised on healthcare."
- "Which parties want to change mortgage interest deduction (hypotheekrenteaftrek)?"

## 6. Context for newcomers

The original personal motivation: years of accumulated context, on demand.

- "What is the stikstofcrisis and why does it keep coming up?"
- "Why is the housing shortage such a dominant theme?"
- "What is the history of the pension reform debate?"

## 7. Evaluative questions (critic mode)

Case-for / case-against, never a verdict, ends with an open question.

- "Has the cabinet delivered on its housing promises?"
- "Is the nitrogen policy working?"
- "Was the childcare benefits scandal handled adequately?"

## 8. Honest refusals (these are features)

- "Tell me what the best party is" → refusal (eval-enforced)
- "Who should I vote for in my municipality?" → national-only scope + not-a-stemhulp, names the local gap
- Anything not in corpus → explicit not-found, never confabulated
