"""Unit tests for the discover planner's date-window extraction.

Regression anchors:
- "since the 2023 election" was parsed as a single-year 2023 query (the
  since-regex required the year adjacent to the anchor word), which truncated
  retrieval at 2024-01-01 and starved the answer of post-election evidence —
  caught by the RAGAS faithfulness metric (eval case 3, score 0.2).
- "de afgelopen maanden" fell through to the 5-year default window, so
  retrieval ranked 2022 debates as readily as last month's and synthesis
  presented them as current — caught by the rubric metric (eval case 2,
  score 1.0).
"""

from datetime import date

from src.agents.political_discover import parse_date_range

TODAY = date(2026, 7, 5)


def _labels(buckets):
    return [b["year_label"] for b in buckets]


def test_since_with_filler_words():
    f, t, _, buckets = parse_date_range(
        "How have party positions on asylum policy shifted since the 2023 election?", TODAY
    )
    assert f == "2023-01-01"
    assert t == "2026-07-05"
    assert _labels(buckets) == ["2023", "2024", "2025", "2026"]


def test_since_adjacent_year():
    f, t, _, buckets = parse_date_range("since 2020, how did unemployment change", TODAY)
    assert (f, t) == ("2020-01-01", "2026-07-05")
    assert _labels(buckets)[0] == "2020" and _labels(buckets)[-1] == "2026"


def test_dutch_sinds_with_fillers():
    f, t, _, buckets = parse_date_range("sinds de verkiezingen van 2023", TODAY)
    assert (f, t) == ("2023-01-01", "2026-07-05")
    assert _labels(buckets) == ["2023", "2024", "2025", "2026"]


def test_single_year_stays_single_year():
    f, t, _, buckets = parse_date_range("What was debated about housing in 2021?", TODAY)
    assert (f, t) == ("2021-01-01", "2022-01-01")
    assert buckets == []


def test_explicit_range_buckets_clamped_to_end_year():
    f, t, _, buckets = parse_date_range("compare 2019 with 2024", TODAY)
    assert (f, t) == ("2019-01-01", "2024-12-31")
    assert _labels(buckets)[-1] == "2024"


def test_no_year_defaults_to_last_five_years():
    f, t, _, buckets = parse_date_range("Waar ging het debat over woningbouw over?", TODAY)
    assert (f, t) == ("2022-01-01", "2026-07-05")
    assert _labels(buckets) == ["2022", "2023", "2024", "2025", "2026"]


def test_recent_months_dutch_gets_twelve_month_window():
    f, t, _, buckets = parse_date_range(
        "Waar ging het debat over woningbouw in de Tweede Kamer de afgelopen maanden over?", TODAY
    )
    assert (f, t) == ("2025-07-05", "2026-07-05")
    assert buckets == []


def test_recent_months_english_gets_twelve_month_window():
    f, t, _, buckets = parse_date_range(
        "What has parliament debated about housing in recent months?", TODAY
    )
    assert (f, t) == ("2025-07-05", "2026-07-05")
    assert buckets == []


def test_recently_adverb_gets_twelve_month_window():
    f, t, _, buckets = parse_date_range("Wat is er onlangs besloten over stikstof?", TODAY)
    assert (f, t) == ("2025-07-05", "2026-07-05")
    assert buckets == []


def test_explicit_year_beats_recency_phrase():
    f, t, _, _ = parse_date_range(
        "How do the last few months of housing debate compare with 2021?", TODAY
    )
    assert f == "2021-01-01"


def test_recent_years_stays_on_five_year_default():
    f, t, _, buckets = parse_date_range(
        "How has the debate shifted in recent years?", TODAY
    )
    assert (f, t) == ("2022-01-01", "2026-07-05")
    assert _labels(buckets) == ["2022", "2023", "2024", "2025", "2026"]
