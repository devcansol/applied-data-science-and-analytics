"""Pins the audit findings quoted in the README.

If a number here changes, the README is now wrong. Update both together.
"""

from __future__ import annotations

import pandas as pd
import pytest

from startup_outcomes import audit


@pytest.fixture(scope="module")
def df(raw: pd.DataFrame) -> pd.DataFrame:
    """Delegates to the session fixture so the CSV is read once for the whole suite."""
    return raw


def test_target_base_rates(df: pd.DataFrame) -> None:
    rates = audit.target_base_rates(df)
    assert rates["Independent"] == 63.06
    assert rates["Acquired"] == 14.85
    assert rates["Closed"] == 13.81
    assert rates["IPO"] == 4.31
    assert rates["Merger"] == 3.97
    assert rates["binary_majority_baseline_accuracy_pct"] == 86.19


def test_funding_stage_ipo_determines_the_outcome(df: pd.DataFrame) -> None:
    leakage = audit.stage_target_leakage(df)
    assert leakage["rows_at_stage_ipo"] == 533
    assert leakage["of_which_outcome_ipo"] == 533
    assert leakage["stage_ipo_implies_outcome_ipo"] is True
    # One-directional: the reverse implication does not hold.
    assert leakage["outcome_ipo_at_other_stages"] == 545


def test_late_stage_arrives_impossibly_early(df: pd.DataFrame) -> None:
    premature = audit.premature_late_stage(df)
    assert premature["founded_2024_or_later_at_late_stage"] == 325
    assert premature["founded_2023_or_later_at_late_stage"] == 574


def test_closed_companies_are_internally_incoherent(df: pd.DataFrame) -> None:
    coherence = audit.closed_company_coherence(df)
    assert coherence["closed_rows"] == 3453
    assert coherence["closed_grew_past_2023_peak"] == 1432
    assert coherence["closed_grew_past_2023_peak_pct"] == 41.5
    assert coherence["closed_min_current_headcount"] == 1
    assert coherence["closed_zero_layoffs_pct"] == 49.1
    assert coherence["independent_zero_layoffs_pct"] == 62.6


def test_sector_and_geography_carry_almost_no_signal(df: pd.DataFrame) -> None:
    by_domain = audit.closed_rate_spread(df, "Domain")
    assert by_domain["levels"] == 20
    assert by_domain["spread_pp"] == 4.82

    by_country = audit.closed_rate_spread(df, "Country")
    assert by_country["levels"] == 20
    assert by_country["spread_pp"] == 5.83


def test_ai_adoption_barely_moves_the_outcome(df: pd.DataFrame) -> None:
    effect = audit.ai_adoption_effect(df)
    assert effect["None"] == 15.84
    assert effect["AI-Native"] == 12.84
    assert effect["spread_pp"] == 3.00
    # Monotone but tiny — the ordering is real, the magnitude is not decision-relevant.
    ordered = [
        effect[level] for level in ["None", "Exploratory", "Moderate", "Advanced", "AI-Native"]
    ]
    assert ordered == sorted(ordered, reverse=True)


def test_runway_is_a_step_function_at_six_months(df: pd.DataFrame) -> None:
    cliff = audit.runway_cliff(df)
    assert cliff["closed_pct_below"] == 23.41
    assert cliff["closed_pct_at_or_above"] == 10.98
    assert cliff["rows_below"] == 5694


def test_runway_is_flat_either_side_of_the_cliff(df: pd.DataFrame) -> None:
    """The step is a threshold, not a gradient: no trend within each side."""
    bands = [(2, 4), (4, 6), (6, 9), (9, 14), (14, 50)]
    rates = []
    for low, high in bands:
        band = df.loc[(df["Runway_Months_2024"] >= low) & (df["Runway_Months_2024"] < high)]
        rates.append((band[audit.TARGET] == audit.CLOSED).mean() * 100)
    below = rates[:2]
    above = rates[2:]
    assert max(below) - min(below) < 2.0, "expected a flat plateau below the cliff"
    assert max(above) - min(above) < 2.0, "expected a flat plateau above the cliff"
    assert min(below) - max(above) > 10.0, "expected a step of more than 10pp at the cliff"


def test_size_measures_are_near_collinear(df: pd.DataFrame) -> None:
    correlations = audit.log_correlations(df)
    assert correlations["Total_Funding_USD_Millions~Valuation_USD_Millions"] == 0.974
    assert correlations["Total_Funding_USD_Millions~Revenue_ARR_Millions"] == 0.968
    assert correlations["Revenue_ARR_Millions~Valuation_USD_Millions"] == 0.933
    assert correlations["Monthly_Burn_Rate_Millions~Revenue_ARR_Millions"] == 0.944
    assert correlations["Peak_Headcount_2023~Current_Headcount_2026"] == 0.989


def test_value_sanity(df: pd.DataFrame) -> None:
    sanity = audit.value_sanity(df)
    assert sanity["valuation_below_funding_rows"] == 0
    assert sanity["layoffs_exceed_peak_headcount_rows"] == 0
    assert sanity["zero_revenue_rows"] == 152
    assert sanity["burn_exceeds_revenue_rows"] == 4799
    assert sanity["valuation_revenue_multiple_p50"] == 31.6
    assert sanity["valuation_revenue_multiple_p90"] == 131.4
    assert sanity["valuation_revenue_multiple_p99"] == 354.6
    assert sanity["founded_before_2020_pct"] == 45.5


def test_run_all_covers_every_check(df: pd.DataFrame) -> None:
    results = audit.run_all(df)
    assert results["shape"] == {"rows": 25_000, "columns": 17}
    assert results["city_country_nesting"] == {"cities_in_multiple_countries": 0}
    assert results["missingness"] == {"AI_Adoption_Level": 2434}
