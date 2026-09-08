"""Reproducible data-integrity audit.

Every quantitative claim in the README comes from a function here. ``run_all`` returns
them together so the README can be checked against the data in one command:

    uv run python -c "from startup_outcomes.audit import run_all; print(run_all())"

The checks are deliberately assertion-free — they report what the data says and leave
judgement to the reader. ``tests/test_audit.py`` is where the documented values are
pinned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startup_outcomes.config import LEAKY_STAGE, TARGET
from startup_outcomes.load import load_raw

#: Stages no company founded within a year or two could plausibly have reached.
LATE_STAGES = ["Series D", "Series E+", "Pre-IPO", "IPO"]

#: Where the closed-rate step change sits, in months of runway.
RUNWAY_CLIFF_MONTHS = 6.0

CLOSED = "Closed"
INDEPENDENT = "Independent"


def shape(df: pd.DataFrame) -> dict[str, int]:
    """Row and column counts."""
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def uniqueness(df: pd.DataFrame) -> dict[str, int]:
    """Distinct identifiers, and distinct rows once the identifier is ignored."""
    return {
        "unique_company_ids": int(df["Company_ID"].nunique()),
        "unique_rows_excluding_id": int(
            df.drop(columns=["Company_ID"]).astype(str).drop_duplicates().shape[0]
        ),
    }


def missingness(df: pd.DataFrame) -> dict[str, int]:
    """Null count per column, restricted to columns that have any."""
    counts = df.isna().sum()
    return {column: int(count) for column, count in counts.items() if count > 0}


def cardinalities(df: pd.DataFrame) -> dict[str, int]:
    """Distinct non-null values for each categorical column."""
    columns = [
        "Domain",
        "Country",
        "City",
        "Funding_Stage",
        "Investor_Tier",
        "AI_Adoption_Level",
        TARGET,
    ]
    return {column: int(df[column].nunique(dropna=True)) for column in columns}


def city_country_nesting(df: pd.DataFrame) -> dict[str, int]:
    """Cities appearing under more than one country.

    A clean hierarchy means ``City`` can be used as a nested feature without first
    disambiguating same-named cities.
    """
    per_city = df.groupby("City", observed=True)["Country"].nunique()
    return {"cities_in_multiple_countries": int((per_city > 1).sum())}


def target_base_rates(df: pd.DataFrame) -> dict[str, float]:
    """Outcome distribution, plus the accuracy a majority-class guess would score."""
    shares = df[TARGET].value_counts(normalize=True) * 100
    closed_share = float(shares.get(CLOSED, 0.0))
    return {
        **{str(outcome): round(float(share), 2) for outcome, share in shares.items()},
        "binary_closed_positive_pct": round(closed_share, 2),
        "binary_majority_baseline_accuracy_pct": round(100 - closed_share, 2),
    }


def stage_target_leakage(df: pd.DataFrame) -> dict[str, int | bool]:
    """Whether ``Funding_Stage == 'IPO'`` determines the outcome.

    If it does, the level hands the label to any model that sees it.
    """
    at_stage = df.loc[df["Funding_Stage"] == LEAKY_STAGE]
    same = at_stage.loc[at_stage[TARGET] == LEAKY_STAGE]
    return {
        "rows_at_stage_ipo": int(len(at_stage)),
        "of_which_outcome_ipo": int(len(same)),
        "stage_ipo_implies_outcome_ipo": bool(len(at_stage) == len(same)),
        "outcome_ipo_at_other_stages": int(
            len(df.loc[(df[TARGET] == LEAKY_STAGE) & (df["Funding_Stage"] != LEAKY_STAGE)])
        ),
    }


def premature_late_stage(df: pd.DataFrame) -> dict[str, int]:
    """Companies at a late funding stage sooner than that stage takes to reach."""
    late = df["Funding_Stage"].isin(LATE_STAGES)
    return {
        "founded_2024_or_later_at_late_stage": int((late & (df["Founding_Year"] >= 2024)).sum()),
        "founded_2023_or_later_at_late_stage": int((late & (df["Founding_Year"] >= 2023)).sum()),
    }


def closed_company_coherence(df: pd.DataFrame) -> dict[str, float | int]:
    """Whether companies recorded as ``Closed`` behave like closed companies."""
    closed = df.loc[df[TARGET] == CLOSED]
    independent = df.loc[df[TARGET] == INDEPENDENT]
    grew = int((closed["Current_Headcount_2026"] > closed["Peak_Headcount_2023"]).sum())
    return {
        "closed_rows": int(len(closed)),
        "closed_grew_past_2023_peak": grew,
        "closed_grew_past_2023_peak_pct": round(grew / len(closed) * 100, 1),
        "closed_min_current_headcount": int(closed["Current_Headcount_2026"].min()),
        "closed_zero_layoffs_pct": round(float((closed["Layoffs_2024_2025"] == 0).mean() * 100), 1),
        "independent_zero_layoffs_pct": round(
            float((independent["Layoffs_2024_2025"] == 0).mean() * 100), 1
        ),
    }


def _closed_rate_by(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby(column, observed=True)[TARGET].apply(lambda s: (s == CLOSED).mean() * 100)


def closed_rate_spread(df: pd.DataFrame, column: str) -> dict[str, float | str]:
    """Spread of closed rate across a categorical column.

    A spread no wider than sampling noise means the column carries no real signal
    about failure, whatever a model's feature importances may suggest.
    """
    rates = _closed_rate_by(df, column).sort_values()
    return {
        "column": column,
        "levels": int(len(rates)),
        "min_pct": round(float(rates.iloc[0]), 2),
        "max_pct": round(float(rates.iloc[-1]), 2),
        "spread_pp": round(float(rates.iloc[-1] - rates.iloc[0]), 2),
        "lowest_level": str(rates.index[0]),
        "highest_level": str(rates.index[-1]),
    }


def ai_adoption_effect(df: pd.DataFrame) -> dict[str, float]:
    """Closed rate at each AI-adoption level, ordered from no adoption to AI-native."""
    order = ["None", "Exploratory", "Moderate", "Advanced", "AI-Native"]
    rates = _closed_rate_by(df, "AI_Adoption_Level")
    present = [level for level in order if level in rates.index]
    result = {level: round(float(rates[level]), 2) for level in present}
    result["spread_pp"] = round(max(result.values()) - min(result.values()), 2)
    return result


def runway_cliff(df: pd.DataFrame) -> dict[str, float]:
    """Closed rate either side of the runway threshold."""
    below = df["Runway_Months_2024"] < RUNWAY_CLIFF_MONTHS
    return {
        "threshold_months": RUNWAY_CLIFF_MONTHS,
        "closed_pct_below": round(float((df.loc[below, TARGET] == CLOSED).mean() * 100), 2),
        "closed_pct_at_or_above": round(float((df.loc[~below, TARGET] == CLOSED).mean() * 100), 2),
        "rows_below": int(below.sum()),
    }


def log_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Pearson r between log1p-transformed size measures.

    These are all size proxies, so high correlation is expected; the point is how
    high, because it decides whether the features can be interpreted separately.
    """
    pairs = [
        ("Total_Funding_USD_Millions", "Valuation_USD_Millions"),
        ("Total_Funding_USD_Millions", "Revenue_ARR_Millions"),
        ("Revenue_ARR_Millions", "Valuation_USD_Millions"),
        ("Monthly_Burn_Rate_Millions", "Revenue_ARR_Millions"),
        ("Peak_Headcount_2023", "Current_Headcount_2026"),
    ]
    logged = {
        column: np.log1p(df[column].to_numpy(dtype=float))
        for column in {column for pair in pairs for column in pair}
    }
    return {
        f"{left}~{right}": round(float(np.corrcoef(logged[left], logged[right])[0, 1]), 3)
        for left, right in pairs
    }


def value_sanity(df: pd.DataFrame) -> dict[str, float | int]:
    """Range checks on the financial columns."""
    with_revenue = df.loc[df["Revenue_ARR_Millions"] > 0.1]
    multiples = with_revenue["Valuation_USD_Millions"] / with_revenue["Revenue_ARR_Millions"]
    return {
        "valuation_below_funding_rows": int(
            (df["Valuation_USD_Millions"] < df["Total_Funding_USD_Millions"]).sum()
        ),
        "zero_revenue_rows": int((df["Revenue_ARR_Millions"] == 0).sum()),
        "burn_exceeds_revenue_rows": int(
            (df["Monthly_Burn_Rate_Millions"] > df["Revenue_ARR_Millions"]).sum()
        ),
        "layoffs_exceed_peak_headcount_rows": int(
            (df["Layoffs_2024_2025"] > df["Peak_Headcount_2023"]).sum()
        ),
        "valuation_revenue_multiple_p50": round(float(multiples.quantile(0.50)), 1),
        "valuation_revenue_multiple_p90": round(float(multiples.quantile(0.90)), 1),
        "valuation_revenue_multiple_p99": round(float(multiples.quantile(0.99)), 1),
        "founding_year_min": int(df["Founding_Year"].min()),
        "founding_year_max": int(df["Founding_Year"].max()),
        "founded_before_2020_pct": round(float((df["Founding_Year"] < 2020).mean() * 100), 1),
    }


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Run every check and return the results keyed by check name."""
    data = df if df is not None else load_raw()
    return {
        "shape": shape(data),
        "uniqueness": uniqueness(data),
        "missingness": missingness(data),
        "cardinalities": cardinalities(data),
        "city_country_nesting": city_country_nesting(data),
        "target_base_rates": target_base_rates(data),
        "stage_target_leakage": stage_target_leakage(data),
        "premature_late_stage": premature_late_stage(data),
        "closed_company_coherence": closed_company_coherence(data),
        "closed_rate_by_domain": closed_rate_spread(data, "Domain"),
        "closed_rate_by_country": closed_rate_spread(data, "Country"),
        "ai_adoption_effect": ai_adoption_effect(data),
        "runway_cliff": runway_cliff(data),
        "log_correlations": log_correlations(data),
        "value_sanity": value_sanity(data),
    }
