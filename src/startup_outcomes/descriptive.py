"""Tier 1 — descriptive: what the data says about itself.

A peer of :mod:`startup_outcomes.audit` in every respect: pandas and numpy only, no
estimator, assertion-free, flat JSON-clean dicts. ``audit.py`` asks whether the data is
*coherent*; this module asks what it *contains* — distributions, ranges, group summaries,
and how much a cleaning decision moves the answer.

Everything here runs on all 25,000 rows. That is deliberate and it is the boundary
against every other tier: the README's "Dataset at a glance" describes the file as
downloaded, while the diagnostic and modelling tiers must first drop the 533 rows where
``Funding_Stage`` gives the target away. Mixing the two would put two different row counts
behind adjacent numbers in the same README.

Table-shaped results are returned as DataFrames because that is what they are; each has a
flat-dict companion (:func:`numeric_ranges` beside :func:`numeric_profile`) holding the
values the README quotes, so a documented number is still pinned by a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from startup_outcomes import audit
from startup_outcomes.config import RAW_CSV, RAW_CSV_SHA256, TARGET
from startup_outcomes.features import INCOMPLETE_COLUMN, MISSING_LEVEL
from startup_outcomes.load import (
    CATEGORICAL,
    FLOAT_COLUMNS,
    INT_COLUMNS,
    load_raw,
    sha256,
)

#: Row blocks the missingness map is binned into. The workbooks heatmap the raw null mask,
#: which at 25,000 rows tall renders as a solid smear; 200 blocks of ~125 rows still shows
#: stripe or block structure while fitting on a screen.
MISSINGNESS_ROW_BINS = 200

#: |z| beyond which a value is counted as an outlier by the z-score rule.
Z_SCORE_LIMIT = 3.0

#: Multiplier on the interquartile range for the Tukey fence.
IQR_FENCE_MULTIPLIER = 1.5

#: Numeric columns, in the contract's order.
NUMERIC_COLUMNS = [*FLOAT_COLUMNS, *INT_COLUMNS]

#: Columns the missingness of ``AI_Adoption_Level`` is cross-tabulated against.
STRATA_COLUMNS = ["Domain", "Funding_Stage", "Country", "Investor_Tier"]


def canonical_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """All 25,000 rows, as downloaded. The frame this tier describes."""
    return df if df is not None else load_raw()


def source_profile(path: Path | None = None) -> dict[str, object]:
    """Physical facts about the file on disk.

    The byte count is quoted in the README and had nothing computing it until now.
    """
    csv_path = path or RAW_CSV
    return {
        "file_name": csv_path.name,
        "bytes": int(csv_path.stat().st_size),
        "sha256_matches_reference": bool(sha256(csv_path) == RAW_CSV_SHA256),
    }


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column distribution table: the workbooks' ``describe`` plus median, IQR, skew.

    ``describe`` alone hides exactly what matters on this data. Every financial column
    here is heavily right-skewed, so its mean is not a typical value and its standard
    deviation exceeds it — both visible only once median and skew sit in the same table.
    """
    numeric = df.loc[:, NUMERIC_COLUMNS]
    described = numeric.describe().T
    described["median"] = numeric.median()
    described["IQR"] = numeric.quantile(0.75) - numeric.quantile(0.25)
    described["skew"] = numeric.skew()
    described["std_exceeds_mean"] = described["std"] > described["mean"]
    return described.round(2)


def numeric_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The min / median / max the README's column table quotes, one entry per column."""
    profile = numeric_profile(df)
    return {
        str(column): {
            "min": round(float(profile.loc[column, "min"]), 2),
            "median": round(float(profile.loc[column, "median"]), 2),
            "max": round(float(profile.loc[column, "max"]), 2),
        }
        for column in profile.index
    }


def category_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Level count and dominant level for every categorical column.

    The dominant-level share is the number the workbooks insist on: a column where one
    level holds over 90% of rows cannot explain much, whatever else is true of it.
    """
    rows = []
    for column in CATEGORICAL:
        counts = df[column].value_counts(dropna=True)
        rows.append(
            {
                "column": column,
                "levels": int(counts.size),
                "largest_level": str(counts.index[0]),
                "largest_level_rows": int(counts.iloc[0]),
                "largest_level_share_pct": round(float(counts.iloc[0] / len(df) * 100), 2),
                "blank_rows": int(df[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def dominant_levels(df: pd.DataFrame) -> dict[str, object]:
    """Flat companion to :func:`category_profile` for the levels the README names."""
    profile = category_profile(df)
    return {
        f"{column}_largest": f"{profile.loc[column, 'largest_level']}"
        f" ({int(profile.loc[column, 'largest_level_rows'])})"
        for column in profile.index
    }


def founding_year_profile(df: pd.DataFrame) -> dict[str, int | float]:
    """When these companies were founded, and where the distribution peaks."""
    counts = df["Founding_Year"].value_counts().sort_index()
    peak_year = int(counts.idxmax())
    return {
        "min_year": int(counts.index.min()),
        "max_year": int(counts.index.max()),
        "peak_year": peak_year,
        "peak_year_rows": int(counts.loc[peak_year]),
        "founded_before_2020_pct": round(float((df["Founding_Year"] < 2020).mean() * 100), 1),
    }


def missingness_by_strata(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Blank rate of ``AI_Adoption_Level`` across the levels of another column.

    A uniform rate is the evidence for treating blanks as missing completely at random.
    This function exists because the README asserted uniformity with nothing computing it;
    :func:`startup_outcomes.diagnostic.missingness_permutation` decides whether the spread
    it finds is larger than chance.
    """
    blank = df[INCOMPLETE_COLUMN].isna()
    rates = blank.groupby(df[column], observed=True).mean() * 100
    return {
        "stratified_by": column,
        "levels": int(rates.size),
        "min_pct": round(float(rates.min()), 2),
        "max_pct": round(float(rates.max()), 2),
        "spread_pp": round(float(rates.max() - rates.min()), 2),
        "lowest_level": str(rates.idxmin()),
        "highest_level": str(rates.idxmax()),
    }


def missingness_map_counts(df: pd.DataFrame, bins: int = MISSINGNESS_ROW_BINS) -> pd.DataFrame:
    """Blank share per block of consecutive rows, for the missingness chart.

    Structure in row order — a stripe, a contiguous block — would say the blanks were
    introduced by a process rather than at random, which is worth seeing before modelling.
    """
    blank = df[INCOMPLETE_COLUMN].isna().to_numpy(dtype=float)
    block = np.minimum((np.arange(len(blank)) / len(blank) * bins).astype(int), bins - 1)
    totals = np.bincount(block, minlength=bins)
    hits = np.bincount(block, weights=blank, minlength=bins)
    return pd.DataFrame(
        {
            "block": np.arange(bins),
            "rows": totals.astype(int),
            "blank_pct": np.round(np.where(totals > 0, hits / totals * 100, np.nan), 2),
        }
    ).set_index("block")


def imputation_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Five ways to handle the blanks, and the five different answers they give.

    This is the workbooks' signature move, moved to a real subject. Their version runs on
    a numeric column, which would be vacuous here — ``AI_Adoption_Level`` is the only
    column with blanks and it is categorical. So the comparison is over the headline the
    blanks actually threaten: the closed-rate gap from ``None`` to ``AI-Native``.

    Nothing is mutated; each row of the returned table is a separate hypothetical.
    """
    order = ["None", "Exploratory", "Moderate", "Advanced", "AI-Native"]

    def gap_of(frame: pd.DataFrame) -> tuple[float, float, int, int]:
        rates = audit.closed_rate_by(frame, INCOMPLETE_COLUMN)
        present = [level for level in order if level in rates.index]
        gap = float(rates[present[0]] - rates[present[-1]])
        overall = float((frame[TARGET] == audit.CLOSED).mean() * 100)
        return gap, overall, int(len(frame)), int(frame[INCOMPLETE_COLUMN].nunique(dropna=True))

    blank = df[INCOMPLETE_COLUMN].isna()
    explicit = df.copy()
    explicit[INCOMPLETE_COLUMN] = (
        explicit[INCOMPLETE_COLUMN].cat.add_categories([MISSING_LEVEL]).fillna(MISSING_LEVEL)
    )

    global_mode = df.loc[~blank, INCOMPLETE_COLUMN].mode().iloc[0]
    mode_filled = df.copy()
    mode_filled[INCOMPLETE_COLUMN] = mode_filled[INCOMPLETE_COLUMN].fillna(global_mode)

    per_domain = df.groupby("Domain", observed=True)[INCOMPLETE_COLUMN].agg(
        lambda levels: levels.mode().iloc[0]
    )
    domain_filled = df.copy()
    domain_filled[INCOMPLETE_COLUMN] = domain_filled[INCOMPLETE_COLUMN].fillna(
        df["Domain"].map(per_domain).astype(df[INCOMPLETE_COLUMN].dtype)
    )

    strategies = {
        "leave blank (grouped out)": df,
        f"explicit {MISSING_LEVEL} level": explicit,
        f"fill with global mode ({global_mode})": mode_filled,
        "fill with per-domain mode": domain_filled,
        "drop the rows": df.loc[~blank],
    }

    rows = []
    for label, frame in strategies.items():
        gap, overall, count, levels = gap_of(frame)
        rows.append(
            {
                "strategy": label,
                "rows": count,
                "levels": levels,
                "closed_rate_pct": round(overall, 2),
                "none_vs_ai_native_gap_pp": round(gap, 2),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def imputation_sensitivity(df: pd.DataFrame) -> dict[str, object]:
    """Flat companion to :func:`imputation_strategies`: how far the headline can move."""
    table = imputation_strategies(df)
    gaps = table["none_vs_ai_native_gap_pp"]
    return {
        "strategies": int(len(table)),
        "gap_min_pp": round(float(gaps.min()), 2),
        "gap_max_pp": round(float(gaps.max()), 2),
        "gap_range_pp": round(float(gaps.max() - gaps.min()), 2),
        "gap_at_explicit_missing_pp": round(float(gaps.loc[f"explicit {MISSING_LEVEL} level"]), 2),
        "rows_lost_by_dropping": int(len(df) - int(table.loc["drop the rows", "rows"])),
    }


def outlier_report(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Outlier counts by both conventional rules, and where they sit.

    Reporting both is the point: the two rules disagree, and on a right-skewed column the
    Tukey fence flags far more. Neither number is a decision — the last field is, because
    outliers concentrated in one category are not an outlier problem, they are a finding.
    """
    values = df[column]
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    fence_low = q1 - IQR_FENCE_MULTIPLIER * spread
    fence_high = q3 + IQR_FENCE_MULTIPLIER * spread
    by_fence = (values < fence_low) | (values > fence_high)

    deviation = values.std()
    by_z = (
        ((values - values.mean()).abs() / deviation > Z_SCORE_LIMIT)
        if deviation > 0
        else pd.Series(False, index=values.index)
    )

    domains = df.loc[by_fence, "Domain"].value_counts()
    return {
        "column": column,
        "iqr_fence_low": round(float(fence_low), 2),
        "iqr_fence_high": round(float(fence_high), 2),
        "iqr_outlier_rows": int(by_fence.sum()),
        "iqr_outlier_pct": round(float(by_fence.mean() * 100), 2),
        "z_outlier_rows": int(by_z.sum()),
        "z_outlier_pct": round(float(by_z.mean() * 100), 2),
        "top_domain_among_outliers": str(domains.index[0]) if len(domains) else "",
        "top_domain_share_pct": (
            round(float(domains.iloc[0] / by_fence.sum() * 100), 2) if by_fence.sum() else 0.0
        ),
    }


def duplicate_report(df: pd.DataFrame) -> dict[str, object]:
    """Duplicate rows, and whether the categorical levels need cleaning.

    The workbooks *apply* ``strip().title()`` to every category at this point. Here that
    would silently rename levels the data contract pins and destroy the declared ``category``
    dtype, so this verifies instead: if a level ever arrives with stray whitespace or
    inconsistent case, this reports it and the loader is what gets fixed.
    """
    untidy = {}
    for column in CATEGORICAL:
        levels = [str(level) for level in df[column].cat.categories]
        offenders = [level for level in levels if level != level.strip()]
        collisions = len(levels) - len({level.strip().casefold() for level in levels})
        if offenders or collisions:
            untidy[column] = {"untrimmed": offenders, "case_collisions": int(collisions)}
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns=["Company_ID"]).duplicated().sum()),
        "columns_needing_cleaning": untidy,
        "categories_are_clean": not untidy,
    }


def group_summary(df: pd.DataFrame, measure: str, by: str) -> pd.DataFrame:
    """The tier's main deliverable table: a measure summarised within a category.

    ``share_of_rows_pct`` is beside the statistics rather than derivable from them because
    every group statistic in this project has to be read against its ``n``.
    """
    summary = (
        df.groupby(by, observed=True)[measure]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_values("mean", ascending=False)
    )
    summary["share_of_rows_pct"] = (summary["n"] / summary["n"].sum() * 100).round(1)
    return summary.round(2)


def distribution_shape(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Whether a column's mean is safe to quote."""
    values = df[column]
    mean, median = float(values.mean()), float(values.median())
    return {
        "column": column,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(float(values.std()), 2),
        "skew": round(float(values.skew()), 2),
        "mean_exceeds_median_pct": round((mean - median) / median * 100, 1) if median else 0.0,
        "heavily_skewed": bool(abs(float(values.skew())) > 1.0),
        "std_exceeds_mean": bool(float(values.std()) > mean),
    }


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every descriptive check, keyed by name."""
    data = canonical_frame(df)
    return {
        "rows": int(len(data)),
        "source_profile": source_profile(),
        "numeric_ranges": numeric_ranges(data),
        "dominant_levels": dominant_levels(data),
        "founding_year_profile": founding_year_profile(data),
        "missingness_by_domain": missingness_by_strata(data, "Domain"),
        "missingness_by_funding_stage": missingness_by_strata(data, "Funding_Stage"),
        "imputation_sensitivity": imputation_sensitivity(data),
        "outliers_in_funding": outlier_report(data, "Total_Funding_USD_Millions"),
        "outliers_in_runway": outlier_report(data, "Runway_Months_2024"),
        "duplicate_report": duplicate_report(data),
        "shape_of_funding": distribution_shape(data, "Total_Funding_USD_Millions"),
        "shape_of_runway": distribution_shape(data, "Runway_Months_2024"),
    }
