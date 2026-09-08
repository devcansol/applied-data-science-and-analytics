"""Tier 2 — diagnostic: why the outcome varies, and whether it varies at all.

The reference workbooks run this tier without a single interval: a group gap is reported
in standard-deviation units beside its ``n``, and the reader judges. That is honest as far
as it goes, but it cannot settle the question this dataset keeps posing — a 4.82 pp
closed-rate spread across 20 domains is *either* a weak real effect *or* exactly what
shuffling produces, and those call for opposite conclusions. So every effect here carries
either a bootstrap interval or a permutation null.

The choice between the two is not stylistic:

* A **bootstrap interval** goes on a *pre-specified* comparison — the ``None`` versus
  ``AI-Native`` gap, the two sides of the runway cliff. Both sides are named in advance,
  so the estimate is unbiased.
* A **permutation null** goes on a *spread across all levels*. Max-minus-min over 20 noisy
  estimates is positive even when every true rate is identical, so an interval around it
  would answer the wrong question. Shuffling the outcome at the observed group sizes
  answers the right one.

Everything runs on the 24,467-row canonical frame, because every function here may touch
``Funding_Stage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startup_outcomes import audit, intervals
from startup_outcomes.audit import CLOSED, RUNWAY_CLIFF_MONTHS, SIZE_PAIRS
from startup_outcomes.config import TARGET
from startup_outcomes.features import INCOMPLETE_COLUMN, canonical_frame
from startup_outcomes.load import load_raw

#: Adoption levels from no adoption to AI-native. The dtype is unordered, so the semantic
#: order has to be stated rather than read off the categories.
ADOPTION_ORDER = ["None", "Exploratory", "Moderate", "Advanced", "AI-Native"]

#: Smallest subgroup the reversal check will draw a conclusion from. Below this, a flipped
#: ordering is noise and reporting it would manufacture a paradox.
MIN_SUBGROUP_ROWS = 20

#: Smallest cell the two-way drill-down will display.
MIN_DRILLDOWN_ROWS = 5

#: Rule-of-thumb boundaries the workbooks use to describe an effect size in SD units.
SMALL_EFFECT, MODERATE_EFFECT, LARGE_EFFECT = 0.2, 0.5, 0.8


def _closed_flags(df: pd.DataFrame) -> pd.Series:
    """Boolean closed indicator, the outcome every function here explains."""
    return df[TARGET] == CLOSED


def group_gap(df: pd.DataFrame, measure: str, by: str) -> dict[str, object]:
    """Top-versus-bottom group gap in a numeric measure, expressed three ways.

    The SD-unit figure is the one the workbooks call honest, and the reason is visible on
    this data: a gap can look large in raw units purely because the measure is skewed.
    The interval is on the gap between the two *named* extreme groups, so it is
    conditional on which groups those turned out to be — stated here rather than hidden.
    """
    summary = df.groupby(by, observed=True)[measure].agg(["count", "mean"])
    eligible = summary.loc[summary["count"] >= MIN_SUBGROUP_ROWS].sort_values("mean")
    lowest, highest = str(eligible.index[0]), str(eligible.index[-1])
    gap = float(eligible["mean"].iloc[-1] - eligible["mean"].iloc[0])
    deviation = float(df[measure].std())
    overall = float(df[measure].mean())
    gap_in_sd = gap / deviation if deviation else 0.0
    return {
        "measure": measure,
        "grouped_by": by,
        "levels_compared": int(len(eligible)),
        "highest_level": highest,
        "lowest_level": lowest,
        "gap_absolute": round(gap, 2),
        "gap_as_pct_of_overall_mean": round(gap / overall * 100, 1) if overall else 0.0,
        "gap_in_sd_units": round(gap_in_sd, 3),
        "effect_size_label": _effect_label(abs(gap_in_sd)),
        "rows_in_highest": int(eligible["count"].iloc[-1]),
        "rows_in_lowest": int(eligible["count"].iloc[0]),
    }


def _effect_label(magnitude: float) -> str:
    """Translate an SD-unit magnitude into the workbooks' vocabulary."""
    if magnitude < SMALL_EFFECT:
        return "negligible"
    if magnitude < MODERATE_EFFECT:
        return "small"
    if magnitude < LARGE_EFFECT:
        return "moderate"
    return "large"


def ai_adoption_gap(df: pd.DataFrame) -> dict[str, object]:
    """Interval on the closed-rate gap from no AI adoption to AI-native.

    This is the dataset's headline premise, and the pre-specified pair makes it the one
    place a bootstrap interval is exactly the right instrument. The README previously
    called the effect absent; an interval is what shows that "absent" is the wrong word
    for it — the honest reading is small, real, and far too small to act on.
    """
    return intervals.rate_gap_interval(
        df[INCOMPLETE_COLUMN],
        _closed_flags(df),
        high=ADOPTION_ORDER[0],
        low=ADOPTION_ORDER[-1],
        name="closed_rate_gap_none_vs_ai_native",
    )


def runway_cliff_effect(df: pd.DataFrame) -> dict[str, object]:
    """Interval on the closed-rate step either side of the six-month runway threshold.

    Also pre-specified — the threshold comes from ``audit.RUNWAY_CLIFF_MONTHS``, which was
    located before any model existed.
    """
    return intervals.rate_difference_interval(
        df["Runway_Months_2024"] < RUNWAY_CLIFF_MONTHS,
        _closed_flags(df),
        name="closed_rate_below_vs_above_runway_cliff",
    )


def spread_permutation(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Whether a column's closed-rate spread exceeds what shuffling produces.

    The test that decides whether "no signal" is a finding or a failure to look. A spread
    that sits inside the null is not a weak effect; it is an absent one.
    """
    return intervals.permutation_spread(
        df[column], _closed_flags(df), name=f"closed_rate_spread_by_{column.lower()}"
    )


def missingness_permutation(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Whether blank ``AI_Adoption_Level`` depends on another column beyond chance.

    Decides the imputation question. If the blanks are missing completely at random, the
    convenient fills are defensible; if they depend on ``Domain``, they are not, because
    filling with a global mode would then distort the domains unevenly.
    """
    return intervals.permutation_spread(
        df[column],
        df[INCOMPLETE_COLUMN].isna(),
        name=f"blank_rate_spread_by_{column.lower()}",
    )


def correlation_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson beside Spearman for each size-proxy pair, raw and on logs.

    The workbooks' rule: a large gap between the two means the relationship is real but
    curved. Here the gap is informative in the other direction — these pairs agree closely
    once logged, which is what makes them one collinear block rather than four features.
    """
    rows = []
    for left, right in SIZE_PAIRS:
        raw_left, raw_right = df[left], df[right]
        log_left = np.log1p(raw_left.to_numpy(dtype=float))
        log_right = np.log1p(raw_right.to_numpy(dtype=float))
        rows.append(
            {
                "pair": f"{left}~{right}",
                "pearson_raw": round(float(raw_left.corr(raw_right)), 3),
                "pearson_log": round(float(np.corrcoef(log_left, log_right)[0, 1]), 3),
                "spearman": round(float(raw_left.corr(raw_right, method="spearman")), 3),
            }
        )
    table = pd.DataFrame(rows).set_index("pair")
    # Spearman is rank-based, so it is unchanged by the log; the gap worth reading is
    # between Pearson on raw values and Pearson on logs.
    table["log_lifts_pearson_by"] = (table["pearson_log"] - table["pearson_raw"]).round(3)
    return table


def collinearity_summary(df: pd.DataFrame) -> dict[str, object]:
    """Flat companion to :func:`correlation_comparison`."""
    table = correlation_comparison(df)
    return {
        "pairs": int(len(table)),
        "min_pearson_log": round(float(table["pearson_log"].min()), 3),
        "max_pearson_log": round(float(table["pearson_log"].max()), 3),
        "min_spearman": round(float(table["spearman"].min()), 3),
        "all_log_pairs_above_0_9": bool((table["pearson_log"] > 0.9).all()),
        "largest_log_lift": round(float(table["log_lifts_pearson_by"].max()), 3),
    }


def crosstab_rates(df: pd.DataFrame, row: str, column: str) -> pd.DataFrame:
    """Row-normalised cross-tabulation of two categorical columns, as percentages."""
    counts = pd.crosstab(df[row], df[column])
    return (counts.div(counts.sum(axis=1), axis=0) * 100).round(2)


def drilldown(df: pd.DataFrame, primary: str, secondary: str) -> pd.DataFrame:
    """Closed rate in every combination of two categories, with its cell size.

    Where the average hides the story, per the workbooks. Cells below
    ``MIN_DRILLDOWN_ROWS`` are dropped rather than shown, because a rate on four rows
    invites a conclusion it cannot support.
    """
    grouped = df.groupby([primary, secondary], observed=True)[TARGET]
    table = grouped.agg(
        n="count",
        closed_pct=lambda outcomes: (outcomes == CLOSED).mean() * 100,
    ).reset_index()
    kept = table.loc[table["n"] >= MIN_DRILLDOWN_ROWS].copy()
    kept["closed_pct"] = kept["closed_pct"].round(2)
    return kept.sort_values("closed_pct", ascending=False).set_index([primary, secondary])


def reversal_check(
    df: pd.DataFrame,
    primary: str,
    secondary: str,
    *,
    min_rows: int = MIN_SUBGROUP_ROWS,
) -> dict[str, object]:
    """Does the ordering of ``primary`` by closed rate survive inside ``secondary``?

    The workbooks call this the most important cell in the whole exercise, and on this
    data it earns the billing. A headline ordering that reverses inside a subgroup is
    Simpson's paradox, and quoting the headline without checking is how an analysis
    becomes confidently wrong.

    Reversal is measured as a negative Spearman correlation between the overall ordering
    of ``primary`` levels and the ordering within a subgroup.
    """
    overall = audit.closed_rate_by(df, primary)
    subgroup_rows = []
    for level, part in df.groupby(secondary, observed=True):
        if len(part) < min_rows:
            continue
        rates = audit.closed_rate_by(part, primary)
        shared = [index for index in overall.index if index in rates.index]
        if len(shared) < 3:
            continue
        correlation = float(overall.loc[shared].corr(rates.loc[shared], method="spearman"))
        subgroup_rows.append(
            {
                "level": str(level),
                "rows": int(len(part)),
                "rank_correlation": round(correlation, 3),
                "reversed": bool(correlation < 0),
            }
        )

    if not subgroup_rows:
        return {
            "primary": primary,
            "secondary": secondary,
            "subgroups_checked": 0,
            "ordering_flips": False,
        }

    table = pd.DataFrame(subgroup_rows).sort_values("rank_correlation")
    return {
        "primary": primary,
        "secondary": secondary,
        "subgroups_checked": int(len(table)),
        "subgroups_reversed": int(table["reversed"].sum()),
        "most_reversed_level": str(table["level"].iloc[0]),
        "most_reversed_rows": int(table["rows"].iloc[0]),
        "most_reversed_correlation": round(float(table["rank_correlation"].iloc[0]), 3),
        "ordering_flips": bool(table["reversed"].any()),
    }


def reversal_table(df: pd.DataFrame, primary: str, secondary: str) -> pd.DataFrame:
    """Per-subgroup closed rates behind :func:`reversal_check`, for plotting."""
    rows = []
    for level, part in df.groupby(secondary, observed=True):
        if len(part) < MIN_SUBGROUP_ROWS:
            continue
        rates = audit.closed_rate_by(part, primary)
        for index, rate in rates.items():
            rows.append(
                {
                    secondary: str(level),
                    primary: str(index),
                    "n": int((part[primary] == index).sum()),
                    "closed_pct": round(float(rate), 2),
                }
            )
    return pd.DataFrame(rows).pivot_table(
        index=secondary, columns=primary, values="closed_pct", aggfunc="first"
    )


def adoption_gap_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """The ``None`` minus ``AI-Native`` closed-rate gap inside each investor tier.

    A sharper reading of the reversal than a rank correlation: this is the sign of the
    headline effect, subgroup by subgroup. Where it goes negative, AI-native companies
    close *more often* than non-adopters.
    """
    rows = []
    for level, part in df.groupby("Investor_Tier", observed=True):
        rates = audit.closed_rate_by(part, INCOMPLETE_COLUMN)
        if not {ADOPTION_ORDER[0], ADOPTION_ORDER[-1]} <= set(rates.index):
            continue
        gap = float(rates[ADOPTION_ORDER[0]] - rates[ADOPTION_ORDER[-1]])
        rows.append(
            {
                "Investor_Tier": str(level),
                "rows": int(len(part)),
                "none_closed_pct": round(float(rates[ADOPTION_ORDER[0]]), 2),
                "ai_native_closed_pct": round(float(rates[ADOPTION_ORDER[-1]]), 2),
                "gap_pp": round(gap, 2),
                "gap_has_expected_sign": bool(gap > 0),
            }
        )
    return pd.DataFrame(rows).set_index("Investor_Tier").sort_values("gap_pp")


def adoption_reversal(df: pd.DataFrame) -> dict[str, object]:
    """Flat companion to :func:`adoption_gap_by_tier`: where the premise inverts."""
    table = adoption_gap_by_tier(df)
    against = table.loc[~table["gap_has_expected_sign"]]
    return {
        "tiers_checked": int(len(table)),
        "tiers_against_expected_sign": int(len(against)),
        "strongest_reversal_tier": str(table.index[0]),
        "strongest_reversal_gap_pp": round(float(table["gap_pp"].iloc[0]), 2),
        "strongest_reversal_rows": int(table["rows"].iloc[0]),
        "gap_sign_flips_somewhere": bool(len(against) > 0),
    }


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every diagnostic check, keyed by name.

    Two frames appear here on purpose. Everything about the *outcome* uses the 24,467-row
    canonical frame, because ``Funding_Stage == 'IPO'`` would otherwise determine the
    label. The missingness checks use all 25,000 rows, because blank
    ``AI_Adoption_Level`` is a property of the file as downloaded and pairs with the range
    reported by ``descriptive.missingness_by_strata`` — running the two on different row
    counts is how a README ends up quoting 8.38 pp and 8.59 pp for the same claim.
    """
    source = df if df is not None else load_raw()
    data = canonical_frame(source)
    return {
        "rows": int(len(data)),
        "missingness_rows": int(len(source)),
        "ai_adoption_gap": ai_adoption_gap(data),
        "runway_cliff_effect": runway_cliff_effect(data),
        "domain_spread_permutation": spread_permutation(data, "Domain"),
        "country_spread_permutation": spread_permutation(data, "Country"),
        "adoption_spread_permutation": spread_permutation(data, INCOMPLETE_COLUMN),
        "missingness_by_domain_permutation": missingness_permutation(source, "Domain"),
        "missingness_by_stage_permutation": missingness_permutation(source, "Funding_Stage"),
        "runway_gap_by_investor_tier": group_gap(data, "Runway_Months_2024", "Investor_Tier"),
        "funding_gap_by_domain": group_gap(data, "Total_Funding_USD_Millions", "Domain"),
        "collinearity_summary": collinearity_summary(data),
        "adoption_reversal": adoption_reversal(data),
        "adoption_reversal_within_tier": reversal_check(data, INCOMPLETE_COLUMN, "Investor_Tier"),
        "domain_reversal_within_stage": reversal_check(data, "Domain", "Funding_Stage"),
    }
