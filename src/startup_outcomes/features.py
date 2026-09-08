"""Engineered columns and the project's only design-matrix constructor.

The reference workbooks build a design matrix by *dropping* columns — anything more than
25 categories, anything over half empty, anything that looks like an identifier. That
default is backwards: a newly added leaky column arrives *included*. Here the allow-list
comes from :func:`startup_outcomes.config.model_features` and nothing else, so a column
has to be classified in the timing contract before a model can see it.

The heuristic would also have been wrong on this data specifically: ``City`` has 58
levels, so a 25-level cap would silently discard a structural feature.

**Engineered columns are not model inputs.** :func:`engineered` exists for the descriptive
and diagnostic tiers, which need the group-relative views the workbooks build in their
feature-engineering section. Feeding them to a model would mean fitting group statistics
on all rows including the held-out fold, so :func:`design_matrix` ignores them and takes
``model_features()`` verbatim. The two ratio columns are the exception worth noting: the
collinearity decision observes that burn/revenue and valuation/revenue carry information
the raw levels do not, and they are available here for the diagnostic tier to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startup_outcomes.audit import CLOSED
from startup_outcomes.config import (
    EXCLUDED_LEAKY,
    TARGET,
    assert_no_leakage,
    model_features,
)
from startup_outcomes.load import drop_leaky_stage, load_raw

#: Explicit stand-in for a blank ``AI_Adoption_Level``. A relabelling, never an estimate.
MISSING_LEVEL = "(Missing)"

#: The column whose blanks that level covers — the only column in the dataset with any.
INCOMPLETE_COLUMN = "AI_Adoption_Level"

#: Band labels by band count, so a quartile split reads as words rather than intervals.
BAND_NAMES: dict[int, list[str]] = {
    2: ["low", "high"],
    3: ["low", "mid", "high"],
    4: ["low", "mid-low", "mid-high", "high"],
}

#: The year the outcome is observed, used to turn a founding year into an age.
OUTCOME_YEAR = 2026

#: Correlation with the target above which a single feature is treated as suspect.
LEAKAGE_CORRELATION_THRESHOLD = 0.95


def canonical_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The frame every modelling and diagnostic result is computed on.

    ``drop_leaky_stage`` is not optional here: these tiers use ``Funding_Stage``, and its
    ``IPO`` level determines the target outright. 24,467 of 25,000 rows survive.
    """
    data = df if df is not None else load_raw()
    return drop_leaky_stage(data)


def binary_target(df: pd.DataFrame) -> pd.Series:
    """The primary label: did this company close?

    The workbooks derive a target by median-splitting a measure because their datasets may
    not have one. This dataset has a real five-class outcome; the binary framing is the
    one the README's baseline (86.19% majority accuracy) refers to.
    """
    return (df[TARGET] == CLOSED).astype(int).rename("closed")


def with_explicit_missing_level(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank ``AI_Adoption_Level`` values with an explicit level.

    This is a relabelling, not an imputation: no value is estimated from other rows, so it
    is fold-independent and cannot leak. It is also load-bearing rather than cosmetic —
    ``sklearn.inspection.partial_dependence`` raises ``mixed data types`` on a categorical
    column holding NaN, and ``OneHotEncoder`` needs a category to encode.

    Mode-filling is the alternative and is not available: the blanks vary with ``Domain``
    beyond chance, so they are missing at random *conditional on domain*, and filling them
    with the global mode would move 2,434 rows into one level.
    """
    filled = df.copy()
    column = filled[INCOMPLETE_COLUMN]
    if MISSING_LEVEL not in column.cat.categories:
        column = column.cat.add_categories([MISSING_LEVEL])
    filled[INCOMPLETE_COLUMN] = column.fillna(MISSING_LEVEL)
    return filled


def make_bands(values: pd.Series, bands: int = 4) -> pd.Series:
    """Quantile bands of a numeric column, labelled in words.

    Falls back to fewer bands when ties make the requested count impossible, and raises
    rather than returning ``None`` when even two are unachievable — the workbook version
    returns ``None`` and every caller then needs a branch for it.
    """
    for count in range(bands, 1, -1):
        try:
            cut = pd.qcut(values, count, duplicates="drop")
        except (ValueError, IndexError):
            continue
        achieved = cut.cat.categories.size
        if achieved >= 2:
            labels = BAND_NAMES.get(achieved, [f"band{index + 1}" for index in range(achieved)])
            return cut.cat.rename_categories(labels)
    raise ValueError(
        f"{values.name} cannot be split into two or more bands: "
        f"{values.nunique(dropna=True)} distinct values."
    )


def engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add the group-relative and ratio columns the diagnostic tier reports.

    Returns a new frame; the caller's is untouched.
    """
    result = df.copy()
    funding = result["Total_Funding_USD_Millions"]
    domain_total = funding.groupby(result["Domain"], observed=True).transform("sum")
    domain_mean = funding.groupby(result["Domain"], observed=True).transform("mean")

    result["Company_Age_Years"] = OUTCOME_YEAR - result["Founding_Year"]
    result["Funding_Share_Of_Domain_Pct"] = funding / domain_total * 100
    result["Funding_Vs_Domain_Mean"] = funding - domain_mean
    result["Above_Median_Funding"] = (funding >= funding.median()).astype(int)
    result["Runway_Band"] = make_bands(result["Runway_Months_2024"])
    # Guarded against the 152 zero-revenue rows, which would otherwise be infinite.
    revenue = result["Revenue_ARR_Millions"].replace(0.0, np.nan)
    result["Burn_To_Revenue_Ratio"] = result["Monthly_Burn_Rate_Millions"] / revenue
    result["Valuation_To_Revenue_Multiple"] = result["Valuation_USD_Millions"] / revenue
    return result


def feature_columns(
    *,
    include_undated: bool = True,
    include_excluded: bool = False,
) -> list[str]:
    """The columns a design matrix may contain.

    Args:
        include_undated: Forwarded to ``model_features``; ``False`` gives the conservative
            variant built only from explicitly dated and structural columns.
        include_excluded: Append the outcome-contemporaneous columns. **Only the labelled
            leakage demonstration may pass this.** The names are never spelled here — they
            come from ``config.EXCLUDED_LEAKY``, which keeps the core-checklist grep
            meaningful and keeps one list authoritative.
    """
    columns = model_features(include_undated=include_undated)
    if include_excluded:
        return [*columns, *EXCLUDED_LEAKY]
    assert_no_leakage(columns)
    return columns


def design_matrix(
    df: pd.DataFrame,
    *,
    include_undated: bool = True,
    include_excluded: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``(X, y)`` for the binary closed-vs-rest problem.

    Categorical columns keep their ``category`` dtype so gradient boosting can handle them
    natively; the one-hot path encodes them inside a fold-fitted pipeline instead. No
    imputation happens here beyond the explicit missing level.
    """
    columns = feature_columns(include_undated=include_undated, include_excluded=include_excluded)
    frame = with_explicit_missing_level(df)
    return frame.loc[:, columns].copy(), binary_target(frame)


def leakage_scan(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    threshold: float = LEAKAGE_CORRELATION_THRESHOLD,
) -> dict[str, object]:
    """Flag any single column that almost *is* the target.

    The timing contract already excludes the columns known to be contemporaneous. This
    catches the other kind — a column that turns out to be computed from the answer — which
    no automatic rule can know about in advance. It reports; it does not raise.
    """
    numeric = features.select_dtypes(include=np.number)
    labels = target.to_numpy(dtype=float)
    suspects: dict[str, float] = {}
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        if np.all(np.isnan(values)) or np.nanstd(values) == 0:
            continue
        correlation = abs(float(np.corrcoef(values, labels)[0, 1]))
        if np.isfinite(correlation) and correlation >= threshold:
            suspects[column] = round(correlation, 3)
    return {
        "threshold": threshold,
        "numeric_columns_scanned": int(numeric.shape[1]),
        "suspects": suspects,
        "any_suspect": bool(suspects),
    }
