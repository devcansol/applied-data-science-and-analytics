"""Tests for the design-matrix constructor and the engineered columns.

These describe the CODE. The point of most of them is that the leakage contract holds
structurally — not that it happens to hold for the columns present today.
"""

from __future__ import annotations

import pandas as pd
import pytest

from startup_outcomes import config, features


def test_the_canonical_frame_drops_the_determining_stage(
    raw: pd.DataFrame, analysis: pd.DataFrame
) -> None:
    assert len(raw) == 25_000
    assert len(analysis) == 24_467
    assert config.LEAKY_STAGE not in set(analysis["Funding_Stage"].cat.categories)


def test_the_design_matrix_is_exactly_the_declared_feature_list(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    features_frame, _ = design
    assert list(features_frame.columns) == config.model_features()
    assert features_frame.shape == (24_467, 13)


def test_the_conservative_variant_drops_the_undated_columns(analysis: pd.DataFrame) -> None:
    conservative, _ = features.design_matrix(analysis, include_undated=False)
    assert list(conservative.columns) == config.model_features(include_undated=False)
    assert conservative.shape[1] == 7
    assert not set(conservative.columns) & set(config.UNDATED)


def test_no_excluded_column_reaches_the_default_design_matrix(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    features_frame, _ = design
    config.assert_no_leakage(features_frame.columns)
    assert not set(features_frame.columns) & set(config.EXCLUDED_LEAKY)


def test_the_excluded_columns_are_reachable_only_by_asking_for_them() -> None:
    """And what you get back is precisely what the guard rejects."""
    controlled = features.feature_columns()
    demonstration = features.feature_columns(include_excluded=True)
    assert set(demonstration) - set(controlled) == set(config.EXCLUDED_LEAKY)
    with pytest.raises(ValueError, match="Current_Headcount_2026"):
        config.assert_no_leakage(demonstration)


def test_the_binary_target_matches_the_documented_base_rate(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    _, target = design
    assert target.name == "closed"
    assert set(target.unique()) == {0, 1}
    # 14.11% on the 24,467 analysis rows, against 13.81% on all 25,000: dropping the
    # 533 stage-IPO rows removes only non-closed outcomes, so the rate rises slightly.
    assert round(float(target.mean()) * 100, 2) == 14.11


def test_the_missing_level_is_a_relabelling_not_an_imputation(analysis: pd.DataFrame) -> None:
    before = analysis[features.INCOMPLETE_COLUMN]
    after = features.with_explicit_missing_level(analysis)[features.INCOMPLETE_COLUMN]

    blanks = int(before.isna().sum())
    assert blanks > 0
    assert int(after.isna().sum()) == 0
    assert int((after == features.MISSING_LEVEL).sum()) == blanks
    # Every level that existed keeps exactly the rows it had; nothing was moved.
    for level in before.cat.categories:
        assert int((after == level).sum()) == int((before == level).sum())


def test_the_missing_level_survives_being_applied_twice(analysis: pd.DataFrame) -> None:
    once = features.with_explicit_missing_level(analysis)
    twice = features.with_explicit_missing_level(once)
    assert list(once[features.INCOMPLETE_COLUMN].cat.categories) == list(
        twice[features.INCOMPLETE_COLUMN].cat.categories
    )


def test_the_design_matrix_keeps_categoricals_as_categories(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Gradient boosting handles them natively only if the dtype survives."""
    features_frame, _ = design
    categorical = [
        column
        for column in features_frame.columns
        if isinstance(features_frame[column].dtype, pd.CategoricalDtype)
    ]
    assert len(categorical) == 6
    assert features_frame[categorical].isna().sum().sum() == 0


def test_bands_are_labelled_in_words(analysis: pd.DataFrame) -> None:
    bands = features.make_bands(analysis["Runway_Months_2024"])
    assert list(bands.cat.categories) == features.BAND_NAMES[4]
    assert int(bands.isna().sum()) == 0


def test_an_unsplittable_column_raises_rather_than_returning_none() -> None:
    constant = pd.Series([1.0] * 100, name="constant")
    with pytest.raises(ValueError, match="two or more bands"):
        features.make_bands(constant)


def test_engineered_columns_are_added_without_touching_the_caller(
    analysis: pd.DataFrame,
) -> None:
    before = list(analysis.columns)
    result = features.engineered(analysis)
    assert list(analysis.columns) == before, "the caller's frame was mutated"
    added = [column for column in result.columns if column not in before]
    assert added == [
        "Company_Age_Years",
        "Funding_Share_Of_Domain_Pct",
        "Funding_Vs_Domain_Mean",
        "Above_Median_Funding",
        "Runway_Band",
        "Burn_To_Revenue_Ratio",
        "Valuation_To_Revenue_Multiple",
    ]


def test_the_ratio_columns_are_finite_where_revenue_exists(analysis: pd.DataFrame) -> None:
    """The 152 zero-revenue rows become NaN rather than infinity."""
    result = features.engineered(analysis)
    for column in ("Burn_To_Revenue_Ratio", "Valuation_To_Revenue_Multiple"):
        present = result[column].dropna()
        assert present.notna().all()
        assert bool((present.abs() < float("inf")).all())
    assert int(result["Burn_To_Revenue_Ratio"].isna().sum()) == 152


def test_the_automatic_leakage_scan_does_not_catch_this_dataset_s_leakage(
    analysis: pd.DataFrame, design: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The finding that justifies having a timing contract at all.

    ``mlkit``'s correlation guard flags a feature that almost *is* the target. Neither
    excluded column is: a 2026 headcount is a partial consequence of the outcome, not a
    restatement of it. So the scan is clean on the controlled matrix *and* on the
    deliberately leaky one — automatic detection would have found nothing to reject.
    """
    controlled_features, target = design
    assert features.leakage_scan(controlled_features, target)["any_suspect"] is False

    leaky_features, leaky_target = features.design_matrix(analysis, include_excluded=True)
    assert features.leakage_scan(leaky_features, leaky_target)["any_suspect"] is False
