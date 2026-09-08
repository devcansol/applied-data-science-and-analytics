"""Pins the descriptive findings quoted in the README.

If a number here changes, the README is now wrong. Update both together.
"""

from __future__ import annotations

import pandas as pd

from startup_outcomes import audit, descriptive
from startup_outcomes.features import MISSING_LEVEL


def test_the_source_file_is_the_reference_download() -> None:
    profile = descriptive.source_profile()
    assert profile["bytes"] == 3_099_505
    assert profile["sha256_matches_reference"] is True


def test_the_dominant_levels_the_readme_names(raw: pd.DataFrame) -> None:
    levels = descriptive.dominant_levels(raw)
    assert levels["Domain_largest"] == "Generative AI (3727)"
    assert levels["Country_largest"] == "United States (7393)"
    assert levels["Acquisition_Status_largest"] == "Independent (15765)"


def test_no_single_level_dominates_enough_to_explain_nothing(raw: pd.DataFrame) -> None:
    """The workbooks' rule of thumb: a column over 90% one level cannot explain much.

    Only the target comes close, at 63%, which is the imbalance the baseline exists for.
    """
    profile = descriptive.category_profile(raw)
    assert float(profile["largest_level_share_pct"].max()) < 90.0
    assert float(profile.loc["Acquisition_Status", "largest_level_share_pct"]) == 63.06


def test_founding_years_peak_in_2021(raw: pd.DataFrame) -> None:
    profile = descriptive.founding_year_profile(raw)
    assert profile["min_year"] == 2012
    assert profile["max_year"] == 2025
    assert profile["peak_year"] == 2021
    assert profile["peak_year_rows"] == 3334
    assert profile["founded_before_2020_pct"] == 45.5


def test_the_blanks_are_not_uniform_across_domains(raw: pd.DataFrame) -> None:
    """Corrects a documented claim.

    The README asserted 9.45%-14.71% "consistent with missing-completely-at-random".
    The true Domain range is 6.33%-14.71% — a 8.38 pp spread — and
    ``test_diagnostic.py`` shows that spread is larger than chance produces.
    """
    by_domain = descriptive.missingness_by_strata(raw, "Domain")
    assert by_domain["levels"] == 20
    assert by_domain["min_pct"] == 6.33
    assert by_domain["max_pct"] == 14.71
    assert by_domain["spread_pp"] == 8.38
    assert by_domain["lowest_level"] == "Autonomous Vehicles"
    assert by_domain["highest_level"] == "Quantum Computing"


def test_the_blanks_are_much_flatter_across_funding_stage(raw: pd.DataFrame) -> None:
    by_stage = descriptive.missingness_by_strata(raw, "Funding_Stage")
    assert by_stage["levels"] == 10
    assert by_stage["min_pct"] == 8.44
    assert by_stage["max_pct"] == 11.31
    assert by_stage["spread_pp"] == 2.87


def test_the_missingness_map_covers_every_row(raw: pd.DataFrame) -> None:
    table = descriptive.missingness_map_counts(raw)
    assert len(table) == descriptive.MISSINGNESS_ROW_BINS
    assert int(table["rows"].sum()) == 25_000
    # The overall blank share is 9.74%; no block should be wildly off it, which is what
    # rules out the blanks having been introduced in a contiguous stretch of rows.
    assert float(table["blank_pct"].max()) < 25.0


def test_the_imputation_choice_barely_moves_the_headline(raw: pd.DataFrame) -> None:
    """Five strategies, five answers — but here they agree, which is itself the result.

    The workbooks' lesson is that this table is usually alarming. On this data the gap
    moves 0.46 pp across every strategy, so the choice is low-stakes and the explicit
    level is chosen for the reasons in the design decision rather than for its effect.
    """
    sensitivity = descriptive.imputation_sensitivity(raw)
    assert sensitivity["strategies"] == 5
    assert sensitivity["gap_min_pp"] == 2.53
    assert sensitivity["gap_max_pp"] == 2.99
    assert sensitivity["gap_range_pp"] == 0.46
    assert sensitivity["gap_at_explicit_missing_pp"] == 2.99
    assert sensitivity["rows_lost_by_dropping"] == 2_434


def test_every_imputation_strategy_keeps_the_five_real_levels(raw: pd.DataFrame) -> None:
    table = descriptive.imputation_strategies(raw)
    assert int(table.loc["leave blank (grouped out)", "levels"]) == 5
    assert int(table.loc[f"explicit {MISSING_LEVEL} level", "levels"]) == 6
    assert int(table.loc["drop the rows", "rows"]) == 25_000 - 2_434


def test_the_two_outlier_rules_disagree_by_an_order_of_magnitude(raw: pd.DataFrame) -> None:
    """Which is why the workbooks insist on reporting both."""
    funding = descriptive.outlier_report(raw, "Total_Funding_USD_Millions")
    assert funding["iqr_outlier_rows"] == 3848
    assert funding["z_outlier_rows"] == 562
    assert funding["iqr_outlier_pct"] == 15.39
    assert funding["z_outlier_pct"] == 2.25


def test_outliers_are_not_concentrated_in_one_domain(raw: pd.DataFrame) -> None:
    """A category holding most of the outliers would be a finding, not a data problem.

    Generative AI leads at 15.33%, but it is also the largest domain at 14.9% of rows —
    so the outliers are spread in proportion to the data, not clustered.
    """
    funding = descriptive.outlier_report(raw, "Total_Funding_USD_Millions")
    assert funding["top_domain_among_outliers"] == "Generative AI"
    assert funding["top_domain_share_pct"] == 15.33


def test_the_categories_need_no_cleaning(raw: pd.DataFrame) -> None:
    """Verified rather than applied: the loader's declared dtypes are the contract."""
    report = descriptive.duplicate_report(raw)
    assert report["duplicate_rows"] == 0
    assert report["duplicate_rows_excluding_id"] == 0
    assert report["categories_are_clean"] is True
    assert report["columns_needing_cleaning"] == {}


def test_funding_is_too_skewed_for_its_mean_to_be_quoted(raw: pd.DataFrame) -> None:
    shape = descriptive.distribution_shape(raw, "Total_Funding_USD_Millions")
    assert shape["mean"] == 226.79
    assert shape["median"] == 17.54
    assert shape["skew"] == 6.78
    assert shape["heavily_skewed"] is True
    assert shape["std_exceeds_mean"] is True
    # The mean sits nearly 12x above the median. Quoting it as "typical" would be wrong.
    assert shape["mean_exceeds_median_pct"] == 1193.0


def test_the_group_summary_carries_an_n_beside_every_statistic(raw: pd.DataFrame) -> None:
    summary = descriptive.group_summary(raw, "Runway_Months_2024", "Investor_Tier")
    assert list(summary.columns) == [
        "n",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "share_of_rows_pct",
    ]
    assert int(summary["n"].sum()) == 25_000
    assert round(float(summary["share_of_rows_pct"].sum()), 0) == 100.0


def test_descriptive_and_audit_agree_on_the_blank_count(raw: pd.DataFrame) -> None:
    """Cross-module consistency: two modules must not report different totals."""
    from_audit = audit.missingness(raw)["AI_Adoption_Level"]
    from_descriptive = int(descriptive.category_profile(raw).loc["AI_Adoption_Level", "blank_rows"])
    assert from_audit == from_descriptive == 2_434


def test_run_all_reports_the_row_count_it_used(raw: pd.DataFrame) -> None:
    """The two-frame guard: this tier describes all 25,000 rows, not the 24,467."""
    results = descriptive.run_all(raw)
    assert results["rows"] == 25_000
    assert "numeric_ranges" in results
    assert "imputation_sensitivity" in results
