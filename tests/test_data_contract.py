"""Contract tests for the raw dataset.

These pin the shape and structure of the reference Kaggle download. A failure here
means the data changed underneath the analysis, not that the code is wrong — check the
sha256 in the README before editing anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from startup_outcomes import config
from startup_outcomes.load import DTYPES, drop_leaky_stage, load_raw


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_raw()


def test_shape(df: pd.DataFrame) -> None:
    assert df.shape == (25_000, 17)


def test_columns_match_contract(df: pd.DataFrame) -> None:
    assert list(df.columns) == config.ALL_COLUMNS


def test_every_column_is_classified_exactly_once() -> None:
    """No column may sit in two timing groups, or in none of them."""
    groups = [
        config.STRUCTURAL,
        config.PRE_OUTCOME,
        config.UNDATED,
        config.EXCLUDED_LEAKY,
    ]
    classified = [column for group in groups for column in group]
    assert len(classified) == len(set(classified)), "a column appears in two timing groups"
    assert set(classified) == set(config.ALL_COLUMNS) - {config.ID_COLUMN, config.TARGET}


def test_declared_dtypes_cover_every_column(df: pd.DataFrame) -> None:
    assert set(DTYPES) == set(df.columns)


def test_identifier_is_unique(df: pd.DataFrame) -> None:
    assert df[config.ID_COLUMN].is_unique


def test_no_duplicate_records_ignoring_identifier(df: pd.DataFrame) -> None:
    without_id = df.drop(columns=[config.ID_COLUMN]).astype(str)
    assert len(without_id.drop_duplicates()) == 25_000


def test_ai_adoption_level_is_the_only_column_with_nulls(df: pd.DataFrame) -> None:
    with_nulls = df.columns[df.isna().any()].tolist()
    assert with_nulls == ["AI_Adoption_Level"]
    assert int(df["AI_Adoption_Level"].isna().sum()) == 2_434


def test_none_is_a_category_not_a_null(df: pd.DataFrame) -> None:
    """Regression test: pandas' default na_values swallows the string "None".

    "None" is a meaningful AI_Adoption_Level (no AI adoption). Reading it as missing
    would inflate the null count to 4,278 and leave only four levels.
    """
    levels = set(df["AI_Adoption_Level"].cat.categories)
    assert "None" in levels
    assert len(levels) == 5
    assert int((df["AI_Adoption_Level"] == "None").sum()) == 1_844


def test_cardinalities(df: pd.DataFrame) -> None:
    expected = {
        "Domain": 20,
        "Country": 20,
        "City": 58,
        "Funding_Stage": 10,
        "Investor_Tier": 6,
        "AI_Adoption_Level": 5,
        config.TARGET: 5,
    }
    assert {column: df[column].nunique() for column in expected} == expected


def test_city_nests_within_country(df: pd.DataFrame) -> None:
    per_city = df.groupby("City", observed=True)["Country"].nunique()
    assert (per_city == 1).all()


def test_founding_year_range(df: pd.DataFrame) -> None:
    assert df["Founding_Year"].min() == 2012
    assert df["Founding_Year"].max() == 2025


def test_no_negative_or_impossible_magnitudes(df: pd.DataFrame) -> None:
    for column in ["Total_Funding_USD_Millions", "Valuation_USD_Millions", "Runway_Months_2024"]:
        assert (df[column] > 0).all(), column
    assert (df["Revenue_ARR_Millions"] >= 0).all()
    assert (df["Layoffs_2024_2025"] >= 0).all()
    assert (df["Current_Headcount_2026"] >= 1).all()
    assert (df["Layoffs_2024_2025"] <= df["Peak_Headcount_2023"]).all()


def test_drop_leaky_stage_removes_the_determining_level(df: pd.DataFrame) -> None:
    cleaned = drop_leaky_stage(df)
    assert len(cleaned) == 24_467
    assert config.LEAKY_STAGE not in set(cleaned["Funding_Stage"].cat.categories)
    # The outcome itself survives — 545 IPO outcomes sit at other stages.
    assert int((cleaned[config.TARGET] == "IPO").sum()) == 545


def test_model_features_exclude_outcome_contemporaneous_columns() -> None:
    for include_undated in (True, False):
        features = config.model_features(include_undated=include_undated)
        assert not set(features) & set(config.EXCLUDED_LEAKY)
        assert config.TARGET not in features
        assert config.ID_COLUMN not in features


def test_assert_no_leakage_rejects_excluded_columns() -> None:
    config.assert_no_leakage(config.model_features())
    with pytest.raises(ValueError, match="Current_Headcount_2026"):
        config.assert_no_leakage([*config.STRUCTURAL, "Current_Headcount_2026"])
