"""Paths, seed, and the feature-timing contract.

The column groups below are the enforcement mechanism behind the leakage claim in the
README: modeling code builds its design matrix from ``model_features()``, so an
outcome-contemporaneous column cannot be used by accident.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_CSV = DATA_RAW / "global_tech_startups_2026.csv"

#: Declared destinations for derived data. Nothing in this project writes to them — every
#: number is recomputed from ``data/raw`` in seconds — but the storage rule promises these
#: directories by name, and a promise with no constant behind it is a comment.
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

#: sha256 of the Kaggle download this project was built against.
RAW_CSV_SHA256 = "c843a3c03495b75c35713126cc3df8f902f8188abdfa876fff5bc4a9f0227ee2"

#: Single seed for every split, shuffle, and model in the project.
RANDOM_SEED = 20260908

#: Resamples behind every reported interval. At 1,000 the second decimal of a percentage
#: point is stable across draws while the diagnostic tier still runs in a few seconds.
BOOTSTRAP_RESAMPLES = 1000

ID_COLUMN = "Company_ID"
TARGET = "Acquisition_Status"

# --- Feature-timing contract -------------------------------------------------
# The outcome (Acquisition_Status) is observed as of 2026. Columns are grouped by
# what they can be known at, relative to that label.

#: Fixed at founding or otherwise time-invariant.
STRUCTURAL = [
    "Domain",
    "Country",
    "City",
    "Founding_Year",
    "Investor_Tier",
]

#: Explicitly dated before the label period by their own column names.
PRE_OUTCOME = [
    "Runway_Months_2024",
    "Peak_Headcount_2023",
]

#: Carry no date in their name. Treated as as-of-latest, which is a stated
#: assumption rather than a fact the data supports.
UNDATED = [
    "Funding_Stage",
    "Total_Funding_USD_Millions",
    "Valuation_USD_Millions",
    "Revenue_ARR_Millions",
    "Monthly_Burn_Rate_Millions",
    "AI_Adoption_Level",
]

#: Contemporaneous with or after the label. Never available to the primary model.
EXCLUDED_LEAKY = [
    "Layoffs_2024_2025",
    "Current_Headcount_2026",
]

#: Funding_Stage == LEAKY_STAGE determines the target outright (533/533 rows in the
#: reference download), so it must be neutralised before Funding_Stage is used.
LEAKY_STAGE = "IPO"

ALL_COLUMNS = [ID_COLUMN, *STRUCTURAL, *PRE_OUTCOME, *UNDATED, *EXCLUDED_LEAKY, TARGET]

#: The size proxies, which are near-identical on logs (r = 0.93–0.97). Individual
#: coefficients and individual importances across them are not interpretable, so they are
#: read as one block; see the collinearity decision in ``.claude/rules/arch-modeling.md``.
SIZE_BLOCK = [
    "Total_Funding_USD_Millions",
    "Valuation_USD_Millions",
    "Revenue_ARR_Millions",
    "Monthly_Burn_Rate_Millions",
]


def model_features(*, include_undated: bool = True) -> list[str]:
    """Columns a leakage-controlled model may use.

    Args:
        include_undated: Include the columns whose as-of date is unknown. Set False
            for the conservative variant that relies only on explicitly dated and
            structural fields.
    """
    features = [*STRUCTURAL, *PRE_OUTCOME]
    if include_undated:
        features += UNDATED
    return features


def assert_no_leakage(columns: object) -> None:
    """Raise if any outcome-contemporaneous column reached a feature set."""
    leaked = sorted(set(EXCLUDED_LEAKY) & set(columns))  # type: ignore[arg-type]
    if leaked:
        raise ValueError(
            f"Columns excluded by the feature-timing contract reached the model: {leaked}. "
            "See README 'Methodology' for why these cannot predict a 2026 outcome."
        )
