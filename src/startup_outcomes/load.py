"""Typed loading of the raw CSV.

Dtypes are declared rather than inferred so that a changed download surfaces as a
load error instead of silently altering downstream results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from startup_outcomes.config import (
    ALL_COLUMNS,
    LEAKY_STAGE,
    RAW_CSV,
    RAW_CSV_SHA256,
    TARGET,
)

CATEGORICAL = [
    "Domain",
    "Country",
    "City",
    "Funding_Stage",
    "Investor_Tier",
    "AI_Adoption_Level",
    TARGET,
]

FLOAT_COLUMNS = [
    "Total_Funding_USD_Millions",
    "Valuation_USD_Millions",
    "Revenue_ARR_Millions",
    "Monthly_Burn_Rate_Millions",
    "Runway_Months_2024",
]

INT_COLUMNS = [
    "Founding_Year",
    "Peak_Headcount_2023",
    "Layoffs_2024_2025",
    "Current_Headcount_2026",
]

DTYPES: dict[str, str] = {
    "Company_ID": "string",
    **dict.fromkeys(CATEGORICAL, "category"),
    **dict.fromkeys(FLOAT_COLUMNS, "float64"),
    **dict.fromkeys(INT_COLUMNS, "int64"),
}


def sha256(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path | None = None, *, verify_checksum: bool = False) -> pd.DataFrame:
    """Load the raw dataset with declared dtypes.

    ``AI_Adoption_Level`` is the one column with blanks; they are read as NaN and
    deliberately left unimputed here so that imputation is an explicit modeling step.

    Args:
        path: Override the default ``data/raw`` location.
        verify_checksum: Fail if the file does not match the reference download.
    """
    csv_path = path or RAW_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. See README 'Getting started' for how to obtain it."
        )
    if verify_checksum:
        actual = sha256(csv_path)
        if actual != RAW_CSV_SHA256:
            raise ValueError(
                f"{csv_path.name} sha256 {actual} does not match the reference "
                f"{RAW_CSV_SHA256}; the documented row counts may no longer hold."
            )

    # keep_default_na=False is load-bearing, not a preference: "None" is a valid
    # AI_Adoption_Level meaning "no AI adoption", and pandas' default na_values would
    # read those 1,844 rows as missing, inflating the null count and dropping the
    # level entirely. Only a genuinely empty field counts as missing here.
    df = pd.read_csv(csv_path, dtype=DTYPES, keep_default_na=False, na_values=[""])

    missing = [column for column in ALL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Expected columns absent from {csv_path.name}: {missing}")
    return df.loc[:, ALL_COLUMNS]


def drop_leaky_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the rows where ``Funding_Stage`` gives the target away.

    The cheaper alternative is collapsing the level into its neighbour; dropping is
    used by default because it leaves no ambiguity about what the model saw.
    """
    kept = df.loc[df["Funding_Stage"] != LEAKY_STAGE].copy()
    kept["Funding_Stage"] = kept["Funding_Stage"].cat.remove_unused_categories()
    return kept
