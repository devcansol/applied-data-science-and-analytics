---
description: raw data, loading, dtypes, schema contract, checksums, data layering, immutability
globs: ["data/**", "src/startup_outcomes/load.py", "src/startup_outcomes/config.py"]
alwaysApply: false
---
> **Architecture family:** Data Storage & Loading. See also: `.claude/rules/arch-data-pipeline.md`, `.claude/rules/arch-modeling.md`, `.claude/rules/arch-notebooks.md`

# Data Storage & Loading

## Raw Data

#### ✅ Design Decision: Raw data is immutable, git-ignored, and checksum-anchored

**Context:**
The dataset is a 3 MB CSV downloaded from Kaggle whose license is unconfirmed, so it cannot safely
be committed. But an analysis whose input can silently change is not reproducible, and the README
quotes exact row counts that only hold for one specific download.

**Pattern:**

```python
# config.py — the checksum is part of the contract, not a comment
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "global_tech_startups_2026.csv"
RAW_CSV_SHA256 = "c843a3c03495b75c35713126cc3df8f902f8188abdfa876fff5bc4a9f0227ee2"

# ✅ GOOD — verify when it matters (notebooks, any published result)
df = load_raw(verify_checksum=True)

# ❌ BAD — writing back over the input
df.to_csv("data/raw/global_tech_startups_2026.csv")
```

Derived data goes to sibling directories (`data/interim/`, `data/processed/`), never back into
`data/raw/`. `.claude/settings.json` denies `Write`/`Edit` on `data/raw/**` and the `rm`/`mv`
spellings that would destroy it.

**Rationale:**
1. Reproducibility — the checksum makes "same data" verifiable rather than assumed
2. License safety — the file is obtainable but not redistributed by this repo
3. The README documents how to re-obtain it, so a fresh clone is not stuck

**When NOT to Change This:**
- Do not commit the CSV, even for convenience, until the Kaggle license is confirmed
- Do not relax the deny rules on `data/raw/**`

**When to Revisit:**
- The license is confirmed as permitting redistribution
- The data becomes large enough to warrant Parquet or a data-version-control tool

**Alternatives Considered:**
- ❌ Commit the CSV: unconfirmed license, and 3 MB of churn in git history
- ❌ Trust the filename alone: a re-download under the same name would silently invalidate every
  documented number

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Loading

#### ✅ Design Decision: Declare dtypes and disable pandas' default NA strings

**Context:**
`AI_Adoption_Level` has a valid level spelled `None`, meaning *no AI adoption*. Pandas' default
`na_values` includes the string `"None"`, so a plain `read_csv` converted 1,844 real observations
to NaN — inflating missingness from 2,434 to 4,278 and reducing the column from five levels to
four. This was caught only because an independent profile disagreed with the loader.

**Pattern:**

```python
# ❌ BAD — "None" becomes NaN, a whole category disappears
df = pd.read_csv(csv_path)

# ✅ GOOD — only a genuinely empty field is missing
df = pd.read_csv(csv_path, dtype=DTYPES, keep_default_na=False, na_values=[""])
```

Dtypes are declared for every column (`string`, `category`, `float64`, `int64`) rather than
inferred, and `load_raw` raises if an expected column is absent. Missing values are left
unimputed at load time so imputation stays an explicit modeling decision.

**Rationale:**
1. A sentinel that collides with a real category is silent and severe — it changes results without
   raising anything
2. Declared dtypes turn a changed file into a load error instead of altered numbers
3. Categories keep memory flat and make cardinality assertions meaningful

**When NOT to Change This:**
- Never remove `keep_default_na=False` — there is a regression test guarding it
- Never impute inside `load_raw`

**When to Revisit:**
- A new column arrives with genuinely blank-but-meaningful values needing different handling

**Alternatives Considered:**
- ❌ Post-load repair (`fillna("None")`): fixes the count but not the lost distinction between a
  real `None` and a blank
- ❌ `dtype=str` everywhere then cast: defers errors to the point of use

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: The schema is asserted by tests, not assumed

**Context:**
The README quotes exact figures — 25,000 rows, 2,434 blanks, 58 cities nested in 20 countries. A
re-download or an upstream revision would make those quietly wrong.

**Pattern:**

```python
# tests/test_data_contract.py — a failure means the DATA changed, not the code
def test_ai_adoption_level_is_the_only_column_with_nulls(df):
    assert df.columns[df.isna().any()].tolist() == ["AI_Adoption_Level"]
    assert int(df["AI_Adoption_Level"].isna().sum()) == 2_434
```

**Rationale:**
1. Turns documentation drift into a test failure
2. Makes the boundary explicit: contract tests describe the data, unit tests describe the code
3. Cheap to run, so it runs on every change

**When NOT to Change This:**
- Do not loosen an assertion to make a test pass — check the checksum first; if the data legitimately
  changed, update the assertions and the README together

**When to Revisit:**
- A second dataset joins the project, at which point each needs its own contract

**Alternatives Considered:**
- ❌ A schema-validation library (pandera, Great Expectations): more machinery than one CSV needs
- ❌ Runtime assertions in `load_raw`: would make every load pay for the checks

**Date:** 2026-09-08 | **Status:** ✅ Approved
