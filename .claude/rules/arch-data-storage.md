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

## Derived Artifacts

#### ✅ Design Decision: Nothing is persisted

**Context:**
The reference workbooks chain their days through files — `day2_clean.csv`, `day3_model.joblib`,
`final_model.joblib`, `model_card.txt`, saved figures — because a Colab session dies between days.
This project runs in one process, and the full pipeline recomputes from `data/raw` in about 45
seconds.

**Pattern:**

```python
# ✅ GOOD — the handoff is a function call
estimator = supervised.chosen_estimator()
print(card.model_card())        # returned string, printed; never written

# ❌ BAD
joblib.dump(model, "final_model.joblib")
open("model_card.txt", "w").write(card)
```

`DATA_INTERIM` and `DATA_PROCESSED` exist in `config.py` as declared destinations, because the rule
above promises those directories by name and a promise with no constant behind it is a comment.
Nothing writes to them. Enforced by `tests/test_repo_invariants.py::test_nothing_in_src_writes_to_disk`,
which greps `src/` for `to_csv|to_parquet|to_pickle|joblib|savefig|.write(`.

**Rationale:**
1. A persisted model is a second source of truth that drifts from the code that made it, and this
   project's whole discipline is "the number comes from a function and a test"
2. `data/` is git-ignored, so an artifact there is invisible to review — the same back-door concern
   that keeps notebook outputs out of version control
3. A joblib pickle is version-brittle across scikit-learn upgrades, and unlike the CSV there is no
   checksum contract to catch a silent behaviour change
4. The model card is the better artifact: a rendering of a flat dict, every value pinned by a test,
   diffable and unable to go stale

**When NOT to Change This:**
- Do not persist a model "so the notebook starts faster" — it starts fast enough
- Do not write figures to disk

**When to Revisit:**
- A fit exceeding roughly five minutes. The cache would then go to `DATA_INTERIM` keyed by
  `(RAW_CSV_SHA256, RANDOM_SEED, feature-list hash, estimator repr)` so a stale cache is
  structurally impossible

**Alternatives Considered:**
- ❌ Persist the fitted model: version-brittle, invisible to review, and saves one second
- ❌ Cache out-of-fold scores to Parquet: unnecessary today; the same effect is achieved by
  threading the score vector through `run_all` as an argument

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: A documented number that could not be reproduced, and how it was handled

**Context:**
The README stated that `AI_Adoption_Level` blanks were "near-uniform across strata (9.45%–14.71% by
domain and funding stage), consistent with missing-completely-at-random". Neither half held. The
true ranges on the reference download are 6.33%–14.71% across `Domain` and 8.44%–11.31% across
`Funding_Stage`; the 9.45% floor matches no grouping that could be constructed. And the inference
is contradicted: a permutation test puts the observed 8.38 pp domain spread beyond the null's 95th
percentile of 6.43 pp, p = 0.007.

It survived because it was one of the few claims in the README with no function behind it.

**Pattern:**

```python
# The claim now has a function and a test, on both halves.
descriptive.missingness_by_strata(df, "Domain")      # the range
diagnostic.missingness_permutation(df, "Domain")     # whether the range beats chance
```

The README sentence, the two functions, and the two tests were changed in one commit, together with
the imputation decision that depended on it (`arch-modeling.md`: missing is its own level, because
mode-filling is not defensible once MCAR is ruled out).

**Rationale:**
1. This is exactly the failure the "every number has a function and a test" rule exists to prevent,
   so the incident belongs in the rules as evidence the rule pays for itself
2. `process-design-decisions.md` already mandates documenting any newly discovered data-integrity
   defect and how it was handled
3. The correction cascaded: it changed an imputation decision, which is what makes an unbacked
   number dangerous rather than merely untidy

**When NOT to Change This:**
- Never add a figure to the README without a function and a test behind it, however obvious it seems
- Never soften a claim to match a number; recompute the number

**When to Revisit:**
- Never. This is a recorded incident, not a policy that expires

**Alternatives Considered:**
- ❌ Quietly fix the range: loses the lesson, and the MCAR inference would have survived
- ❌ Delete the sentence: the question it answers is load-bearing for the imputation choice

**Date:** 2026-09-08 | **Status:** ✅ Approved
