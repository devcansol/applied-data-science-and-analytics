# `src/startup_outcomes/` — the importable package

## Purpose

Every number quoted anywhere in this project originates here. Notebooks import from this package
and define nothing; the root README quotes it; the test suite pins it. If a figure exists only in a
notebook cell, it is not a finding yet.

## Structure

The package is layered, and the layering *is* the leakage control. Dependencies run strictly one
way, with no cycles:

```
config ─────────────────────────────▶ (nothing)
intervals ──▶ config
load ───────▶ config
audit ──────▶ config, load
features ───▶ config, load, audit
descriptive ▶ config, load, audit, features
diagnostic ─▶ config, load, audit, features, intervals
models/protocol ────▶ config, features
models/baselines ───▶ audit, config
models/supervised ──▶ protocol, baselines, features, intervals
models/unsupervised ▶ protocol, features
models/prescriptive ▶ protocol, supervised, features, intervals
models/card ────────▶ protocol, supervised, features, config
plots ──────▶ (tier outputs only; no sklearn, no pyplot)
report ─────▶ everything;  imported by nothing
```

`models/` is a package rather than a flat module because `.claude/rules/arch-modeling.md` globs
`models/**`: code under that path is where the leakage, baseline and collinearity mandates apply,
and the glob is what loads them for a reviewer. `descriptive.py` and `diagnostic.py` sit *outside*
it deliberately — they must never fit an estimator, and their reviewer should be reading
`arch-data-pipeline.md` instead.

## Conventions

- **Checks are assertion-free.** They report what the data says; judgement belongs to the reader
  and the pins belong in tests.
- **Returns are flat, JSON-clean dicts** — `int(...)` and `round(float(...), n)`, never numpy
  scalars. Unit suffixes are load-bearing: `_pct`, `_pp`, `_rows`, `_auc`, `_sd`, `_units`.
- **Table-shaped results return DataFrames**, and each has a flat-dict companion holding the values
  the README quotes (`numeric_ranges` beside `numeric_profile`), so a documented number is always
  pinned by a test even when the table itself is not.
- **`run_all(df=None)` per tier.** The default is `None` rather than `load_raw()` because a call in
  a default argument is both a lint error and a shared-mutable trap.
- **Pure functions.** Take a frame, return a new one. No `inplace`, no returning `None`, no
  module-level mutable state — the reference `mlkit` accumulates results in a module dict, which
  makes output depend on which cells ran before.
- **One seed.** Every line mentioning `random_state`, `np.random` or a seed names `RANDOM_SEED` on
  the same line, so the rule is greppable.
- **Features only from `config.model_features()`.** Never a hand-written list, never a drop-list.
- **Nothing is persisted.** No `to_csv`, no `joblib`, no `savefig`.

## Files Summary

| File | What it holds |
|---|---|
| `config.py` | Paths, `RANDOM_SEED`, `BOOTSTRAP_RESAMPLES`, the feature-timing groups, `SIZE_BLOCK`, `model_features()`, `assert_no_leakage()` |
| `load.py` | `DTYPES`, `sha256()`, `load_raw()`, `drop_leaky_stage()` |
| `intervals.py` | `generator()`, `bootstrap_rows()`, `rate_gap_interval()`, `paired_metric_difference()`, `permutation_spread()` |
| `audit.py` | 15 data-integrity checks, `closed_rate_by()`, `SIZE_PAIRS` |
| `features.py` | `canonical_frame()`, `binary_target()`, `with_explicit_missing_level()`, `make_bands()`, `engineered()`, `design_matrix()`, `leakage_scan()` |
| `descriptive.py` | Tier 1: profiles, outliers, the imputation-sensitivity table, missingness by strata |
| `diagnostic.py` | Tier 2: effect sizes with intervals, permutation nulls, crosstabs, the reversal check |
| `plots.py` | ~20 Figure-returning functions |
| `report.py` | `run_all()`, `headline()`, `summary_text()` |
| `models/protocol.py` | The splitter, preprocessing, metrics, `out_of_fold_probabilities()`, `evaluate()` |
| `models/baselines.py` | `RunwayThresholdClassifier` and the two dummies |
| `models/supervised.py` | Tier 3a: the zoo, `beats_baseline()`, depth sweep, tuning, `leakage_demonstration()`, grouped importance |
| `models/unsupervised.py` | Tier 3b: cluster scan, profile, PCA |
| `models/prescriptive.py` | Tier 4: response curves, ranked actions, cost thresholds, expected value |
| `models/card.py` | `card_facts()` and `model_card()` |

## Tips & Key Notes

- **`predictive` costs ~35 seconds** and `report.run_all()` about 45. Both are dominated by
  cross-validation. Thread out-of-fold scores through with the `scores=` argument rather than
  letting each function refit.
- **`categorical_features="from_dtype"` must stay explicit** on the boosting model. It became the
  default only in scikit-learn 1.6, and passing it makes 1.4 through 1.9 behave identically.
- **Never rank the size features against each other.** At r > 0.92 on logs the ordering is
  arbitrary; `grouped_importance()` has no key for an individual size column and a test enforces it.
- **Two frames.** `descriptive` profiles all 25,000 rows; everything touching `Funding_Stage` uses
  the 24,467-row canonical frame. Mixing them is how a README ends up quoting two numbers for one
  claim, so every `run_all()` returns its row count.
- **Assigning a numpy array to a categorical column silently makes it `object`**, after which the
  boosting model refuses it. Assign an aligned Series instead — this bit once, in
  `grouped_importance`.
- **Watch the pandas-vet lint rules** (`PD009`, `PD010`, `PD011`): the crosstab and profiling work
  naturally reaches for `.iat`, `.pivot` and `.values`, and all three are flagged.
