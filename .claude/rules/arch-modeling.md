---
description: features, leakage, feature timing, baselines, cross-validation, metrics, class imbalance, collinearity
globs: ["src/startup_outcomes/config.py", "src/startup_outcomes/models/**", "src/startup_outcomes/features*.py"]
alwaysApply: false
---
> **Architecture family:** Modeling & Evaluation. See also: `.claude/rules/arch-data-storage.md`, `.claude/rules/arch-data-pipeline.md`, `.claude/rules/arch-notebooks.md`

# Modeling & Evaluation

## Leakage Control

#### ✅ Design Decision: Feature eligibility is declared in config, not decided at the call site

**Context:**
`Acquisition_Status` is observed as of 2026, and the dataset contains `Current_Headcount_2026` and
`Layoffs_2024_2025`. A 2026 headcount cannot predict a 2026 outcome — it partly *is* the outcome.
Column-by-column vigilance at each call site is exactly the discipline that fails quietly, and the
failure looks like success: better metrics.

**Pattern:**

```python
# config.py groups every column by what it can be known at, relative to the label
STRUCTURAL      = ["Domain", "Country", "City", "Founding_Year", "Investor_Tier"]
PRE_OUTCOME     = ["Runway_Months_2024", "Peak_Headcount_2023"]
UNDATED         = ["Funding_Stage", "Total_Funding_USD_Millions", ...]
EXCLUDED_LEAKY  = ["Layoffs_2024_2025", "Current_Headcount_2026"]

# ✅ GOOD — the allowed set is derived, and asserted
features = config.model_features()
config.assert_no_leakage(features)
X = df[features]

# ❌ BAD — a hand-written list, or a drop-list that a new column can slip past
X = df.drop(columns=["Company_ID", "Acquisition_Status"])
```

A contract test asserts every column sits in exactly one timing group, so a newly added column
cannot be silently unclassified.

**Rationale:**
1. Excluding these columns costs real apparent accuracy — a rule that costs something has to be
   structural, or it will be quietly relaxed
2. `model_features(include_undated=False)` gives the conservative variant, so the undated-column
   assumption can be tested rather than argued about
3. One place to read to know what a model saw

**When NOT to Change This:**
- Never add `Layoffs_2024_2025` or `Current_Headcount_2026` to a feature set, including "just to
  see" — record the experiment as a leakage demonstration and label it as such
- Never bypass `model_features()` with a manual list

**When to Revisit:**
- A column's as-of date is established from the dataset author, which could promote one of the
  undated financial columns to `PRE_OUTCOME`

**Alternatives Considered:**
- ❌ Drop-list at the call site: a new leaky column defaults to *included*, the wrong default
- ❌ Naming convention only (exclude anything matching `*_2026`): breaks on the first undated
  leaky column

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: Neutralise `Funding_Stage == 'IPO'` before using the column

**Context:**
`Funding_Stage == 'IPO'` implies `Acquisition_Status == 'IPO'` in all 533 rows where it appears.
The implication is one-directional — 545 further IPO outcomes sit at other stages — so the outcome
does not need redefining, but the level hands the label to any model that sees it.

**Pattern:**

```python
# ✅ GOOD — explicit removal, categories cleaned up
df = drop_leaky_stage(df)   # 25,000 → 24,467 rows; 545 IPO outcomes remain

# ❌ BAD — one-hot encoding Funding_Stage as-is gives the model a perfect rule
X = pd.get_dummies(df["Funding_Stage"])
```

**Rationale:**
1. Dropping leaves no ambiguity about what the model saw
2. Keeps the IPO outcome class alive, so the 5-class problem stays intact
3. Cheap: 533 of 25,000 rows

**When NOT to Change This:**
- Do not use `Funding_Stage` without this step
- Do not drop the IPO outcome class instead — that discards 545 legitimate rows

**When to Revisit:**
- If collapsing `IPO` into `Pre-IPO` is preferred to preserve sample size, document the choice and
  show that results are unchanged

**Alternatives Considered:**
- ❌ Keep the level and rely on regularization: a deterministic rule survives regularization
- ❌ Redefine the target to exclude IPO: throws away a real outcome class

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Evaluation

#### 🚧 Design Decision: Baselines before models, and accuracy is never the headline

**Context:**
The majority class scores 86.19% accuracy on binary closed-vs-rest at a 13.81% positive rate. Any
model will look strong on accuracy while potentially having learned nothing. Meanwhile the real
effects in this synthetic data are small (a 3.0 pp spread for AI adoption) or threshold-shaped (the
6-month runway step), so an unqualified metric is close to meaningless.

**Pattern:**

```python
# ✅ GOOD — the comparison is the result
results = {
    "majority":   evaluate(DummyClassifier(strategy="most_frequent")),
    "stratified": evaluate(DummyClassifier(strategy="stratified",
                                           random_state=RANDOM_SEED)),
    "runway_rule": evaluate(threshold_model(months=6)),   # the known generator rule
    "model":      evaluate(HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
}
# Report balanced accuracy / PR-AUC / per-class recall with intervals, and the delta vs baseline.

# ❌ BAD
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

Required for any reported result: a baseline comparison, an interval (bootstrap or CV spread) on
the delta, and stratified k-fold on the target using `RANDOM_SEED`. A model that does not beat the
runway-threshold baseline by more than its interval is reported as not beating it.

**Rationale:**
1. At this class balance, accuracy rewards predicting "survives" forever
2. Much of any apparent skill here is rediscovered generator scaffolding; the threshold baseline
   measures how much
3. Small effects without intervals are unreadable

**When NOT to Change This:**
- Never report a bare accuracy figure
- Never report an effect without an interval

**When to Revisit:**
- A cost-sensitive framing arrives with real misclassification costs, which would change the
  metric but not the baseline requirement

**Alternatives Considered:**
- ❌ Accuracy with a note about imbalance: the note gets dropped when the number is quoted
- ❌ ROC-AUC as headline: over-optimistic under strong imbalance; PR-AUC is the sharper read

**Date:** 2026-09-08 | **Status:** 🚧 Proposed — no models exist yet; ratify with the first one

#### 🚧 Design Decision: Treat the size measures as one collinear block

**Context:**
On logs, funding↔valuation r=0.974, funding↔revenue r=0.968, revenue↔valuation r=0.933,
burn↔revenue r=0.944, and peak↔current headcount r=0.989. These are all proxies for company size.
Individual coefficients and individual feature importances across them are not interpretable.

**Pattern:**

```python
# ✅ GOOD — read them as a group, or reduce them to one factor
size_block = ["Total_Funding_USD_Millions", "Valuation_USD_Millions",
              "Revenue_ARR_Millions", "Monthly_Burn_Rate_Millions"]
importance_of_size = sum(importances[c] for c in size_block)

# ❌ BAD — "valuation matters more than funding" is noise at r=0.974
top_feature = importances.idxmax()
```

Log-transform before any linear model; prefer regularized models over unpenalized regression; when
reporting importances, aggregate the block or use a permutation scheme that permutes it together.

**Rationale:**
1. Coefficient signs and magnitudes flip arbitrarily between near-identical predictors
2. Tree importances split credit arbitrarily among correlated features
3. Ratios (burn/revenue, valuation/revenue) carry information the raw levels do not

**When NOT to Change This:**
- Do not rank individual size features against each other and report the ordering as a finding

**When to Revisit:**
- Engineered ratios replace the raw levels, changing the correlation structure

**Alternatives Considered:**
- ❌ Drop all but one: loses the ratio information
- ❌ PCA the block: fixes the statistics, costs the interpretability that motivated keeping them

**Date:** 2026-09-08 | **Status:** 🚧 Proposed — no models exist yet; ratify with the first one
