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

#### ✅ Design Decision: Baselines before models, and accuracy is never the headline

**Context:**
The majority class scores 86.19% accuracy on binary closed-vs-rest at a 13.81% positive rate. Any
model will look strong on accuracy while potentially having learned nothing. Meanwhile the real
effects in this synthetic data are small (a 3.0 pp spread for AI adoption) or threshold-shaped (the
6-month runway step), so an unqualified metric is close to meaningless.

**Pattern:**

```python
# ✅ GOOD — the comparison is the result, and "beats" is arithmetic
results = supervised.leaderboard_results(design, target)   # baselines first, then the zoo
print(protocol.leaderboard(results))
verdict = supervised.beats_baseline(design, target)        # paired bootstrap on the difference
assert verdict["beats_baseline"] is False                  # and it is

# ❌ BAD
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

Required for any reported result: a baseline comparison, an interval (bootstrap or CV spread) on
the delta, and stratified k-fold on the target using `RANDOM_SEED`. A model that does not beat the
runway baseline by more than its interval is reported as not beating it.

`beats_baseline` returns `True` only when the paired bootstrap interval excludes zero **and** the
difference exceeds one fold standard deviation. Both conditions, because either alone is gameable.

**Applied 2026-09-08.** The rule bit on first use, which is the outcome it was written for. The
gradient-boosted model's PR-AUC advantage over the same estimator given `Runway_Months_2024` alone
is +0.0016 with an interval of [−0.005, +0.009], so it is reported as not beating it. Three further
consequences the proposal did not anticipate:

1. **The baseline had to get stronger.** The literal two-level threshold rule (PR-AUC 0.186) is too
   easy to clear, because it emits only two distinct probabilities and therefore ranks coarsely.
   The honest comparator is the *same estimator restricted to the runway column* (0.191). Both are
   reported; the second is the one the headline rests on.
2. **`accuracy_pct` exists in exactly one function**, `protocol.threshold_counts`, and is always
   returned beside `majority_baseline_accuracy_pct` in the same dict so the two cannot be quoted
   apart. On this data that pairing is the whole finding: 85.88% against 85.89%.
3. **Balanced accuracy turned out to be the wrong headline too.** At the default threshold the
   model flags 2 of 24,467 rows, so balanced accuracy is 50.0% — true, but it describes the
   threshold rather than the model. PR-AUC against the base rate is what leads.

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

**Date:** 2026-09-08 | **Status:** ✅ Approved — ratified by `models/supervised.py`; enforced by
`tests/test_supervised.py::test_no_model_beats_the_runway_column`

#### ✅ Design Decision: Treat the size measures as one collinear block

**Context:**
On logs, funding↔valuation r=0.974, funding↔revenue r=0.968, revenue↔valuation r=0.933,
burn↔revenue r=0.944, and peak↔current headcount r=0.989. These are all proxies for company size.
Individual coefficients and individual feature importances across them are not interpretable.

**Pattern:**

```python
# ✅ GOOD — the block is permuted together, so the ordering cannot be read off
table = supervised.grouped_importance(design, target)
table.loc["size block (4 columns)", "pr_auc_drop"]

# ❌ BAD — "valuation matters more than funding" is noise at r=0.974
top_feature = importances.idxmax()
```

Log-transform before any linear or distance-based model (`protocol.LOG_COLUMNS`); prefer regularized
models over unpenalized regression; when reporting importances, permute the block together.

**Applied 2026-09-08, and now structurally enforced.** `supervised.feature_groups()` returns the
four size proxies as one entry, so `grouped_importance()` has **no key** for an individual size
column and the forbidden comparison cannot be made from its output even by accident.
`tests/test_supervised.py::test_importance_is_only_ever_reported_by_block` asserts that, and
`test_every_feature_appears_in_exactly_one_importance_group` asserts the grouping is a partition.

Two findings came out of applying it:

1. **Logging is what creates the block.** On raw values three of the five pairs sit below r = 0.9 —
   readable as "strong but separable", which would license interpreting them individually. On logs
   every pair clears 0.92 and Pearson lands on top of Spearman: the relationships are monotone but
   curved. The decision rests on the logged figures, and `diagnostic.correlation_comparison()`
   reports both so the difference is visible.
2. **The block barely matters anyway.** Scrambling all four size columns together costs 0.0046
   PR-AUC; scrambling `Runway_Months_2024` alone costs 0.0327. Reporting the block as a group cost
   nothing in this case — but the rule is what stopped an arbitrary within-block ordering being
   published as a finding.

`Runway_Months_2024` is deliberately **not** logged: its effect is a step at six months, and logging
would blur the threshold the generator encoded.

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

**Date:** 2026-09-08 | **Status:** ✅ Approved — ratified by
`supervised.grouped_importance()`; structurally enforced by
`tests/test_supervised.py::test_importance_is_only_ever_reported_by_block`

## Evaluation Protocol

#### ✅ Design Decision: Pooled out-of-fold predictions are the reporting unit

**Context:**
A metric can be computed two ways: average the per-fold values, or score one length-n out-of-fold
prediction vector. On this data the two disagree in the third decimal (PR-AUC 0.1930 against
0.1940) — the same order as the model-versus-baseline difference under study. A third choice,
ordinal category codes instead of native categorical handling, moves ROC-AUC by 0.0048, which is
larger than the effect being measured. Leaving any of this implicit makes the README's digits
depend on an unrecorded coin flip.

**Pattern:**

```python
# ✅ GOOD — one vector, scored once
scores = protocol.out_of_fold_probabilities(estimator, design, target)
metrics = protocol.score_probabilities(target, scores)   # the headline
spread = protocol.fold_spread(target, scores)            # the yardstick, from the same vector

# ❌ BAD — a second, quietly different convention
cross_val_score(estimator, X, y, scoring="average_precision").mean()
```

**Rationale:**
1. Average precision is a ranking statistic and does not decompose meaningfully over 4,893-row folds
2. One number per model instead of two competing ones
3. The fold spread derived from the same vector costs no extra fitting and cannot contradict the
   headline it qualifies
4. The prescriptive tier reuses the vector at zero cost, which is what makes its thresholds honest —
   it never scores a training row

**When NOT to Change This:**
- Never report a metric averaged over folds as the headline
- Never mix conventions within a comparison

**When to Revisit:**
- A metric that genuinely decomposes per fold becomes the headline, which PR-AUC does not

**Alternatives Considered:**
- ❌ Mean of per-fold metrics: differs at the third decimal, and averages a ranking statistic
- ❌ A single holdout: see the next decision

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: No train/test split; the tuned score is labelled optimistic

**Context:**
The reference workbooks use `train_test_split(test_size=0.25)`. With 3,453 positives a 25% holdout
carries roughly 860 of them and a PR-AUC standard error near 0.012 — larger than every effect in
this project and about twenty times the model-versus-baseline gap. Measured directly: changing only
the split seed moves the score by 0.0067, four times the difference being tested.

**Pattern:**

```python
# ✅ GOOD — 5-fold out-of-fold over all rows
scores = protocol.out_of_fold_probabilities(estimator, design, target)

# The one place a single split survives: showing that a single split lies.
supervised.seed_instability(design, target)   # range 0.0067 across five seeds

# ❌ BAD — a holdout that cannot resolve the effect
train_test_split(X, y, test_size=0.25)
```

`GridSearchCV.best_score_` is reported in its own dict carrying `optimistic: True`, never compared
against the out-of-fold numbers.

**Rationale:**
1. A holdout that cannot resolve the effect adds variance and a false sense of rigour
2. Cross-validation-only removes the temptation to peek at a holdout
3. The cost — no untouched data behind the tuned figure — is disclosed rather than hidden

**When NOT to Change This:**
- Do not introduce a holdout to "validate" a cross-validated number
- Do not quote the tuned score beside the out-of-fold scores as if comparable

**When to Revisit:**
- An effect large enough for a holdout to resolve, or enough rows that a holdout carries thousands
  of positives

**Alternatives Considered:**
- ❌ Nested CV: ~110 s to reach a conclusion already reached — the tuning gain (0.007) is inside two
  fold standard deviations either way
- ❌ Holdout plus CV: two numbers that disagree, with no rule for which to quote

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: Model metrics are pinned at 3 dp, each pin beside a band test

**Context:**
Fold-to-fold standard deviation is 0.004–0.008, so a fourth decimal carries no information and
pinning it converts numerical noise into a red build. But two decimals make the model-versus-
baseline comparison unquotable: 0.19 against 0.19. Meanwhile `uv.lock` pins scikit-learn exactly,
so drift only enters through a deliberate `uv lock --upgrade` — at which point a maintainer needs
to know whether a moved digit is cosmetic.

**Pattern:**

```python
# The pin guards the README digit.
def test_the_documented_model_metrics_are_exact(results):
    assert results["leaderboard"]["gradient boosting"]["pr_auc"] == 0.193, UPGRADE_HINT

# The band, in a SEPARATE test, distinguishes drift from a changed finding.
def test_model_metrics_stay_inside_the_documented_band(results):
    assert abs(results["leaderboard"]["gradient boosting"]["pr_auc"] - 0.193) < 0.01
```

Exact fails + band passes → cosmetic, update the digit and the README in one commit. Both fail →
the finding changed; investigate. **Never widen a band to make a test pass.**

`best_params_` is never pinned or quoted: with eight near-tied grid combinations, any change to
scikit-learn's tie-breaking flips it, so it documents a coin toss.

**Rationale:**
1. The band width (~1.5 fold SDs) is chosen from the measured noise, not guessed
2. Two tests give a free diagnosis; one test would only say "something moved"
3. It is not `pytest.approx`, which the project bans — an explicit documented band, like the
   existing `< 2.0` plateau assertion

**When NOT to Change This:**
- Do not pin a fourth decimal
- Do not replace the pair with a single tolerance assertion

**When to Revisit:**
- A metric with materially lower fold variance becomes the headline

**Alternatives Considered:**
- ❌ Pin exactly at 4 dp: turns noise into failures
- ❌ Band only: the README's digits then have nothing guarding them

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: `AI_Adoption_Level` — missing is its own level, never imputed

**Context:**
The only column with blanks (2,434 rows, 9.74%). A permutation test shows the blank rate varies
with `Domain` far beyond chance — 8.38 pp observed against a 6.43 pp null 95th percentile,
p = 0.007 — so the blanks are missing at random *conditional on domain*, not completely at random.
Global mode-filling would move all 2,434 rows into one level, distorting every table keyed on it,
and would discard information the missingness itself carries. Separately,
`sklearn.inspection.partial_dependence` raises `mixed data types` on a categorical column holding
NaN, and `OneHotEncoder` needs something to encode.

**Pattern:**

```python
# ✅ GOOD — a relabelling, not an estimate: fold-independent, cannot leak
frame = features.with_explicit_missing_level(df)   # blanks become "(Missing)"

# ❌ BAD — invents data, and is not even defensible as "it's random anyway"
df["AI_Adoption_Level"].fillna(df["AI_Adoption_Level"].mode()[0])
```

**Rationale:**
1. An explicit level neither invents data nor discards a signal shown to exist
2. Being a relabelling rather than an estimate, it is fold-independent by construction
3. It unblocks partial dependence and one-hot encoding, both of which the tiers need
4. Measured cost: the headline gap moves from 2.99 pp under an explicit level to 2.53 pp under
   per-domain mode-filling — a 0.46 pp range across all five strategies, so the choice is
   low-stakes but is still made explicitly

**When NOT to Change This:**
- Never `fillna(mode)`, never `dropna()` on this column
- Never impute inside `load_raw`

**When to Revisit:**
- The dataset author confirms how the blanks were generated

**Alternatives Considered:**
- ❌ Global mode: not defensible once MCAR is ruled out
- ❌ Per-domain mode: respects the MAR structure but still invents 2,434 values
- ❌ Drop the rows: discards 9.7% of the data to avoid a decision

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: Two encoding paths, both fitted inside the fold

**Context:**
Trees and distance-based models need incompatible representations. One shared preprocessor would
either waste the boosting model's native categorical support or hand k-means raw category codes as
if they were magnitudes.

**Pattern:**

```python
# Path A — boosting and trees: categories pass through untouched.
HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=RANDOM_SEED)

# Path B — logistic, forest, k-NN, k-means, PCA: fitted per fold, never on all rows.
protocol.encoded_pipeline(estimator, design)   # log -> scale -> one-hot, inside a Pipeline
```

`categorical_features="from_dtype"` is passed **explicitly**: it arrived in scikit-learn 1.4 and
became the default only in 1.6, so passing it makes 1.4 through 1.9 behave identically.
`OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist")` keeps 58 city levels from
becoming 58 near-empty columns (126 naive columns become 112) and routes a level unseen in a fold
into the infrequent bucket instead of raising.

**Rationale:**
1. In-fold fitting is the leakage rule; fitting a scaler or encoder on all rows first is the subtle
   leakage the workbooks warn about and then commit
2. Native categorical handling is why the boosting model needs no encoding at all
3. `min_frequency` bounds the matrix width against a future high-cardinality column

**When NOT to Change This:**
- Never fit an encoder or scaler outside a `Pipeline`
- Never rely on `from_dtype` being the default

**When to Revisit:**
- A model arrives that needs a third representation, e.g. target encoding — which would need its
  own fold-safety argument

**Alternatives Considered:**
- ❌ One shared one-hot matrix: wastes native support and mis-scales for k-means
- ❌ Ordinal codes everywhere: moves ROC-AUC by 0.0048 and implies an order the dtype does not have

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: The labelled-leakage-demonstration protocol

**Context:**
The modeling rules permit a leakage experiment if it is recorded and labelled, and the
core-checklist enforces a grep that no module outside `config`/`load`/`audit` may name a leaky
column. A demonstration that hand-listed `Current_Headcount_2026` would turn that grep red and
silently delete the enforcement — the guard would be gone precisely because the one legitimate
exception was written carelessly.

**Pattern:**

```python
# ✅ GOOD — references the constant; the names are never spelled here
features.design_matrix(frame, include_excluded=True)   # [*model_features(), *EXCLUDED_LEAKY]

# ❌ BAD — turns the enforcement grep red and takes the contract with it
design[["Domain", "Current_Headcount_2026"]]
```

Five requirements: (i) the feature set is built from `config.EXCLUDED_LEAKY`; (ii) exactly one
function does this, and its name contains `leakage`; (iii) every dict in the tier carries
`leakage_controlled: bool`, with a partition test asserting exactly one is `False`; (iv) a test
asserts `config.assert_no_leakage(demo_features)` **raises** — the demo proves it is precisely what
the guard rejects; (v) the numbers appear in the README only in a row labelled
"leakage demonstration — not a result".

**Rationale:**
1. The label has to be machine-checkable, because a prose caption is the first thing lost when a
   number is copied
2. Building from the constant keeps one list authoritative and the existing grep meaningful
3. The demonstration earns its place: it is the only change in the project that moves the headline
   by more than three fold standard deviations (0.193 → 0.235, 6.8 SDs), which is the evidence for
   the timing contract rather than an assertion of it

**When NOT to Change This:**
- Never add a second function that can see the excluded columns
- Never quote the demonstration's score without its label

**When to Revisit:**
- A column's as-of date is established, which would move it out of `EXCLUDED_LEAKY` entirely

**Alternatives Considered:**
- ❌ No demonstration at all: leaves "excluding these costs real accuracy" as an unquantified claim
- ❌ Naming the columns directly with a `# noqa`-style exemption: deletes the enforcement

**Date:** 2026-09-08 | **Status:** ✅ Approved
