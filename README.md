# Startup Outcomes: A Leakage-Controlled Analysis

An applied data science project on the Kaggle dataset
[Global Tech & AI Startups Insights 2020–2026](https://www.kaggle.com/datasets/beamhonor0911/global-tech-ai-startups-insights-2020-2026)
(25,000 companies × 17 columns), predicting company outcome — independent, acquired, closed,
merged, or IPO.

The dataset turns out to be **synthetic**, with most of its headline effects absent or engineered.
Rather than work around that, this project treats it as the subject: the deliverables are a
reproducible data-integrity audit, a four-tier analysis (descriptive, diagnostic, predictive,
prescriptive), a modeling design where leakage is prevented by construction rather than by good
intentions, and results reported against honest baselines — including where the answer is "no
effect". Every number below is produced by a tier module in
[`src/startup_outcomes/`](src/startup_outcomes/) and pinned by that module's test file, so the
claims here can be re-derived in one command:

```bash
uv run python -c "from startup_outcomes.report import run_all; print(run_all())"
```

**The headline is a negative result.** A thirteen-feature gradient-boosted model does not beat the
same estimator given `Runway_Months_2024` alone, and the only change anywhere in this project that
measurably improves it is deliberate leakage. Everything below is arranged to make that visible
rather than to hide it.

## Data provenance and honesty

**This data is not real.** No company in it exists, and none of its findings transfer to actual
startups. That conclusion is inferred from the file itself, not quoted from the dataset author
(see [License and citation](#license-and-citation)). The evidence:

| Check | Result | Why it can't be real |
|---|---|---|
| Founded ≥2024, already at Series D/E+/Pre-IPO/IPO | **325** rows (574 if ≥2023) | Reaching Series D takes years, not months |
| `Closed` companies whose 2026 headcount exceeds their 2023 peak | **1,432 / 3,453 = 41.5%** | A shut-down company doesn't grow |
| Smallest `Current_Headcount_2026` among `Closed` companies | **1** — never 0 | Nothing that closed has zero staff |
| `Closed` companies with no layoffs at all | **49.1%** (vs 62.6% of `Independent`) | Closure without layoffs is near-impossible |
| Closed-rate spread across all 20 `Domain` values | **4.82 pp** | Real sector risk varies far more |
| Closed-rate spread across all 20 `Country` values | **5.83 pp** | Real geographic risk varies far more |
| `AI_Adoption_Level` effect on closed rate, `None` → `AI-Native` | **15.84% → 12.84%** (3.0 pp) | The dataset's headline premise is ~flat |

Two further findings shape everything downstream:

- **The target is partly given away.** `Funding_Stage == 'IPO'` implies `Acquisition_Status == 'IPO'`
  in **533 of 533** rows. The implication is one-directional — 545 further IPO outcomes sit at other
  stages — so the level must be neutralised rather than the outcome redefined.
- **One relationship is a hard-coded rule.** The closed rate is **23.41%** below 6 months of runway
  and **10.98%** at or above it, flat on both sides. That is a threshold in a data generator, not
  the smooth mortality gradient real runway produces.

### What this means for the project

The interesting questions become methodological rather than commercial. A model here can look
excellent while learning nothing: the majority-class baseline already scores **86.19%** accuracy on
binary closed-vs-rest, and the leakage paths above will happily manufacture more. So the work is
structured to make those failures visible — which is transferable practice, even though the findings
are not.

## Dataset at a glance

25,000 rows × 17 columns · 3,099,505 bytes · one row per company · no duplicate records once
`Company_ID` is ignored.

| Column | Type | Notes |
|---|---|---|
| `Company_ID` | string | Identifier, `TECH-00001`…; unique across all 25,000 rows |
| `Domain` | category | 20 sectors; largest is Generative AI (3,727) |
| `Founding_Year` | int | 2012–2025, peaking in 2021. **45.5% predate 2020**, despite the dataset's 2020–2026 title |
| `Country` | category | 20 countries; largest is United States (7,393) |
| `City` | category | 58 cities, each nested cleanly within exactly one country |
| `Funding_Stage` | category | 10 stages, Bootstrapped → IPO. **The `IPO` level leaks the target** |
| `Total_Funding_USD_Millions` | float | 0.13 – 12,000; median 17.54 |
| `Valuation_USD_Millions` | float | 0.50 – 251,890; median 146.43; never below total funding |
| `Revenue_ARR_Millions` | float | 0.00 – 10,745; median 4.59; 152 rows are exactly zero |
| `Monthly_Burn_Rate_Millions` | float | 0.01 – 927.69; median 0.85; exceeds revenue in 4,799 rows |
| `Runway_Months_2024` | float | 2.0 – 49.2; median 9.2. Dated 2024 |
| `Peak_Headcount_2023` | int | 1 – 55,000; median 51. Dated 2023 |
| `Layoffs_2024_2025` | int | 0 – 45,106; median 0; never exceeds peak headcount |
| `Current_Headcount_2026` | int | 1 – 89,924; median 51. Dated 2026 — same period as the label |
| `Investor_Tier` | category | 6 tiers, Angel/Individual → Tier 1 VC |
| `AI_Adoption_Level` | category | 5 levels, `None` → `AI-Native`. **The only column with missing values:** 2,434 blanks (9.74%) |
| `Acquisition_Status` | category | **Target.** Independent 63.06% · Acquired 14.85% · Closed 13.81% · IPO 4.31% · Merger 3.97% |

`AI_Adoption_Level` blanks are **2,434 rows (9.74%), and they are not uniform across strata**. The
rate ranges **6.33%–14.71%** across the 20 `Domain` values and 8.44%–11.31% across `Funding_Stage`.
A permutation test puts the observed 8.38 pp domain spread beyond the null's 95th percentile of
6.43 pp (**p = 0.007**), while the funding-stage spread sits comfortably inside its null (p = 0.34).
So the blanks are missing at random *conditional on domain*, not completely at random — which rules
out filling them with a global mode, and is why the modeling choice is an explicit `(Missing)`
level. They are left unimputed at load time so that imputation stays an explicit, reviewable
decision.

> **A correction, and why it happened.** An earlier version of this README claimed the blanks were
> "near-uniform across strata (9.45%–14.71%), consistent with missing-completely-at-random". The
> range matched no grouping of the reference download and the inference was contradicted at
> p = 0.007. It survived because it was one of the few numbers here with no function behind it.
> It now comes from `descriptive.missingness_by_strata()` and `diagnostic.missingness_permutation()`
> and is pinned by two tests. This is the failure mode the "every number has a function and a test"
> rule exists to prevent, so the incident is recorded in `.claude/rules/arch-data-storage.md`.

> **Loading gotcha, the hard way.** `AI_Adoption_Level` has a valid level spelled `None`, meaning
> *no AI adoption*. Pandas' default `na_values` includes the string `"None"`, so a plain
> `read_csv` silently converts those 1,844 rows to NaN — inflating missingness from 2,434 to 4,278
> and reducing the column from five levels to four. `load_raw` passes `keep_default_na=False` and
> there is a regression test for it. This is the project's cheapest lesson in why a data contract
> beats a spot check.

## Research questions

1. **Primary — company outcome.** Can outcome be predicted from what was knowable *before* the
   outcome, and does any model beat its baseline by a margin wider than its confidence interval?

   **Answered: no.** The leakage-controlled model reaches PR-AUC 0.193 against a 0.141 base rate,
   but the same estimator given only `Runway_Months_2024` reaches 0.191. The paired bootstrap
   interval on the difference is [−0.005, +0.009] — it contains zero, and the difference is a
   quarter of one fold standard deviation. At the default threshold the model flags 2 of 24,467
   companies, so its accuracy (85.88%) is *below* the majority baseline's (85.89%). It is a weak
   ranker and not a classifier at all.

2. **Does AI adoption matter?** The dataset's premise.

   **Answered: a real effect, far too small to act on — and it reverses.** The `None` → `AI-Native`
   closed-rate gap is 2.95 pp with a 95% interval of [1.11, 5.01], which *excludes zero*, and a
   permutation test agrees (p = 0.004). So "no effect" — this README's earlier phrasing — was
   imprecise: the effect is detectable and immaterial, which is a different claim. It also reverses
   inside `Tier 2 VC` (5,157 rows), where AI-native companies close slightly *more* often.

3. **What did the generator actually encode?** How much of any model's apparent skill is
   rediscovered scaffolding?

   **Answered: one threshold, and almost nothing else.** (i) The 6-month runway step is the only
   strong effect, and a model given that column alone matches the full model. (ii) `Domain` and
   `Country` closed-rate spreads sit *below* the median spread that shuffling the outcome produces
   (p = 0.54 and 0.66) — not weak signal, no signal. (iii) Scrambling runway costs 0.033 PR-AUC;
   scrambling any other feature group costs at most 0.005. (iv) A depth-unlimited tree memorises
   perfectly (train PR-AUC 1.000) while its out-of-fold score falls to 0.145, indistinguishable
   from the 0.141 base rate — there is nothing beyond the rule to learn, and the best depth is 2.
   (v) The blanks are domain-dependent. The generator encoded a threshold, a weak monotone AI
   effect, a correlated size block, and non-random missingness.

## Methodology

### The four tiers

Each tier is a module with one `run_all()`, one test file, and one notebook driving it. The tiers
mirror `references/Day2`–`Day5`, restructured to satisfy this project's rules.

| Tier | Module | Question | Tests | Rows |
|---|---|---|---|---|
| — | `audit.py` | Is the data coherent? | `test_audit.py` | 25,000 |
| Descriptive | `descriptive.py` | What does the file contain? | `test_descriptive.py` | 25,000 |
| Diagnostic | `diagnostic.py` | Why does the outcome vary — and does it? | `test_diagnostic.py` | 24,467 |
| Predictive | `models/supervised.py`, `models/unsupervised.py` | What can be predicted, against what baseline? | `test_supervised.py`, `test_unsupervised.py` | 24,467 |
| Prescriptive | `models/prescriptive.py` | What would acting on it be worth? | `test_prescriptive.py` | 24,467 |

**Two row counts, on purpose.** The descriptive tier profiles the file as downloaded, which is what
"Dataset at a glance" describes. Every tier that touches `Funding_Stage` first drops the 533 rows
where its `IPO` level determines the target. Each `run_all()` returns the row count it used, and
every table below states which frame it came from.

### The feature-timing contract

`Acquisition_Status` is observed as of 2026. Several columns date themselves in their names, which
makes leakage control concrete rather than a matter of taste:

| Group | Columns | Status |
|---|---|---|
| **Structural** | `Domain`, `Country`, `City`, `Founding_Year`, `Investor_Tier` | Usable — fixed at founding |
| **Explicitly pre-outcome** | `Runway_Months_2024`, `Peak_Headcount_2023` | Usable — dated before the label |
| **Undated** | `Funding_Stage`, `Total_Funding_USD_Millions`, `Valuation_USD_Millions`, `Revenue_ARR_Millions`, `Monthly_Burn_Rate_Millions`, `AI_Adoption_Level` | Usable as *as-of-latest* — a stated assumption, not a fact the data supports |
| **Excluded** | `Layoffs_2024_2025`, `Current_Headcount_2026` | **Never used.** Contemporaneous with or after the label |

Excluding `Current_Headcount_2026` costs real apparent accuracy, which is the point: a 2026
headcount cannot predict a 2026 outcome, it partly *is* the outcome. `Layoffs_2024_2025` spans into
the label period, so it goes too.

This lives in [`config.py`](src/startup_outcomes/config.py) as named column groups, and models build
their design matrix from `model_features()` rather than by dropping columns by hand — so an excluded
column cannot reach a model by accident. `assert_no_leakage()` is the guard, and
`model_features(include_undated=False)` gives the conservative variant that relies only on
explicitly dated and structural fields.

Separately, `drop_leaky_stage()` removes the 533 rows where `Funding_Stage == 'IPO'` determines the
label. Dropping is the default over collapsing the level because it leaves no ambiguity about what
the model saw.

### Evaluation

- **Baselines before models.** Majority class (86.19% binary accuracy), then a stratified random
  baseline, then a single-feature model on the runway threshold. A gradient-boosted model that
  cannot clear these is reported as not clearing them.
- **Accuracy is not the metric.** At a 13.81% positive rate, accuracy rewards predicting "survives"
  forever. Balanced accuracy, PR-AUC, and per-class recall lead; calibration is checked, not
  assumed.
- **Stratified k-fold** with the seed in `config.RANDOM_SEED`, stratified on the target.
- **Effect sizes with intervals**, not point estimates. A 3.0 pp spread across 25,000 rows needs an
  interval before it means anything.
- **Collinearity is handled explicitly.** The size measures are near-identical on logs — funding↔
  valuation r=0.974, funding↔revenue r=0.968, revenue↔valuation r=0.933, burn↔revenue r=0.944,
  peak↔current headcount r=0.989. Coefficients on these are uninterpretable individually, so they
  are regularised or reduced, and feature importances among them are read as a group. This is
  enforced rather than intended: `grouped_importance()` has no key for an individual size column,
  and a test asserts it.
- **Pooled out-of-fold predictions are the reporting unit.** One `cross_val_predict` vector per
  model; every metric computed once on it. Averaging per-fold metrics instead moves PR-AUC by
  ~0.001 — the same order as the model-versus-baseline difference under study — so the convention
  cannot stay implicit. Per-fold values are reported only as the *spread*, and that spread is the
  yardstick every comparison is measured against.
- **There is no train/test split anywhere.** With 3,453 positives a 25% holdout carries a PR-AUC
  standard error near 0.012, larger than every effect being measured. Changing only the split seed
  moves the score by 0.0067 — four times the model-versus-baseline difference. The one consequence
  is that the tuned figure has no untouched data behind it; it is labelled `optimistic` and is
  never compared against the out-of-fold numbers.
- **Model metrics are quoted and pinned at three decimals**, and every pin is paired with a band
  test. If a pin fails and its band passes, a library upgrade moved a digit; if both fail, the
  finding changed. `best_params_` is never quoted — with eight near-tied grid combinations it
  documents a coin toss.
- **Intervals come from a seeded percentile bootstrap** (1,000 resamples), paired within a
  comparison. Spreads across many levels get a permutation null instead, because max-minus-min over
  twenty noisy estimates is positive even when every true rate is identical.

## Findings so far

Established and reproducible; see the audit output for the full set.

1. **The data is synthetic** — seven independent internal contradictions, tabulated above.
2. **AI adoption has a real but immaterial effect**: 15.84% → 12.84% closed rate from `None` to
   `AI-Native`, a 2.95 pp gap with a 95% interval of **[1.11, 5.01]** and a permutation p of
   **0.004**. The interval excludes zero, so this is not a null result — it is a detectable effect
   too small to act on. **And it reverses**: inside `Tier 2 VC` (5,157 rows) the gap is −0.54 pp,
   with AI-native companies closing slightly more often. Quoting the headline without that caveat
   would be Simpson's paradox in action.
3. **Sector and geography carry no signal at all.** The closed-rate spreads across 20 domains
   (4.74 pp) and 20 countries (6.09 pp) sit *below* the median spread that shuffling the outcome
   produces at the same group sizes — p = 0.54 and 0.66. Not weak signal: none. Any importance a
   model assigns to these columns is noise.
4. **Runway is the one strong predictor, and it is a step function** at exactly 6 months
   (23.41% → 10.98% on all rows; a 12.33 pp step with interval [11.14, 13.51] on the analysis
   frame), flat on both sides.
5. **Two leakage paths exist** and both are now closed by construction: the `IPO` funding stage, and
   the outcome-contemporaneous 2026 columns.
6. **Valuations are internally consistent but not market-realistic** — median valuation/revenue
   multiple 31.6×, p90 131×, p99 355×.
7. **No model beats a single column.** Thirteen leakage-controlled features give PR-AUC **0.193**;
   the same estimator given `Runway_Months_2024` alone gives **0.191**. Difference +0.0016, interval
   [−0.005, +0.009], fold spread 0.006. Logistic regression (0.171) and k-nearest-neighbours (0.157)
   score *worse* than the one column. The model does clear the coarse two-level threshold rule
   (0.186) — that says the runway column carries more than the six-month step, not that the other
   twelve features help.

   | Model | Features | PR-AUC | Fold SD |
   |---|---|---|---|
   | random forest | 13 | 0.194 | 0.004 |
   | gradient boosting | 13 | 0.193 | 0.006 |
   | **runway only (same estimator)** | **1** | **0.191** | 0.006 |
   | decision tree (depth 4) | 13 | 0.190 | 0.005 |
   | runway threshold rule | 1 | 0.186 | 0.005 |
   | logistic regression | 13 | 0.171 | 0.008 |
   | k-nearest neighbours | 13 | 0.157 | 0.004 |
   | majority baseline | — | 0.141 | 0.000 |
   | *leakage demonstration — not a result* | *15* | *0.235* | — |

8. **The only thing that improves the model is leakage.** Adding the two excluded 2026 columns
   lifts PR-AUC to **0.235** — a gain of 6.8 fold standard deviations, where every legitimate
   change sits inside one. That contrast is the argument for the timing contract, and it is why
   the row above is labelled rather than ranked.
9. **The undated-column assumption is immaterial.** Dropping all six undated columns costs 0.004
   PR-AUC, well under one fold standard deviation. The project's largest documented judgement call
   does not change any conclusion.
10. **Tuning, extra models and unlabelled structure add nothing.** Grid search buys 0.007 (inside
    two fold SDs). Clustering finds one weak split (best silhouette 0.42 at k=2) that is purely a
    size gradient — median funding 3.9 against 158.3 — separating the outcome by 3.41 pp. A single
    principal component carries 68% of the numeric variance, restating the collinearity finding
    from another direction.
11. **`AI_Adoption_Level` blanks are domain-dependent** (6.33%–14.71%, permutation p = 0.007), not
    missing completely at random — correcting a previously documented claim.

## Prescriptive layer — a decision framework, not a recommendation

The data is synthetic, so a tier that emits "target these thirty companies, expected net X" would
be fake advice. What the tier demonstrates is the arithmetic that turns a ranking into a decision,
and applied honestly to this ranking that arithmetic returns a negative verdict — which is the
result.

Costs and values are **unitless relative weights**, chosen to demonstrate the method. No quantity
here carries a currency, because there is no currency behind it. Every returned dict carries
`basis: "synthetic-data-methodology-demo"` as a key rather than a caption, since a caption is the
first thing lost when a number is pasted elsewhere.

- **The cost-optimal policy is very nearly a blanket policy.** At a 10:1 miss-to-false-alarm ratio
  the cheapest threshold flags **97.1%** of companies and saves **1.69%** against simply flagging
  everyone. The model is steering almost nothing. (A cheapest threshold quoted without that
  comparator would imply the opposite.)
- **The default 0.5 threshold is exactly as good as doing nothing**, because the model's maximum
  predicted probability is 0.507. Nothing about 0.5 is principled; the analytic indifference point
  at these weights is 0.091.
- **The model overstates its own hit rate by more than double.** Among the top 30 it predicts a
  44.3% closure rate; the true labels say **20.0%**. The reference workbooks use the mean predicted
  score as the hit rate, which lets a model grade its own homework — both numbers are reported here
  and the gap is a calibration finding.
- **Targeting the top 30 is indistinguishable from picking at random.** Six events, realised rate
  20.0% with a 95% interval of **[6.67, 36.67]** that contains the 14.11% base rate. The apparent
  1.42× lift is not resolvable at that capacity. Top-decile lift is 1.68×, top-fifth 1.60×.
- **The intervention does not pay.** Net −840 units at the illustrative 40% success rate;
  break-even needs **55.6%**. Whether that is plausible is not something this data can inform, and
  that sentence is the honest end of the analysis.

The three recommendations, each with what would falsify it, are in
[`notebooks/05_prescriptive.ipynb`](notebooks/05_prescriptive.ipynb). The first is: do not deploy
this model as a targeting tool.

## Repo layout

```
data/raw/                       # the CSV; git-ignored, see Reproducibility
src/startup_outcomes/
  config.py                     # paths, seed, and the feature-timing contract
  load.py                       # typed loader, checksum verification, leaky-stage removal
  intervals.py                  # bootstrap intervals and permutation tests
  audit.py                      # data-integrity checks
  features.py                   # engineered columns and the only design-matrix constructor
  descriptive.py                # tier 1 — what the file contains
  diagnostic.py                 # tier 2 — why the outcome varies, and whether it does
  plots.py                      # figures; computes no statistic, never imports pyplot
  report.py                     # composes every tier; imported by nothing
  models/
    protocol.py                 # splitter, preprocessing, metrics, out-of-fold convention
    baselines.py                # majority, stratified, and the runway threshold rule
    supervised.py               # tier 3a — the zoo, tuning, importance, leakage demo
    unsupervised.py             # tier 3b — clustering and PCA
    prescriptive.py             # tier 4 — thresholds, capacity, expected value
    card.py                     # the model card, rendered from a pinned dict
notebooks/                      # 01 profile, then one notebook per tier (02–05)
tests/                          # one file per module, plus test_repo_invariants.py
references/                     # the course workbooks this mirrors; never imported or linted
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/). The system Python is not used — everything runs in the
project venv via `uv run`.

```bash
uv sync                                    # create the venv and install from the lockfile
uv run pytest                              # every documented finding; ~50s
uv run ruff check . && uv run ruff format --check .
uv run python -c "from startup_outcomes.report import run_all; print(run_all())"
uv run python -c "from startup_outcomes.report import summary_text; print(summary_text())"
uv run jupyter lab                          # exploration
```

`uv run pytest tests/test_data_contract.py tests/test_audit.py` is the sub-second loop while
editing; the full suite spends most of its time cross-validating. Individual tiers:

```bash
uv run python -c "from startup_outcomes.descriptive import run_all; print(run_all())"
uv run python -c "from startup_outcomes.diagnostic import run_all; print(run_all())"
uv run python -c "from startup_outcomes.models.supervised import run_all; print(run_all())"
uv run python -c "from startup_outcomes.models.prescriptive import run_all; print(run_all())"
uv run python -c "from startup_outcomes.models.card import model_card; print(model_card())"
```

The dataset is not in the repo. Download it from the
[Kaggle page](https://www.kaggle.com/datasets/beamhonor0911/global-tech-ai-startups-insights-2020-2026)
and place it at `data/raw/global_tech_startups_2026.csv`, or:

```bash
uv run kaggle datasets download -d beamhonor0911/global-tech-ai-startups-insights-2020-2026 \
  -p data/raw --unzip
```

## Reproducibility

- **Data integrity.** The reference download is
  `sha256 c843a3c03495b75c35713126cc3df8f902f8188abdfa876fff5bc4a9f0227ee2`. Verify with
  `shasum -a 256 data/raw/global_tech_startups_2026.csv`, or in code via
  `load_raw(verify_checksum=True)`.
- **Dependencies** are pinned in `uv.lock`; `uv sync` reproduces the environment exactly. The model
  metrics above were generated under **scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1**
  on Python 3.12. `pyproject.toml` floors scikit-learn at 1.6 because
  `categorical_features="from_dtype"` only became the default there, and caps it below 2.0 so a
  major bump is a reviewed act rather than a `uv lock --upgrade` side effect.
- **Randomness** comes from the single `config.RANDOM_SEED`. Every line in `src/` that mentions
  `random_state`, `np.random` or a seed names that constant on the same line, which makes the rule
  greppable and is asserted by `tests/test_repo_invariants.py`.
- **Raw data is immutable, and nothing else is written either.** No model, table or figure is
  persisted: the full pipeline recomputes from `data/raw` in about 45 seconds, and a saved artifact
  would be a second source of truth that drifts from the code that made it. `DATA_INTERIM` and
  `DATA_PROCESSED` exist as declared destinations should that ever change.
- **Documented numbers are tested.** If a finding here drifts from the data, the tier's test file
  fails rather than the README quietly going stale. Model metrics additionally carry a band test,
  so a scikit-learn upgrade that moves a third decimal is distinguishable from a changed
  conclusion — the failure message says which and what to do about it.

## Limitations — what this project does not claim

- **Nothing here describes real startups.** No survival rate, sector risk, valuation multiple, or
  AI-adoption effect should be repeated as a fact about the world.
- **No causal claims.** The data has no randomisation or instrument, and its correlations were
  authored by a generator. Even the runway threshold is a rule someone wrote, not evidence that
  short runway causes failure.
- **Findings describe a simulator.** Where this project reports an effect, the honest reading is
  "the generating process encodes this", not "this is true of startups". Even the *missingness* is
  a generator artifact: the blanks vary with `Domain` from 6.33% to 14.71% (p = 0.007), so the
  convenient missing-completely-at-random assumption is not available — and was previously
  documented here in error.
- **The undated columns rest on an assumption — and it turns out not to matter.** Six columns carry
  no as-of date; using them as pre-outcome features is a documented judgement call, and
  `model_features(include_undated=False)` exists so results can be checked without them. Checked:
  dropping all six costs 0.004 PR-AUC, under one fold standard deviation.
- **The prescriptive layer is arithmetic, not advice.** Its costs, capacities and values are
  unitless weights chosen to demonstrate a method. No ranked list, threshold, lift or break-even
  rate here is a recommendation about anything, and none of the money-shaped quantities carry a
  currency because there is no currency behind them.
- **The model is not usable as a decision rule even inside the simulation.** It does not beat one
  column, it flags 2 of 24,467 companies at the default threshold, it is over-confident at the top
  of its ranking (44.3% predicted against 20.0% realised), and under the illustrative weights the
  cost-optimal policy flags 97% of companies.
- **No untouched holdout exists.** Every figure comes from 5-fold out-of-fold predictions over all
  24,467 rows. The grid-search figure is additionally optimistic because the grid was selected on
  those same folds; nested cross-validation was not run because the tuning gain is already smaller
  than the fold-to-fold spread.
- **Model metrics are environment-bound.** They are pinned at three decimals under the library
  versions recorded in Reproducibility. An upgrade can move the third decimal with nothing being
  wrong.

## License and citation

Dataset: *Global Tech & AI Startups Insights 2020–2026* by Kaggle user `beamhonor0911` —
<https://www.kaggle.com/datasets/beamhonor0911/global-tech-ai-startups-insights-2020-2026>

<!-- TODO(license): the Kaggle page is client-rendered, so its stated license and author
description could not be read programmatically. Record the exact license string here, and confirm
whether the author labels the data synthetic — the conclusion above is inferred from the data
itself. Until the license is confirmed, the CSV stays git-ignored rather than committed. -->

**License: unconfirmed.** Check the Kaggle page's license field before redistributing the CSV or
publishing derived data. The dataset is deliberately not committed to this repo for that reason.

Project code is available under the MIT License.
