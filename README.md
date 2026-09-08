# Startup Outcomes: A Leakage-Controlled Analysis

An applied data science project on the Kaggle dataset
[Global Tech & AI Startups Insights 2020–2026](https://www.kaggle.com/datasets/beamhonor0911/global-tech-ai-startups-insights-2020-2026)
(25,000 companies × 17 columns), predicting company outcome — independent, acquired, closed,
merged, or IPO.

The dataset turns out to be **synthetic**, with most of its headline effects absent or engineered.
Rather than work around that, this project treats it as the subject: the deliverables are a
reproducible data-integrity audit, a modeling design where leakage is prevented by construction
rather than by good intentions, and results reported against honest baselines — including where
the answer is "no effect". Every number below is produced by
[`src/startup_outcomes/audit.py`](src/startup_outcomes/audit.py) and pinned by
[`tests/test_audit.py`](tests/test_audit.py), so the claims here can be re-derived in one command.

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

`AI_Adoption_Level` blanks are near-uniform across strata (9.45%–14.71% by domain and funding
stage), consistent with missing-completely-at-random. They are left unimputed at load time so that
imputation is an explicit, reviewable modeling decision.

> **Loading gotcha, the hard way.** `AI_Adoption_Level` has a valid level spelled `None`, meaning
> *no AI adoption*. Pandas' default `na_values` includes the string `"None"`, so a plain
> `read_csv` silently converts those 1,844 rows to NaN — inflating missingness from 2,434 to 4,278
> and reducing the column from five levels to four. `load_raw` passes `keep_default_na=False` and
> there is a regression test for it. This is the project's cheapest lesson in why a data contract
> beats a spot check.

## Research questions

1. **Primary — company outcome.** Can outcome be predicted from what was knowable *before* the
   outcome, and does any model beat the 86.19% majority baseline by a margin wider than its
   confidence interval? Reported as binary closed-vs-rest and as the 5-class problem.
2. **Does AI adoption matter?** The dataset's premise. The descriptive answer looks like no — a
   3.0 pp spread, monotone but tiny — so this gets a formal test with an effect size and interval,
   and a null result is a publishable outcome of this project, not a failure.
3. **What did the generator actually encode?** The 6-month runway threshold is one recoverable rule.
   Finding the rest characterises how much of any model's apparent skill is rediscovered
   scaffolding.

## Methodology

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
  are regularised or reduced, and feature importances among them are read as a group.

## Findings so far

Established and reproducible; see the audit output for the full set.

1. **The data is synthetic** — seven independent internal contradictions, tabulated above.
2. **AI adoption does not meaningfully affect outcomes** in this data: 15.84% → 12.84% closed rate
   from `None` to `AI-Native`. Monotone in the expected direction, but a 3.0 pp spread.
3. **Sector and geography carry almost no signal** — 4.82 pp and 5.83 pp closed-rate spread across
   20 levels each, consistent with noise around a constant base rate.
4. **Runway is the one strong predictor, and it is a step function** at exactly 6 months
   (23.41% → 10.98%), flat on both sides.
5. **Two leakage paths exist** and both are now closed by construction: the `IPO` funding stage, and
   the outcome-contemporaneous 2026 columns.
6. **Valuations are internally consistent but not market-realistic** — median valuation/revenue
   multiple 31.6×, p90 131×, p99 355×.

## Repo layout

```
data/raw/                     # the CSV; git-ignored, see Reproducibility
src/startup_outcomes/
  config.py                   # paths, seed, and the feature-timing contract
  load.py                     # typed loader, checksum verification, leaky-stage removal
  audit.py                    # every check behind the numbers in this README
notebooks/01_profile.ipynb     # exploration; logic imported from src/, not defined here
tests/
  test_data_contract.py       # shape, dtypes, cardinality, missingness, nesting, leakage guards
  test_audit.py               # pins every documented finding
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/). The system Python is not used — everything runs in the
project venv via `uv run`.

```bash
uv sync                                    # create the venv and install from the lockfile
uv run pytest                              # data contract + audit findings
uv run ruff check . && uv run ruff format --check .
uv run python -c "from startup_outcomes.audit import run_all; print(run_all())"
uv run jupyter lab                          # exploration
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
- **Dependencies** are pinned in `uv.lock`; `uv sync` reproduces the environment exactly.
- **Randomness** comes from the single `config.RANDOM_SEED`.
- **Raw data is immutable.** Nothing writes to `data/raw/`; derived data goes to sibling directories.
- **Documented numbers are tested.** If a finding in this README drifts from the data,
  `tests/test_audit.py` fails rather than the README quietly going stale.

## Limitations — what this project does not claim

- **Nothing here describes real startups.** No survival rate, sector risk, valuation multiple, or
  AI-adoption effect should be repeated as a fact about the world.
- **No causal claims.** The data has no randomisation or instrument, and its correlations were
  authored by a generator. Even the runway threshold is a rule someone wrote, not evidence that
  short runway causes failure.
- **Findings describe a simulator.** Where this project reports an effect, the honest reading is
  "the generating process encodes this", not "this is true of startups".
- **The undated columns rest on an assumption.** Four financial columns carry no as-of date; using
  them as pre-outcome features is a documented judgement call, and
  `model_features(include_undated=False)` exists so results can be checked without them.

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
