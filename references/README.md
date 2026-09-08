# `references/` — the course workbooks this project mirrors

## Purpose

Four teaching notebooks and their shared helper module, kept as the methodological source the four
analytics tiers are checked against. They are read, not run, and never imported.

## Structure

| File | Tier it corresponds to | This project's version |
|---|---|---|
| `Day2_Team_Workbook.ipynb` | Descriptive & diagnostic | `descriptive.py`, `diagnostic.py`, notebook 02 |
| `Day3_Team_Workbook.ipynb` | Predictive I — baselines | `models/baselines.py`, notebook 03 |
| `Day4_Team_Workbook.ipynb` | Predictive II — selection, clustering | `models/supervised.py`, `models/unsupervised.py`, notebook 04 |
| `Day5_Team_Workbook.ipynb` | Prescriptive | `models/prescriptive.py`, notebook 05 |
| `mlkit.py` | The shared helpers, pasted into Days 3–5 | Replaced by `models/protocol.py` |

## Conventions

- **The workbooks themselves are not committed.** Only this README is. They are third-party course
  material whose redistribution licence is unconfirmed — the same caution that keeps the dataset out
  of the repo. A fresh clone will therefore have this directory documented but empty; nothing in the
  project depends on its contents. Commit them, or add the directory to `.gitignore`, once the
  licence is established.
- **Nothing here is imported, tested, or linted.** `pyproject.toml` excludes the directory from
  ruff, and `tests/test_repo_invariants.py` asserts that no module imports `mlkit`.
- **Do not reformat these files.** Reformatting would destroy the fidelity of the source the tiers
  are checked against — the same argument that excludes `.claude/` from the formatter.

## Files Summary — why `mlkit.py` is deliberately not used

It is a good teaching module and it breaks four of this project's rules:

| `mlkit` does | Why it cannot be used here |
|---|---|
| `make_features` builds the matrix by *dropping* columns, capped at 25 categories | A new leaky column would default to *included*. It would also silently drop `City`, which has 58 levels. Features come from `config.model_features()` instead. |
| `seed=42` hard-coded at roughly a dozen call sites | One seed, in config, greppable. |
| `warnings.filterwarnings("ignore")` at import | Lint is the gate here; a `FutureWarning` fails the build. |
| A module-level mutable `RESULTS = {}` accumulates the leaderboard | Makes a notebook's output depend on which cells ran before. `protocol.leaderboard()` recomputes from its argument. |
| One 75/25 split; Day 5 then scores all rows with a model fitted on 75% of them | Inflates every lift, threshold and expected value. Replaced by pooled out-of-fold predictions — the single most consequential difference between the two codebases. |
| `hit_rate = acted["score"].mean()` | The model grading its own homework. Realised labels are used, and the 44.3%-vs-20.0% gap is reported as a calibration finding. |

## Tips & Key Notes

- **Where this project deliberately diverges**, beyond the table above: cross-validation only with
  no holdout; a bootstrap interval or permutation null on every reported effect (the workbooks use
  effect size plus `n` and no inferential statistics at all); an explicit `(Missing)` level rather
  than mode-filling; clustering restricted to continuous columns; and nothing persisted to disk.
- **Where the workbooks were right and it mattered:** the reversal check. Day 2 calls it "the most
  important cell in this workbook", and it is the only thing in the whole methodology that caught
  the `Tier 2 VC` sign flip in the AI-adoption effect.
- **`mlkit.py` is pasted verbatim as a cell in Days 3, 4 and 5** — the file is the extracted single
  copy, not an import target for the notebooks either.
