# `notebooks/` — thin drivers, one per tier

## Purpose

Exploration and presentation only. Every notebook here imports from `src/startup_outcomes/` and
defines nothing: logic in a notebook is untestable, unlintable and unimportable, and a number
computed in a cell can reach the README with nothing pinning it.

## Structure

| Notebook | Tier | Mirrors | Runtime |
|---|---|---|---|
| `01_profile.ipynb` | The raw file and the audit | — | ~1 s |
| `02_descriptive_diagnostic.ipynb` | What happened, and why | Day 2 | ~10 s |
| `03_predictive_baselines.ipynb` | Baselines before models | Day 3 | ~20 s |
| `04_predictive_selection.ipynb` | Selection, tuning, clustering, PCA | Day 4 | ~30 s |
| `05_prescriptive.ipynb` | Scores into decisions | Day 5 | ~10 s |

Each follows the same shape: a markdown header restating the import-only rule, one import cell
ending in `load_raw(verify_checksum=True)`, alternating markdown and code driving one `run_all()`
plus the `plots.py` functions, a numbered deliverable, and a trailing empty **Scratch** cell.

## Conventions

- **No `def`, no `class`.** Verified by `grep -n '"\(def\|class\) ' notebooks/*.ipynb` and by
  `tests/test_repo_invariants.py`.
- **Commit with outputs stripped** (Kernel → Restart & Clear Output). Also asserted by a test.
  Outputs make diffs unreadable, and because `data/` is git-ignored a committed output table is a
  back door for the dataset to reach version control.
- **Must survive Restart & Run All.** That is the only honest check that a notebook still works.
- **Knobs are passed as arguments, never edited into a module.** The reference workbooks ask you to
  change `COST_MISS` or `K` in place; here `prescriptive.cost_threshold_sweep(cost_miss=3.0)` does
  the same thing reviewably, and the module default is what the README quotes.
- **Notebooks choose nothing.** `chosen_k` and the best tree depth are computed, not hand-set.
- **Notebooks are linted and formatted** like any other source — `references/` is excluded from the
  gate, `notebooks/` is not.

## Files Summary

- `01_profile.ipynb` — shape, the audit, what a model is allowed to see, and a pointer to 02–05.
- `02_descriptive_diagnostic.ipynb` — missingness map, the five-strategy imputation table, outliers
  by both rules, the group summary, the six-panel grid, effect sizes with intervals, permutation
  nulls, and the Simpson's-paradox reversal check.
- `03_predictive_baselines.ipynb` — the leakage contract, three baselines, the first model, the
  accuracy trap shown explicitly, `beats_baseline`, and the overfitting sweep.
- `04_predictive_selection.ipynb` — seed instability, the cross-validated zoo, tuning, grouped
  importance, clustering with elbow and silhouette, PCA, and the model card.
- `05_prescriptive.ipynb` — response curve, ranked actions, the cost-threshold sweep,
  capacity-constrained expected value, sensitivity, three recommendations, and the capstone summary.

## Tips & Key Notes

- **`%matplotlib inline` is required in the first cell.** `plots.py` returns bare `Figure` objects
  and never imports pyplot, so without the magic there is no registered PNG formatter and the
  figures silently do not render — the cell produces no output and no error. This was caught by
  executing the notebooks, not by reading them.
- **`verify_checksum=True` in the first cell.** A published result should fail loudly on a
  different download rather than quietly reporting different numbers.
- **`04` is the slow one** (~30 s), almost all of it cross-validation.
- **Charts are never saved.** Nothing in this project writes a figure to disk; run the notebook to
  see them.
- **To check a notebook end to end without touching the repo copy:**
  `uv run jupyter nbconvert --to notebook --execute --output-dir /tmp/x notebooks/04_*.ipynb`
