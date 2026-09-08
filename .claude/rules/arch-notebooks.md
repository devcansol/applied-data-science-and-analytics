---
description: notebooks, exploration, output hygiene, promotion into src, cell-order independence
globs: ["notebooks/**/*.ipynb", "src/startup_outcomes/plots.py"]
alwaysApply: false
---
> **Architecture family:** Notebooks & Exploration. See also: `.claude/rules/arch-data-storage.md`, `.claude/rules/arch-data-pipeline.md`, `.claude/rules/arch-modeling.md`

# Notebooks & Exploration

## Scope

#### ✅ Design Decision: Notebooks explore; `src/` defines

**Context:**
Logic defined in a notebook is untestable, unlintable, and unimportable. The gate
(`uv run ruff check .`) does not cover notebooks, so a function living in one gets no review from
the harness at all — and a number computed there can reach the README with nothing pinning it.

**Pattern:**

```python
# ✅ GOOD — the notebook is a thin driver
from startup_outcomes import config
from startup_outcomes.audit import run_all
from startup_outcomes.load import load_raw

df = load_raw(verify_checksum=True)
for check, result in run_all(df).items():
    print(f"{check}:\n  {result}\n")

# ❌ BAD — a definition that belongs in src/, where it can be tested
def closed_rate_by(df, column):
    return df.groupby(column)["Acquisition_Status"].apply(lambda s: (s == "Closed").mean())
```

Verify with:
```bash
grep -n '"\(def\|class\) ' notebooks/*.ipynb    # expect no output
```

Promotion is the normal path: explore in a cell, and once it earns a place, move it to
`src/startup_outcomes/`, give it a test, and import it back.

**Rationale:**
1. Anything worth keeping is worth linting and testing
2. Any number worth quoting needs an audit function behind it (see the documentation process rule)
3. Keeps notebooks short enough to read

**When NOT to Change This:**
- No function or class definitions in committed notebooks
- Never quote a figure in the README that was computed only in a notebook cell

**When to Revisit:**
- Notebook count grows enough to justify a shared plotting module — which is itself `src/` code

**Alternatives Considered:**
- ❌ `%run`-style shared notebooks: import order becomes implicit and untestable
- ❌ `nbdev`-style export: real machinery for a project with one notebook

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Hygiene

#### ✅ Design Decision: Commit notebooks without outputs, and treat them as order-independent

**Context:**
Notebook outputs make diffs unreadable, inflate the repo with base64 image blobs, and can carry
data into version control that the repo deliberately git-ignores. Separately, a notebook that only
works when cells are run in a particular order is a reproducibility trap.

**Pattern:**

```
✅ Clear all outputs before committing (Kernel → Restart & Clear Output)
✅ A notebook must produce the same results under Restart & Run All
❌ Committing execution counts and rendered outputs
❌ A cell that mutates a frame in place, so re-running it changes the result
```

Because `data/` is git-ignored, a committed output table is a way for the dataset to leak into the
repo by the back door — the same license concern that keeps the CSV out (see the data-storage rule).

**Rationale:**
1. Reviewable diffs — cell source only
2. Restart & Run All is the only honest check that a notebook still works
3. Keeps ignored data out of version control

**When NOT to Change This:**
- Do not commit rendered outputs to "show results" — findings belong in the README, backed by
  `audit.py`
- Do not rely on in-place mutation in a cell (see the pipeline rule's pure-transform decision)

**When to Revisit:**
- If reviewing rendered output becomes genuinely necessary, add `nbstripout` as a pre-commit hook
  rather than relaxing the rule

**Alternatives Considered:**
- ❌ Commit outputs for reviewer convenience: unreadable diffs, and data in git
- ❌ Jupytext paired scripts: a reasonable option, but more moving parts than this project needs

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Tier Notebooks

#### ✅ Design Decision: One notebook per tier, and knobs are arguments rather than edited constants

**Context:**
The reference workbooks put a human decision in a cell and ask you to change it — `BEST_DEPTH = 4`,
`K = 3`, `COST_MISS = 10`. Pedagogically that is the most valuable part of them and reproducibly it
is the worst: an edited constant is an unreviewable, untestable change to a documented number.

**Pattern:**

```python
# ✅ GOOD — the knob is an argument; the module default is what the README quotes
prescriptive.cost_threshold_sweep(target, scores, cost_miss=3.0)

# ✅ GOOD — a choice the workbooks leave to a human is computed here
best_k = unsupervised.chosen_k(matrix)

# ❌ BAD — edit this cell and re-run
COST_MISS = 3
```

Notebooks are `02_descriptive_diagnostic`, `03_predictive_baselines`, `04_predictive_selection`,
`05_prescriptive`, each a thin driver over one `run_all()` plus `plots.py`. Every knob is both a
documented module constant *and* a keyword argument.

**Rationale:**
1. A defaulted parameter is reviewable and testable; an edited constant is neither
2. The default is the value the README quotes, so the two cannot drift
3. The pedagogical point survives: "change `cost_miss` to 3 and re-run" is still one edit, it just
   happens at the call site instead of inside a module

**When NOT to Change This:**
- Do not move a threshold or a cluster count into a notebook cell
- Do not let a notebook choose anything a function could compute

**When to Revisit:**
- A genuinely irreducible human choice appears; it becomes a `#:` constant in `src/` with a test
  that re-derives it

**Alternatives Considered:**
- ❌ Papermill parameters: real machinery for four notebooks
- ❌ Keep the workbooks' edit-in-place style: the numbers stop being reproducible

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: Plots live in `src/plots.py`, return Figures, and never import pyplot

**Context:**
Four notebooks drawing their own charts would duplicate plotting code across four unlintable,
untestable places — exactly what the promotion rule exists to prevent. This rule file's own "When
to Revisit" forecast the module.

**Pattern:**

```python
# ✅ GOOD — built directly, no global registry
figure = Figure(figsize=(7, 4), layout="constrained")
axes = figure.subplots()

# ❌ BAD — mutates global state, and renders twice in a notebook
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
```

Two rules: **no plot function computes a statistic** (every value arrives pre-computed from a tier
module that has a test behind it), and **no plot function imports pyplot**.

**Rationale:**
1. A chart that calculates its own group means is an unpinned number with a picture around it. This
   is also why seaborn is not used: `barplot`, `regplot` and `heatmap` estimate means and confidence
   intervals internally
2. Bypassing pyplot's registry means a returned figure renders exactly once, plot functions are
   order-independent, and `grep -rn "pyplot" src/` is an enforceable check
3. Both rules are tested — `test_plots.py` greps the module source for `.groupby(`, `.mean()`,
   `.corr(` and friends. It caught a real violation on first run

**When NOT to Change This:**
- Do not add seaborn
- Do not compute a value inside a plot function, including a reference line

**When to Revisit:**
- An interactive backend is genuinely needed, which would change the return type not the rules

**Alternatives Considered:**
- ❌ Charts inline in notebooks: four copies, none linted or tested
- ❌ seaborn: a new dependency whose `set_theme()` mutates global rcParams, plus statistics in charts

**Note for notebook authors:** `%matplotlib inline` is required in the first cell. Without it there
is no registered PNG formatter for a bare `Figure`, and the cell silently produces no output and no
error — a failure invisible to anyone reading the notebook source rather than executing it.

**Date:** 2026-09-08 | **Status:** ✅ Approved
