---
description: notebooks, exploration, output hygiene, promotion into src, cell-order independence
globs: ["notebooks/**/*.ipynb"]
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
