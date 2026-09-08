---
description: transforms, seeds, vectorization, pipeline stages, reproducibility, determinism
globs: ["src/startup_outcomes/**/*.py", "tests/**/*.py"]
alwaysApply: false
---
> **Architecture family:** Analysis Pipeline. See also: `.claude/rules/arch-data-storage.md`, `.claude/rules/arch-modeling.md`, `.claude/rules/arch-notebooks.md`

# Analysis Pipeline

## Determinism

#### ✅ Design Decision: One seed, defined in config

**Context:**
Splits, shuffles, bootstrap intervals, and model fitting all draw randomness. Seeds scattered
across call sites make a result reproducible in principle and irreproducible in practice.

**Pattern:**

```python
from startup_outcomes.config import RANDOM_SEED

# ✅ GOOD
StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ❌ BAD — an inline seed, or none at all
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_test_split(X, y)
```

Verify with:
```bash
grep -rn "random_state\|np.random\|random_seed" src/ --include='*.py' | grep -v RANDOM_SEED
```

**Rationale:**
1. One place to change when a result needs re-checking under a different draw
2. Makes an unseeded call site visible to a grep, so the mandate is enforceable
3. Bootstrap confidence intervals are only comparable across runs if seeded

**When NOT to Change This:**
- Do not add a second module-level seed constant
- Do not seed the global `np.random` state as a substitute for passing `random_state`

**When to Revisit:**
- A result needs deliberate multi-seed replication, which should take a seed *parameter* rather
  than introducing a second constant

**Alternatives Considered:**
- ❌ Per-module seeds: the failure mode this rule exists to prevent
- ❌ No seeding: metrics move between runs and small effects become unreadable

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Transforms

#### ✅ Design Decision: Transforms are pure functions returning new frames

**Context:**
In-place mutation makes notebook cells order-dependent: re-running a cell after an edit gives a
different frame than a fresh run, and the bug surfaces as an unreproducible number.

**Pattern:**

```python
# ✅ GOOD — takes a frame, returns a new one, mutates nothing
def drop_leaky_stage(df: pd.DataFrame) -> pd.DataFrame:
    kept = df.loc[df["Funding_Stage"] != LEAKY_STAGE].copy()
    kept["Funding_Stage"] = kept["Funding_Stage"].cat.remove_unused_categories()
    return kept

# ❌ BAD — caller's frame is modified; cell order now matters
def drop_leaky_stage(df):
    df.drop(df[df["Funding_Stage"] == "IPO"].index, inplace=True)
```

Each stage must be runnable on its own from a freshly loaded frame — no stage may depend on
another having already mutated shared state.

**Rationale:**
1. Cell-order independence in notebooks, which is where this code is mostly driven
2. Makes each stage independently testable
3. `.copy()` after a boolean mask also avoids `SettingWithCopyWarning`

**When NOT to Change This:**
- No `inplace=True` in `src/`
- Do not accept a frame and return `None`

**When to Revisit:**
- Data grows enough that copying dominates runtime — at which point the fix is chunking or a lazy
  engine, not in-place mutation

**Alternatives Considered:**
- ❌ In-place for memory: at 25,000 rows the copies are free
- ❌ Method chaining on a single frame: harder to test one stage in isolation

**Date:** 2026-09-08 | **Status:** ✅ Approved

#### ✅ Design Decision: Vectorized operations, never row iteration

**Context:**
`iterrows` is the reflexive way to express a row-wise rule and is orders of magnitude slower, while
also losing dtypes — each row arrives as an object-dtype Series.

**Pattern:**

```python
# ✅ GOOD
grew = (closed["Current_Headcount_2026"] > closed["Peak_Headcount_2023"]).sum()

# ❌ BAD
grew = sum(1 for _, row in closed.iterrows()
           if row["Current_Headcount_2026"] > row["Peak_Headcount_2023"])
```

Verify with:
```bash
grep -rn "iterrows\|itertuples\|\.apply(lambda row" src/ --include='*.py'
```

Note `.apply` on a *Series* is acceptable where it is genuinely clearer; the target of this rule is
row-wise iteration over a DataFrame.

**Rationale:**
1. Speed, which matters once bootstrap intervals multiply the work
2. Preserves dtypes; `iterrows` silently upcasts to object
3. Vectorized expressions read closer to the claim being made

**When NOT to Change This:**
- Do not reintroduce `iterrows` for readability — extract a named helper instead

**When to Revisit:**
- A genuinely sequential computation appears (a recurrence with no vectorized form)

**Alternatives Considered:**
- ❌ `df.apply(axis=1)`: still row-wise, with the same cost
- ❌ Python loops over `zip` of columns: faster than `iterrows` but still loses vectorization

**Date:** 2026-09-08 | **Status:** ✅ Approved
