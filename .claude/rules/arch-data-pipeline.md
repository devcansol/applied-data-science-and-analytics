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

## Uncertainty

#### ✅ Design Decision: Bootstrap for a named comparison, permutation for a spread

**Context:**
`arch-modeling.md` requires an interval on every reported effect; the reference workbooks report
none, substituting effect size in SD units plus an `n`. That cannot settle the question this dataset
keeps posing — a 4.74 pp closed-rate spread across 20 domains is *either* a weak real effect *or*
exactly what shuffling produces, and those call for opposite conclusions.

The choice of instrument is not stylistic. Max-minus-min over twenty noisy estimates is positive
even when every true rate is identical, so an interval around a spread answers the wrong question.

**Pattern:**

```python
# ✅ A pre-specified pair -> bootstrap interval. Both sides named in advance, so unbiased.
intervals.rate_gap_interval(groups, flags, high="None", low="AI-Native", name="...")

# ✅ A spread across all levels -> permutation null. Shuffle at the observed group sizes.
intervals.permutation_spread(df["Domain"], closed_flags, name="...")

# ✅ Two models on one draw -> paired, which is what makes the difference interval narrow.
intervals.paired_metric_difference(y, model_scores, baseline_scores, metric=..., name="...")

# ❌ An interval around max-minus-min: always positive, answers the wrong question.
```

Conventions: `BOOTSTRAP_RESAMPLES = 1_000` in `config.py`; percentile method at 2.5/97.5; the
generator is built **inside** the function on one physical line naming `RANDOM_SEED`, never at
module level; statistics are vectorised with `np.bincount` rather than a groupby inside the loop;
the index is drawn per iteration, never pre-allocated as a matrix.

Returned dicts carry unit-suffixed keys — `point_pp`, `ci_low_pp`, `ci_high_pp`, `ci_width_pp`,
`resamples`, `excludes_zero` — and every interval gets **two** assertions in tests: the pinned
endpoints, and separately the claim (`excludes_zero is True`).

**Rationale:**
1. A module-level generator would make every interval depend on how many ran before it — the same
   call-order defect the pure-transform rule exists to prevent — and would hide from the seed grep
2. Sharing one draw across a comparison is a feature, not an oversight: it is what makes the
   difference interval [−0.005, +0.009] rather than the sum of two marginal widths
3. `bincount` over a groupby is roughly twenty times faster, and this runs a thousand times
4. A 1,000 x 24,467 index matrix is 196 MB for no benefit
5. B = 1,000 makes the second decimal of a percentage point stable; at B = 200 the third decimal
   moves while the second does not

**When NOT to Change This:**
- Never construct a generator at module level
- Never put an interval on a max-minus-min spread
- Do not draw independently within a comparison

**When to Revisit:**
- A statistic skewed enough to need BCa; at n = 24,467 with these near-symmetric statistics the
  percentile method is adequate and BCa would need its own tests

**Alternatives Considered:**
- ❌ Analytic standard errors: unavailable for average precision
- ❌ Hypothesis tests throughout: the workbooks deliberately avoid p-values, and an interval says
  more; permutation p-values appear only where the question genuinely is "more than chance?"

**Date:** 2026-09-08 | **Status:** ✅ Approved
