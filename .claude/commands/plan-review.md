---
description: Review implementation plans before code is written for architecture compliance, reusability, leakage control, and reproducibility. Use when reviewing a plan, proposal, or design before implementation begins, or when the user asks to review a plan.
---

# Plan Review & Architecture Compliance Check

## Purpose
Critically review implementation plans before code is written. Ensure plans comply with the architecture rules in `.claude/rules`, identify reusable code opportunities, optimize for leakage control/reproducibility/quality, and ensure proper documentation todos are in place.

## Domain → Rule File Routing

When checking compliance, route each topic to its authoritative rule file. Never rely on memory of the architecture — read the rule file.

| Domain | Rule File |
|--------|-----------|
| raw data, loading, dtypes, schema contract, checksums, data layering | `.claude/rules/arch-data-storage.md` |
| transforms, seeds, vectorization, pipeline stages, reproducibility | `.claude/rules/arch-data-pipeline.md` |
| features, leakage, baselines, cross-validation, metrics, class imbalance | `.claude/rules/arch-modeling.md` |
| notebooks, exploration, output hygiene, promotion into src | `.claude/rules/arch-notebooks.md` |

## Steps

### 1. Analyze Plan Scope
Review the plan to identify:
- **Files to be created/modified**: List all files mentioned or implied
- **New analysis stages/features**: Identify new functionality being added
- **Dependencies**: New packages, or existing modules in `src/startup_outcomes/` required
- **Data flow**: How data moves from `data/raw/` through transforms to a result
- **Claims produced**: Any number or finding the work will assert, and where it will be documented

### 2. Architecture Compliance Check
For each planned change, verify compliance with `.claude/rules/_core-checklist.md` (Critical Patterns) and the relevant rule file from the routing table above.

**Data Loading & Storage:**
- ✅ Loads via `load_raw()` rather than a bare `pd.read_csv`?
- ✅ Nothing writes to `data/raw/`? Derived data goes to a sibling directory?
- ✅ New columns classified into a feature-timing group in `config.py`?
- ✅ Declared dtypes extended if the schema changes, and contract tests updated with it?

**Leakage Control:**
- ✅ Feature sets built from `config.model_features()`, never hand-listed or drop-listed?
- ✅ `Layoffs_2024_2025` and `Current_Headcount_2026` stay out of every model?
- ✅ `drop_leaky_stage()` applied before `Funding_Stage` is used?
- ✅ Any preprocessing fitted inside the CV fold, not on the full dataset before splitting?
- ✅ Imputation of `AI_Adoption_Level` treated as an in-fold step, not a global one?

**Reproducibility:**
- ✅ All randomness routed through `config.RANDOM_SEED`?
- ✅ Transforms planned as pure functions returning new frames (no `inplace=True`)?
- ✅ Each stage runnable independently from a freshly loaded frame?
- ✅ New dependencies added via `uv add` so the lockfile stays authoritative?

**Evaluation:**
- ✅ Baselines planned before models — majority class, stratified, and the 6-month runway rule?
- ✅ Metric appropriate to a 13.81% positive rate (balanced accuracy / PR-AUC / per-class recall), not bare accuracy?
- ✅ Intervals planned for every reported effect, not point estimates?
- ✅ Stratified k-fold on the target?
- ✅ Collinear size measures handled as a block rather than ranked individually?

**Claims & Honesty:**
- ✅ Every number destined for the README backed by a function in `audit.py` and a test?
- ✅ Findings phrased as properties of a synthetic generator, not of real startups?
- ✅ Null results reported as results rather than dropped?

**Code Quality:**
- ✅ Vectorized operations planned instead of row iteration?
- ✅ Notebooks kept as thin drivers, with logic in `src/`?

### 3. Identify Reusable Code Opportunities

**Search for existing patterns before approving anything new:**

- ✅ Does `src/startup_outcomes/config.py` already declare the paths, seed, or column group needed?
- ✅ Does `src/startup_outcomes/load.py` already cover the loading, dtype, or filtering need (`load_raw`, `drop_leaky_stage`, `sha256`)?
- ✅ Does `src/startup_outcomes/audit.py` already compute this statistic (`closed_rate_spread`, `log_correlations`, `target_base_rates`, ...)?
- ✅ Do `tests/` already assert this property, making a new test a duplicate?

**For each reusable opportunity:**
```
🔍 REUSABLE CODE FOUND

Existing: [file path and function name]
Planned: [what the plan wants to create]
Match: [how they're similar]

Recommendation:
- [ ] Extend existing code with backward compatibility
- [ ] Create shared helper for both use cases
- [ ] Refactor existing code to be more reusable

Impact:
- Code reduction: [X lines]
- Consistency: [benefit]
- Maintenance: [benefit]
```

**Before creating any new module/helper (MANDATORY):**
- ✅ **REQUIRED**: Search `src/startup_outcomes/` for existing similar code
- ✅ **REQUIRED**: Check if existing code can be extended with new options
- ✅ **REQUIRED**: Justify why new code is needed vs reusing existing
- ✅ **REQUIRED**: Design new code for reuse; extract common logic to shared locations

**Anti-patterns to flag:**
- ❌ Creating new code without checking for existing similar code
- ❌ Duplicating logic instead of extracting to a shared location
- ❌ Recomputing a statistic `audit.py` already provides
- ❌ One-off implementations of something that will recur

### 4. Leakage & Validity Review

**Critical checks:**
- ✅ No outcome-contemporaneous column in any feature set?
- ✅ No target-derived feature (a ratio or flag computed from `Acquisition_Status`)?
- ✅ No preprocessing fitted on data the model will be tested on?
- ✅ Row-level independence respected — no company appearing in both train and test?
- ✅ Undated financial columns used only under the documented as-of-latest assumption?
- ✅ Any deliberate leakage demonstration clearly labelled as such?

**Anti-patterns to flag:**
- ❌ Scaling, encoding, or imputing before the split
- ❌ Feature selection using the full dataset's target
- ❌ Tuning hyperparameters on the test set
- ❌ Reporting the best of many runs without noting the selection
- ❌ Presenting a synthetic-data finding as a real-world insight

### 5. Efficiency Review

- ✅ Vectorized operations rather than `iterrows`/`apply(axis=1)`?
- ✅ Columns subset before expensive operations?
- ✅ Categorical dtypes preserved rather than exploded to object?
- ✅ Repeated expensive computation hoisted out of loops?
- ✅ Cross-validation cost proportionate — no needless refitting?

**Anti-patterns:** row-wise iteration · recomputing the same groupby repeatedly · loading the CSV more than once per session · one-hot encoding a 58-level column without considering the alternative.

### 6. Code Quality Review

- ✅ Error handling planned for the missing-data-file case?
- ✅ Type hints planned for new functions?
- ✅ DRY — no planned duplication?
- ✅ Clear naming conventions?
- ✅ Edge cases considered (zero-revenue rows, the 9.74% missing `AI_Adoption_Level`)?

**Anti-patterns:** stub functions with hardcoded returns · missing error handling · magic numbers that should be named constants in `config.py` · deep nesting · missing type hints.

### 7. Report Architecture Deviations

If ANY deviation from architecture patterns is found:

```
⚠️ ARCHITECTURE DEVIATION FOUND

Plan Element: [what's being planned]
Pattern Violated: [pattern name from the relevant rule file]
Planned Approach: [what the plan proposes]
Expected Pattern: [what architecture requires]
Impact: [validity/reproducibility/maintainability concern]

Recommendation:
- [Specific fix needed to align with architecture]

Design Decision Required — should we:
a) Update plan to match architecture (recommended)
b) Update architecture to match new pattern (requires justification)
   - Rationale: [why this deviation is necessary]
   - Trade-offs: [what we're giving up]
   - Documentation: [update the appropriate rule file per the design-decisions process rule]
c) Create exception for this specific case
   - Scope: [limited to this feature]
   - Review date: [when to revisit]
```

For approved deviations, record a design decision in the appropriate rule file:

```markdown
#### ✅ Design Decision: [Decision Made]

**Context:** What problem, existing state, trigger
**Pattern:** [Code example]
**Rationale:** 1. [Primary reason] 2. [Benefits] 3. [Trade-offs]
**When NOT to Change This:** [Constraints]
**When to Revisit:** [Trigger conditions]
**Alternatives Considered:** ❌ [Alternative]: [Why rejected]
**Date:** YYYY-MM-DD | **Status:** 🚧 Proposed | ✅ Approved
```

### 8. Project-Specific Mandatory Checks

Each mandate below is zero-tolerance. Verify the plan honors it, and note the verification command to run after implementation. Mandates accrete here via the design-decision process — add to this list when a task establishes a new one.

- [ ] No outcome-contemporaneous column reaches a model — verify: `grep -rn "Current_Headcount_2026\|Layoffs_2024_2025" src/ --include='*.py' | grep -v "config.py\|load.py\|audit.py"` (those three declare the schema; nothing else may name these columns)
- [ ] All randomness routed through the single seed — verify: `grep -rn "random_state\|np.random" src/ --include='*.py' | grep -v RANDOM_SEED`
- [ ] `data/raw/` never written to — verify: `grep -rn "to_csv\|to_parquet\|open(.*w" src/ --include='*.py' | grep "data/raw"`
- [ ] No in-place mutation in `src/` — verify: `grep -rn "inplace=True" src/ --include='*.py'`
- [ ] No row iteration — verify: `grep -rn "iterrows\|apply(lambda row" src/ --include='*.py'`
- [ ] Notebooks define no reusable logic — verify: `grep -n '"\(def\|class\) ' notebooks/*.ipynb`
- [ ] Every README number has an audit function and a test — verify: `uv run pytest tests/test_audit.py`

### 9. Ensure Todos Are Complete

**Verify the plan includes todos for:**

**Implementation:** core functionality · error handling · feature-timing classification for any new column · in-fold preprocessing.

**Documentation:** update README files in affected directories · record design decisions in the appropriate rule file in `.claude/rules` · add an audit function plus test for any new documented number.

**Honesty:** state assumptions (undated columns) · report null results · frame findings as properties of synthetic data.

**Quality (MANDATORY — Zero Tolerance):**
- [ ] Run `uv run ruff check .` and fix ALL issues
- [ ] 0 errors, 0 warnings, exit code 0 (required)
- [ ] Run `uv run ruff format --check .`
- [ ] Run `uv run pytest` — all tests pass

**If todos are missing, add them:**
```
Missing Todo Identified:
- [ ] [Todo description]
- Reason: [why this todo is needed]
- Priority: [High/Medium/Low]
```

### 10. Optimization Suggestions

Suggest (don't demand): additional baselines worth comparing against · sensitivity checks (results with and without the undated columns) · effect-size intervals where only point estimates are planned · engineered ratios that may beat raw collinear levels · additional contract assertions.

### 11. Summary Report

```markdown
# Plan Review Summary

## 📋 Plan Overview
- **Scope**: [Brief description]
- **Files Affected**: [Count] to create/modify
- **Dependencies**: [New packages / existing modules]

## ✅ Architecture Compliance
- [List of compliant patterns]

## ⚠️ Architecture Deviations
- [List of deviations with recommendations]

## 🔍 Reusable Code Opportunities
- [List of reuse suggestions; estimated code reduction]

## 🔒 Leakage & Validity Review
- [Issues flagged / enhancements recommended]

## ⚡ Efficiency Review
- [Optimizations suggested]

## 📝 Code Quality
- [Improvements suggested]

## ✅ Todos Verified
- [Missing todos added]

## 🎯 Recommendations
### High Priority
- [Critical — must address before implementation]
### Medium Priority
- [Important]
### Low Priority
- [Nice-to-have]

## ✅ Plan Approval Status
- Architecture Compliance: [PASS/FAIL with conditions]
- Leakage & Validity: [PASS/FAIL with conditions]
- Reproducibility: [PASS/FAIL with conditions]
- Code Quality: [PASS/FAIL with conditions]
- Project Mandates: [PASS/FAIL]
- Todos Complete: [YES/NO]

**Overall Status**: ✅ APPROVED | ⚠️ APPROVED WITH CONDITIONS | ❌ NEEDS REVISION

**Conditions for Approval** (if applicable):
- [List of conditions that must be met]
```

## Production-Ready Criteria

A plan is ready for implementation when:
- ✅ Architecture compliance verified (or deviations documented with justification)
- ✅ Leakage controls addressed
- ✅ Reproducibility addressed (single seed, pure transforms, lockfile)
- ✅ Reusable code identified and planned for reuse
- ✅ All project mandates honored
- ✅ All required todos present (implementation, documentation, honesty, quality)
- ✅ Lint gate requirement understood (0 errors, 0 warnings mandatory)

## Notes

- **Proactive Review**: Catching issues before implementation saves time
- **Architecture First**: Always prefer aligning with existing patterns
- **Document Deviations**: If deviation is necessary, document thoroughly
- **Reusability**: Prioritize extending existing code over creating new
- **Validity Priority**: Leakage and reproducibility issues are non-negotiable
- **Suggest, Don't Demand**: Provide recommendations; the user decides the final approach

## Integration with Development Workflow

```
/plan-review        → produces findings (APPROVED / CONDITIONS / NEEDS REVISION)
/apply-plan-review  → applies findings to plan + calls ExitPlanMode
[user approves]     → implementation begins
/code-review        → catches implementation-time issues
```
