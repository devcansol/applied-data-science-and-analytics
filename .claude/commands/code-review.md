---
description: Review code changes for quality, optimization, and architecture compliance. Use when reviewing pull requests, examining code changes, running pre-merge checks, or when the user asks for a code review.
---

# Code Review & Architecture Compliance Check

## Purpose
Review code changes for quality, optimization, and compliance with the architecture rules in `.claude/rules`.

## Domain → Rule File Routing

When checking compliance, route each topic to its authoritative rule file. Never rely on memory of the architecture — read the rule file.

| Domain | Rule File |
|--------|-----------|
| raw data, loading, dtypes, schema contract, checksums, data layering | `.claude/rules/arch-data-storage.md` |
| transforms, seeds, vectorization, pipeline stages, reproducibility | `.claude/rules/arch-data-pipeline.md` |
| features, leakage, baselines, cross-validation, metrics, class imbalance | `.claude/rules/arch-modeling.md` |
| notebooks, exploration, output hygiene, promotion into src | `.claude/rules/arch-notebooks.md` |

## Steps

### 1. Run the Lint Gate (MANDATORY — Before Review)

**CRITICAL:** Run the project gate before proceeding with code review.

```bash
uv run ruff check .
uv run ruff format --check .

# Expected output (REQUIRED):
# 0 errors, 0 warnings
# Exit code: 0
```

**If check fails:**
```
❌ STOP — Code is NOT ready

Lint gate failed. Fix ALL errors and warnings before proceeding.
- Errors: [X]   - Warnings: [Y]   (Required: 0 / 0)

DO NOT proceed with code review until:
✅ uv run ruff check . passes with 0 errors, 0 warnings
✅ uv run ruff format --check . reports no reformatting needed
✅ Commands exit with status code 0
```

**Only proceed to step 2 if the gate passes with 0 errors and 0 warnings.**

Then run the test suite — a failure here is also blocking, and a failure in `tests/test_audit.py` specifically means a documented finding no longer matches the data:

```bash
uv run pytest
```

---

### 2. Identify Changed Files
Review git diff (staged, unstaged, and branch vs default branch as appropriate) to identify all modified files.

### 3. Architecture Compliance Check
For each changed file, verify compliance with `.claude/rules/_core-checklist.md` (Critical Patterns) and the relevant rule file from the routing table above.

**Data Loading & Storage:**
- ✅ Loading goes through `load_raw()`, not a bare `pd.read_csv`?
- ✅ `keep_default_na=False` still in place, and dtypes still declared?
- ✅ Nothing writes to `data/raw/`?
- ✅ Any new column classified into a feature-timing group, and contract tests updated?

**Leakage Control:**
- ✅ Feature sets come from `config.model_features()`?
- ✅ `Layoffs_2024_2025` / `Current_Headcount_2026` absent from every model path?
- ✅ `drop_leaky_stage()` applied wherever `Funding_Stage` is used as a feature?
- ✅ Scaling/encoding/imputation fitted inside the fold, not before the split?

**Reproducibility:**
- ✅ Every `random_state` is `RANDOM_SEED`?
- ✅ No `inplace=True`; transforms return new frames?
- ✅ `uv.lock` updated if dependencies changed?

**Evaluation:**
- ✅ Baseline comparison present alongside every reported model metric?
- ✅ Metric suited to the class imbalance — not bare accuracy?
- ✅ Intervals reported for effects, not point estimates alone?

**Claims & Honesty:**
- ✅ Every new number in the README traceable to an `audit.py` function and pinned by a test?
- ✅ Findings worded as properties of synthetic data, not claims about real startups?
- ✅ Any deliberate leakage demonstration labelled as such?

**Efficiency:**
- ✅ Vectorized operations, no row iteration?
- ✅ No repeated CSV loads or recomputed groupbys?

**Notebooks:**
- ✅ No function or class definitions in committed notebooks?
- ✅ Outputs cleared before commit?

### 4. Project-Specific Mandatory Checks (grep-verifiable)

Run each verification command. Expected result: no output (no violations). Mandates accrete here via the design-decision process — add to this list when a task establishes a new one.

```bash
# No outcome-contemporaneous column reaches a model.
# config/load/audit legitimately enumerate the whole schema; nothing else may name these columns.
grep -rn "Current_Headcount_2026\|Layoffs_2024_2025" src/ --include='*.py' | grep -v "config.py\|load.py\|audit.py"

# All randomness routed through the single seed
grep -rn "random_state\|np.random\|random_seed" src/ --include='*.py' | grep -v RANDOM_SEED

# Nothing writes to the immutable raw directory
grep -rn "to_csv\|to_parquet\|open(.*[\"']w" src/ --include='*.py' | grep "data/raw"

# No in-place mutation
grep -rn "inplace=True" src/ --include='*.py'

# No row-wise iteration
grep -rn "iterrows\|apply(lambda row" src/ --include='*.py'

# Loader still disables pandas' default NA strings ("None" is a real category)
grep -L "keep_default_na=False" src/startup_outcomes/load.py

# Notebooks define no reusable logic (quote-anchored: ^def never matches inside .ipynb JSON)
grep -n '"\(def\|class\) ' notebooks/*.ipynb

# Notebooks committed without outputs
grep -l '"output_type"' notebooks/*.ipynb
```

**If a violation is found:**
```
⚠️ MANDATE VIOLATION

File: [file path]
Line: [line number]
Issue: [what rule is broken]
Impact: [why this matters]

Required Fix:
Replace: [current code]
With: [required pattern]
```

### 5. Code Reusability Review

- ✅ Existing helpers in `src/startup_outcomes/` reused where they should be?
- ✅ Could new code use `load_raw`, `drop_leaky_stage`, `sha256`, or an existing `audit.py` function instead?
- ✅ Common logic extracted to shared locations?
- ✅ No duplicate logic across files, and no statistic recomputed that `audit.py` already provides?

**Anti-patterns to flag:**
- ❌ Duplicate logic instead of reusing existing code
- ❌ New helper created without checking for existing similar ones
- ❌ A constant redefined locally that belongs in `config.py`
- ❌ Shared logic not extracted for reuse

### 6. Report Deviations

If ANY deviation from architecture patterns is found:

```
⚠️ ARCHITECTURE DEVIATION FOUND

File: [file path]
Pattern Violated: [pattern name from the relevant rule file]
Current Implementation: [what the code does]
Expected Pattern: [what it should do per architecture]
Impact: [validity/reproducibility/maintainability concern]

Recommendation:
- [Specific fix needed]

Should we:
a) Update code to match architecture
b) Update architecture to match new pattern (with justification)
c) Create exception for this specific case (scoped, with review date)
```

### 7. Code Quality Review

**Check for:** proper error handling · type hints on new functions · DRY (no duplication) · clear naming · comments only where logic is genuinely complex.

**Anti-patterns to flag:** stub functions returning hardcoded values · magic numbers that belong in `config.py` · debug `print` calls left in `src/` · deeply nested conditionals · a threshold or column name hardcoded in two places.

### 8. Validity Review

- [ ] No outcome-contemporaneous or target-derived feature in any model
- [ ] No preprocessing fitted outside the CV fold
- [ ] No hyperparameter tuning against the test set
- [ ] Undated financial columns used only under the documented assumption
- [ ] Results not selected as the best of many unreported runs
- [ ] No synthetic-data finding stated as a real-world claim
- [ ] Documented numbers still match the data (`uv run pytest tests/test_audit.py`)

### 9. Efficiency Review

- [ ] Vectorized rather than row-wise
- [ ] Columns subset before expensive operations
- [ ] Categorical dtypes preserved
- [ ] No redundant recomputation across cells or functions
- [ ] Cross-validation cost proportionate

### 10. Documentation Check

- [ ] README files updated for affected directories
- [ ] Design decisions documented (if architectural)
- [ ] New documented numbers backed by an audit function and a test
- [ ] Breaking changes noted

### 11. Update Architecture Documentation

If a legitimate new pattern was introduced, add a design-decision record to the appropriate rule file in `.claude/rules` (route by domain using the design-decisions process rule):

```markdown
#### ✅ Design Decision: [Decision Made]

**Pattern:** [Code example]
**Rationale:** 1. [Why this approach] 2. [Benefits] 3. [Trade-offs]
**When NOT to Change This:** [Constraints]
**When to Revisit:** [Conditions]
**Date:** YYYY-MM-DD | **Status:** ✅ Approved
```

Update architecture docs only with user approval.

### 12. Summary Report

```markdown
# Code Review Summary

**Lint Gate Status:** ✅ 0 errors, 0 warnings / ❌ FAILED
**Test Status:** ✅ All pass / ❌ FAILED
**Files Changed:** [count]
**Architecture Compliance:** ✅ Compliant / ⚠️ Deviations Found
**Project Mandates:** ✅ All pass / ⚠️ Violations found
**Code Quality:** ✅ Good / ⚠️ Issues Found
**Validity:** ✅ Sound / ⚠️ Concerns

## Gate Results (MANDATORY)
**Command:** `uv run ruff check .`
**Status:** [PASS ✅ / FAIL ❌]  **Errors:** [n] (Required: 0)  **Warnings:** [n] (Required: 0)  **Exit Code:** [n]
**Tests:** `uv run pytest` — [n] passed, [n] failed

## ✅ Compliant Patterns
- [Patterns followed correctly]

## ⚠️ Deviations Found
- [Deviations with recommendations]

## 🔒 Validity Review
- [Leakage/reproducibility checks or concerns]

## ⚡ Efficiency Review
- [Considerations]

## 📝 Documentation Status
- [README and rule-file updates needed]

## 🎯 Recommendations
1. [Recommendation]

## ✅ Ready to Merge
Ready: [YES/NO]
```

## Production-Ready Criteria

Code is **NOT** ready if:
- ❌ Lint gate fails (any errors or warnings)
- ❌ Tests fail
- ❌ Architecture deviations without justification
- ❌ Any project mandate violated
- ❌ A leakage or reproducibility concern is present
- ❌ A documented number no longer matches the data
- ❌ A finding is stated as a real-world claim

## Notes
- **MANDATORY:** Lint gate with 0 errors, 0 warnings before reporting completion
- Flag ALL deviations, even minor ones
- Prioritize validity issues over efficiency
- Suggest, don't demand — the user decides the final approach
