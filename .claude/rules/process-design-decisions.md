---
description: Design decision documentation - identify and document architectural decisions in the appropriate rule file
alwaysApply: false
---

# Design Decision Documentation

## Core Rule
**Identify and document design decisions** made during ANY task (bugfix, feature, enhancement).
Add them to the appropriate architecture rule file in `.claude/rules`.

## What Is a Design Decision?
Any choice affecting: data loading and schema contracts, feature construction and what counts as
leakage, imputation, train/test protocol, evaluation metrics, code organization, or the handling of
a data-quality defect.

**Document when:** choosing between approaches, impacting other stages of the analysis,
establishing a pattern, or a future reader would wonder "why?"

**Skip for:** plot styling, comment wording, simple bug fixes with no analytical consequence.

**Always document, in this project:**
- Any change to the feature-timing groups in `config.py` — these are the leakage contract
- Any imputation or encoding choice for `AI_Adoption_Level` (the only column with missing values)
- Any decision to include or exclude the undated financial columns
- Any newly discovered data-integrity defect, and how it is handled

## Where to Document

Route decisions to the appropriate rule file:

| Domain | Rule File |
|--------|-----------|
| raw data, loading, dtypes, schema contract, checksums, data layering | `.claude/rules/arch-data-storage.md` |
| transforms, seeds, vectorization, pipeline stages, reproducibility | `.claude/rules/arch-data-pipeline.md` |
| features, leakage, baselines, cross-validation, metrics, class imbalance | `.claude/rules/arch-modeling.md` |
| notebooks, exploration, output hygiene, promotion into src | `.claude/rules/arch-notebooks.md` |

## Template

```markdown
#### ✅ Design Decision: [Decision Made]

**Context:** What problem, existing state, trigger
**Pattern:** [Code example]
**Rationale:** 1. Primary reason 2. Secondary 3. Cost/trade-off
**When NOT to Change:** Constraints
**When to Revisit:** Trigger conditions
**Alternatives Considered:** Why rejected
**Date:** YYYY-MM-DD | **Status:** Implemented
```

## Process
1. Before implementing: read the relevant rule file
2. Check if a similar decision exists — follow it or document the deviation
3. Document in the appropriate rule file using the template
4. If the decision changes a documented finding, update the README and its test in the same change
5. Update relevant README files with usage examples
