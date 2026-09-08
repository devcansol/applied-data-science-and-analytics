---
description: Core post-task checklist, critical architectural patterns, and rule index for startup-outcomes
alwaysApply: true
---

# Core Checklist & Rule Index

## Post-Task Checklist (MANDATORY)

After completing ANY task with code changes:

1. **Lint Gate**: Run `uv run ruff check .` — must pass with 0 errors, 0 warnings
2. **Tests**: Run `uv run pytest` — all tests must pass
3. **README Updates**: Update README.md files in all affected directories
4. **Security Audit**: Clean up debug logs, remove sensitive data from log output
5. **Design Decisions**: Document architectural decisions in the appropriate rule file (see routing table in the design-decisions process rule)
6. **Documentation**: Never create standalone .md files — use README.md files only

## Critical Patterns — NEVER Break

- **No outcome-contemporaneous column reaches a model.** Build feature lists from
  `config.model_features()`; never assemble one by hand or by dropping columns
  (verify: `grep -rn "Current_Headcount_2026\|Layoffs_2024_2025" src/ --include='*.py' | grep -v "config.py\|load.py\|audit.py"` — expect no output;
  those three modules legitimately enumerate the whole schema, everything else must not name these columns)
- **`Funding_Stage == 'IPO'` is neutralised before `Funding_Stage` is used** — it determines the
  target in every row where it appears; use `drop_leaky_stage()`
  (verify: `uv run pytest tests/test_data_contract.py -k leaky`)
- **`data/raw/` is immutable.** Nothing in the repo writes to it, and it is never committed
  (verify: `git check-ignore data/raw/global_tech_startups_2026.csv`)
- **Every number quoted in the README comes from `audit.py` and is pinned by a test** — if a
  documented finding changes, a test fails rather than the README going stale
  (verify: `uv run pytest tests/test_audit.py`)
- **Loading declares dtypes and `keep_default_na=False`** — pandas otherwise reads the valid
  `AI_Adoption_Level` value `"None"` as missing
  (verify: `uv run pytest tests/test_data_contract.py -k none_is_a_category`)
- **One seed.** All randomness derives from `config.RANDOM_SEED`
  (verify: `grep -rn "random_state\|np.random\|random_seed" src/ --include='*.py' | grep -v RANDOM_SEED` — expect no output)
- **Model results are reported against a baseline**, never as bare accuracy — the majority class
  already scores 86.19% on binary closed-vs-rest
- **Notebooks import from `src/`** and define no reusable logic
  (verify: `grep -n '"\(def\|class\) ' notebooks/*.ipynb` — expect no output; note `^def` cannot
  match inside notebook JSON, so the quote-anchored form is the one that works)
- Never create adhoc markdown documentation files or adhoc test/debug scripts — docs go in
  `README.md`, tests go in `tests/`

Mandates accrete here via the design-decision process — when a task establishes a new
non-negotiable, add it with a runnable verification command.

## Architecture Rule Index

Detailed patterns and design decisions live in domain-specific rule files. Do not enumerate counts
here — the directory listing of `.claude/rules` is the authoritative index.

| Domain | Rule File |
|--------|-----------|
| raw data, loading, dtypes, schema contract, checksums, data layering | `.claude/rules/arch-data-storage.md` |
| transforms, seeds, vectorization, pipeline stages, reproducibility | `.claude/rules/arch-data-pipeline.md` |
| features, leakage, baselines, cross-validation, metrics, class imbalance | `.claude/rules/arch-modeling.md` |
| notebooks, exploration, output hygiene, promotion into src | `.claude/rules/arch-notebooks.md` |

## Process Rule Index

| Rule File | Purpose |
|-----------|---------|
| `.claude/rules/process-lint-check.md` | `uv run ruff check .` requirement (always applied) |
| `.claude/rules/process-documentation.md` | README requirements, no standalone docs |
| `.claude/rules/process-design-decisions.md` | How and where to document design decisions |
