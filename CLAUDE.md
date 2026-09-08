# startup-outcomes — Claude Guide

Leakage-controlled analysis of a synthetic global tech/AI startup outcomes dataset (25,000
companies × 17 columns). Python + pandas + scikit-learn, managed with `uv`.

## Quick Start

```bash
uv sync
uv run jupyter lab
```

The dataset is not in the repo. See the README for how to obtain `data/raw/global_tech_startups_2026.csv`
and verify its checksum.

## Slash Commands

Project workflow commands live in `.claude/commands/` — list that directory for the current set.
The core loop:

| Command | When to use |
|---------|-------------|
| `/plan-review` | Before writing any code — validates architecture compliance, reusability, security |
| `/apply-plan-review` | After plan-review — applies its findings to the plan, then presents for approval |
| `/code-review` | Before merging — runs the lint gate, then compliance/security/performance review |

## Critical Patterns — NEVER Break

See `.claude/rules/_core-checklist.md` for the authoritative list with verification commands.
Highlights:

- Never let an outcome-contemporaneous column reach a model — build feature lists from
  `config.model_features()`, never by hand
- `data/raw/` is immutable and never committed — nothing writes to it
- Every number quoted in the README must come from `audit.py` and be pinned by a test
- Notebooks are exploration only — reusable logic gets promoted into `src/startup_outcomes/`
- Report model performance against a baseline, never as bare accuracy
- No adhoc `.md` files or scratch scripts — docs go in `README.md`, decisions in `.claude/rules`

## Architecture Rules

Detailed patterns live in `.claude/rules/` — one file per domain. Read the relevant file before
working in a domain:

| Domain | Rule File |
|--------|-----------|
| raw data, loading, dtypes, schema contract, checksums, data layering | `.claude/rules/arch-data-storage.md` |
| transforms, seeds, vectorization, pipeline stages, reproducibility | `.claude/rules/arch-data-pipeline.md` |
| features, leakage, baselines, cross-validation, metrics, class imbalance | `.claude/rules/arch-modeling.md` |
| notebooks, exploration, output hygiene, promotion into src | `.claude/rules/arch-notebooks.md` |

## Post-Task Checklist (MANDATORY)

After any task with code changes:

1. `uv run ruff check .` — must pass with **0 errors, 0 warnings**
2. **Tests**: Run `uv run pytest` — all tests must pass
3. Update `README.md` in all affected directories
4. Security audit — remove debug logs, clean sensitive data from output
5. Document architectural decisions in the relevant rule file in `.claude/rules`

## Development Scripts

```bash
uv sync                             # create the venv and install from the lockfile
uv run pytest                       # data contract + audit findings
uv run ruff check .                 # the quality gate
uv run ruff format --check .        # formatting gate (uv run ruff format to fix)
uv run jupyter lab                  # exploration
uv run python -c "from startup_outcomes.audit import run_all; print(run_all())"
shasum -a 256 data/raw/global_tech_startups_2026.csv   # verify the dataset
```
