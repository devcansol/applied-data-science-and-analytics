---
description: Mandatory lint and format gate - uv run ruff check . must pass with 0 errors and 0 warnings
alwaysApply: true
---

# Lint Gate Requirement

This project has no static type checker, so **lint is the gate**. Both commands below must pass.

## Mandatory After Every Task

```bash
uv run ruff check .
uv run ruff format --check .
```

**Success Criteria:** Exit code 0 from both, with `0 errors, 0 warnings`

`uv run ruff format .` fixes formatting; there is no auto-fix mandate for lint findings — read
them and fix the cause.

## Rules

- Run after EVERY code change before marking a task complete
- Fix ALL errors AND warnings — zero tolerance
- Rerun after making fixes until clean
- Cannot commit code that fails the check
- No exceptions for "quick fixes"
- Never widen `ignore` in `pyproject.toml` to silence a finding — fix the code, or record a design
  decision explaining why the rule does not fit this project
- Notebooks **are** in scope: ruff lints and formats `.ipynb` cells. That is another reason logic
  belongs in `src/` — but a notebook cell still has to pass the gate
- `.claude/` is excluded via `extend-exclude`, because ruff also formats fenced Python blocks
  inside Markdown and would rewrite the illustrative snippets in these rule files

## Companion Gate

`uv run pytest` must also pass, but it is a separate checklist item, not this gate. A green lint
run says nothing about whether the documented findings still hold.
