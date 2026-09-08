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
- `references/` is excluded for the same reason, recorded as a design decision below

## Excluded Paths

#### ✅ Design Decision: `references/` is excluded from the gate

**Context:**
`uv run ruff check .` reported **306 errors** and `ruff format --check .` five files needing
reformatting — every one of them in `references/`, unmodified third-party course material that
`src/` deliberately never imports. The mandatory post-task checklist could not pass, and the
failures had nothing to do with any change being made.

**Pattern:**

```toml
extend-exclude = [".claude", "references"]
```

Paired with a mandate that nothing may import from there
(verify: `grep -rn "^\s*\(from\|import\) mlkit" src/ tests/` — expect no output, and
`tests/test_repo_invariants.py::test_nothing_imports_the_reference_workbooks` asserts it).

**Rationale:**
1. Reformatting the workbooks would destroy the fidelity of the source the four tiers are checked
   against — the same argument the `.claude` exclusion already makes
2. 306 findings in files nobody may edit hide the findings that matter
3. This is a **path** exclusion with a written rationale, not a widened `ignore`. The rule above
   forbids silencing a finding in code this project owns; it explicitly permits recording a design
   decision where a rule does not fit

**When NOT to Change This:**
- Do not add a path to `extend-exclude` without a design decision here
- Do not exclude anything under `src/`, `tests/` or `notebooks/`

**When to Revisit:**
- If anything in `references/` ever becomes importable, it moves into `src/` and into the gate

**Alternatives Considered:**
- ❌ Reformat the workbooks: destroys their value as a reference and produces a large unreviewable diff
- ❌ Widen `ignore` to cover the rule codes involved: would silence the same findings in real code
- ❌ Delete `references/`: they are the methodological source this project mirrors

**Date:** 2026-09-08 | **Status:** ✅ Approved

## Companion Gate

`uv run pytest` must also pass, but it is a separate checklist item, not this gate. A green lint
run says nothing about whether the documented findings still hold.
