---
description: After plan-review completes, apply its findings to the current plan and call ExitPlanMode to present the revised plan for approval.
---

# Apply Plan Review

## Purpose
After `/plan-review` has run, read its findings from the conversation, apply them to the current plan file, and call `ExitPlanMode` to present the revised plan to the user for approval.

## Steps

### 1. Parse Plan-Review Findings from Conversation Context

Extract from the most recent plan-review output:
- **Overall Status** — `APPROVED` / `APPROVED WITH CONDITIONS` / `NEEDS REVISION`
- **Blocking issues** — anything that caused NEEDS REVISION
- **Conditions for approval** — items listed under "Conditions for Approval"
- **Architecture deviations** — deviations requiring justification or plan changes
- **Project mandatory checks** — the zero-tolerance mandates the review flagged
- **Leakage and reproducibility issues** — non-negotiable, always blocking
- **Missing todos** — implementation, documentation, verification, quality todos the review flagged as absent
- **Reusable code** — existing helpers in `src/startup_outcomes/` the plan should reference
- **High-priority recommendations** — items the review marked as High Priority

### 2. Locate the Current Plan File

Find the active plan file at `~/.claude/plans/` — it was created at the start of this planning session and is the file currently being edited. Read its full contents before making any changes.

### 3. Apply Changes

Edit the plan file based on review status:

**NEEDS REVISION** → Fix every blocking finding before calling ExitPlanMode:
- Correct architecture deviations (feature set bypassing `model_features()`, unseeded randomness, a write to `data/raw/`, a documented number with no audit function behind it, etc.)
- Add leakage controls flagged as missing
- Add mandated patterns to the affected steps
- Insert missing todos into the plan's verification / checklist section

**APPROVED WITH CONDITIONS** → Incorporate conditions as explicit checklist items:
- Add a "Review Conditions" section listing each condition that must be satisfied during implementation
- Add missing todos
- Note reusable code identified by the review

**APPROVED** → Minimal edits:
- Add any missing todos
- Note identified reusable code as a reminder in the relevant steps

For all statuses, append a concise **Review Findings Applied** section at the end of the plan:

```markdown
## Review Findings Applied
- **Overall status**: [APPROVED / APPROVED WITH CONDITIONS / NEEDS REVISION]
- **Changes made**: [bullet list of what was updated in the plan]
- **Deferred to /code-review**: [findings that are implementation-time checks, not plan-time]
```

### 4. Preserve User Intent

Do NOT change the fundamental approach, chosen technique, or overall analytical design unless a blocking finding requires it. The goal is to patch the plan, not redesign it. In particular, do not substitute a different model family, metric, or research question because the review suggested one — surface it as a recommendation instead.

### 5. Call `ExitPlanMode`

After the plan file is updated, call `ExitPlanMode` to present the revised plan to the user for approval.

## Rules

- Fix ALL blocking findings before calling ExitPlanMode — never exit with unresolved NEEDS REVISION items
- If a finding cannot be addressed in the plan (e.g., requires seeing a fitted model), add an explicit note in the plan so `/code-review` can catch it
- Do not duplicate content already in the plan; update in-place where possible
- Do not change the plan's structure or intent beyond what the review requires

## Integration with Development Workflow

This command closes the loop between plan-review and implementation:

```
/plan-review        → produces findings (APPROVED / CONDITIONS / NEEDS REVISION)
/apply-plan-review  → applies findings to plan + calls ExitPlanMode
[user approves]     → implementation begins
/code-review        → catches implementation-time issues
```
