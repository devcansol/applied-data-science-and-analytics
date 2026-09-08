---
description: Documentation requirements - README files for all significant directories, no standalone markdown docs
alwaysApply: false
---

# Documentation Requirements

## README.md Rules

Every significant directory MUST have a `README.md` with:
1. **Purpose** — what the directory is for
2. **Structure** — organization and subdirectories
3. **Conventions** — standards specific to this directory
4. **Files Summary** — list and description of files
5. **Tips & Key Notes** — pitfalls, performance, security considerations

**Before changes:** read README files for all affected directories.
**After changes:** update them — add new files to the summary, revise structure/purpose if changed.

## Findings Are Documentation Too

This project's README quotes specific numbers about the dataset. Those are load-bearing claims:

- A number in the README MUST come from a function in `src/startup_outcomes/audit.py`
- It MUST be pinned by an assertion in `tests/test_audit.py`
- Never write a figure into the README from a notebook cell or a one-off script — if it is worth
  documenting, it is worth an audit function and a test
- When a finding changes, update the README and the test together in the same change

---

## No Standalone Documentation Files

**NEVER create** standalone markdown files like `*_GUIDE.md`, `*_SUMMARY.md`, `*_NOTES.md`,
`FIXES_*.md`, `TROUBLESHOOTING.md`, `TESTING_GUIDE.md`, etc.

**Only acceptable .md files:** `README.md` (per directory), `CLAUDE.md`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `LICENSE.md`, and the harness files in `.claude/commands` / `.claude/rules`.

Analysis write-ups are the common temptation here. A findings narrative goes in the root
`README.md`; the reproducible computation behind it goes in `audit.py`. Neither goes in a new
markdown file.

### Where Documentation Goes

| Type | Location |
|------|----------|
| Coding standards | `CLAUDE.md` |
| Architecture decisions | `.claude/rules` (appropriate domain rule) |
| Feature / usage docs | README.md of the directory where the code lives |
| Dataset findings and caveats | Root `README.md`, backed by `audit.py` + a test |
| Data provenance, license, checksum | Root `README.md` |
| Testing instructions | Relevant directory README "Testing" section |
| Version history | `CHANGELOG.md` |

### Decision Tree
```
Need to document something?
├─ Architectural decision? → .claude/rules (appropriate domain rule)
├─ Coding standard? → CLAUDE.md
├─ A number about the dataset? → audit.py + test, then quote it in README.md
├─ Usage/feature docs? → README.md next to the code
├─ Testing? → relevant README "Testing" section
└─ Default → README.md in the directory where the code lives
```
