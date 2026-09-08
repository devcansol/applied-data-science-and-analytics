# `tests/` — the suite that keeps the README honest

## Purpose

Two different jobs live here, and telling them apart is what makes a failure actionable:

- **A contract test failing means the DATA changed.** Check the sha256 before editing anything.
- **A tier-pin failing means a documented NUMBER changed.** The README is now wrong; update both
  together in one commit.
- **A unit test failing means the CODE is wrong.**

## Structure

One test file per source module, named after it.

| Test file | Covers |
|---|---|
| `conftest.py` | Session fixtures — the only place the tiers actually run |
| `test_data_contract.py` | Shape, dtypes, cardinality, column ranges, nesting, leakage guards |
| `test_audit.py` | The data-integrity findings |
| `test_intervals.py` | The interval machinery, on synthetic inputs with known answers |
| `test_features.py` | The design-matrix constructor and the leakage contract |
| `test_descriptive.py` | Tier 1 findings |
| `test_diagnostic.py` | Tier 2 findings: intervals, permutation nulls, the reversal |
| `test_protocol.py` | Splitter, metrics, encoding |
| `test_baselines.py` | The runway threshold rule as an estimator |
| `test_supervised.py` | Tier 3a findings — the project's headline |
| `test_unsupervised.py` | Tier 3b findings |
| `test_prescriptive.py` | Tier 4 findings |
| `test_model_card.py` | The card's facts and its rendering |
| `test_plots.py` | Figures return, and compute no statistics |
| `test_report.py` | The composer, and the headline it extracts |
| `test_repo_invariants.py` | Every `_core-checklist.md` grep, as a test |

## Conventions

- **Full-sentence test names stating the finding** — `test_no_model_beats_the_runway_column`, not
  `test_leaderboard`. The name is the claim; the assertions are the evidence.
- **Exact equality on rounded values. Never `pytest.approx`.** A documented number is either the
  number or it is not.
- **Every model metric gets a pin *and* a band test**, in separate functions. Exact fails + band
  passes → a library moved a digit, update the pin. Both fail → the finding changed, investigate.
  Never widen a band to make a test pass.
- **Every interval gets a pin *and* a structural assertion** — the digits, and separately the claim
  (`excludes_zero is True`).
- **Structural assertions are the real tests.** The pins exist only because the README quotes
  digits. Where a claim is a relation, assert the relation.
- **Only sub-second functions may be called directly from a test.** Anything that fits a model is
  reached through a session fixture.

## Files Summary — the fixture layout

`conftest.py` holds everything expensive, at session scope:

| Fixture | Cost | What it is |
|---|---|---|
| `raw` | 20 ms | All 25,000 rows |
| `analysis` | — | The 24,467-row canonical frame |
| `design` | — | `(X, y)` for the 13-feature problem |
| `oof_scores` | ~1 s | Out-of-fold probabilities from the chosen model |
| `report_results` | ~45 s | **Every tier, run once for the whole suite** |
| `supervised_results` | — | A slice of `report_results` |

`report_results` is deliberately the only place the tiers run. Each tier's test file slices it
rather than recomputing: running `supervised.run_all` in its own file as well doubled the suite's
runtime for identical numbers. The blast radius is intended — a broken `run_all` invalidates every
tier's documentation at once, so failing broadly is the correct signal.

Caching lives here rather than in `src/` on purpose. A memo inside a tier module would be
module-level mutable state, which `arch-data-pipeline.md` rules out because it makes results depend
on call order. A fixture is scoped, visible, and discarded when the session ends.

## Tips & Key Notes

- **Budget: ~50 s, ceiling 60 s.** `--durations=5` prints the slowest tests on every run, so a
  regression is visible the moment it lands. If the ceiling is crossed, cut in this order: the grid
  to four combinations, the depth-12 rung, bootstrap resamples to 500. **Never cut an assertion.**
- **The sub-second loop** while editing is
  `uv run pytest tests/test_data_contract.py tests/test_audit.py`. `-k` works on the full-sentence
  names.
- **No markers, deliberately.** A two-tier suite means the slow tier stops being run, and
  `uv run pytest` is the single gate `CLAUDE.md` names.
- **No `conftest.py` magic beyond fixtures.** The two original test files keep their local `df`
  fixture, which now delegates to `raw` — a one-line change that avoided churning 26 signatures.
- **When a model pin fails after `uv lock --upgrade`:** read `UPGRADE_HINT` in
  `test_supervised.py`. It names the four artifacts to update in one commit and tells you how to
  regenerate them. Do not transcribe values from a plan or a chat — run the module.
- **Tests that check for forbidden patterns must not match themselves.** `test_repo_invariants.py`
  matches import statements rather than words, because two earlier versions tripped on their own
  docstrings.
