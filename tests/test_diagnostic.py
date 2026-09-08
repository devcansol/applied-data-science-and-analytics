"""Pins the diagnostic findings quoted in the README.

Every interval and p-value here is deterministic under ``config.RANDOM_SEED``, so they are
pinned exactly like any other documented number. Each pin is paired with a structural
assertion stating the *claim*, so a numpy change that moves a digit is distinguishable
from a change that moves a conclusion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from startup_outcomes import audit, diagnostic


@pytest.fixture(scope="module")
def results(report_results: dict[str, object]) -> dict[str, object]:
    """This tier's slice of the shared report; see tests/conftest.py."""
    return report_results["diagnostic"]


def test_the_tier_reports_which_frame_each_claim_used(results: dict[str, object]) -> None:
    """Guards the two-frame trap: outcome claims and missingness claims differ."""
    assert results["rows"] == 24_467
    assert results["missingness_rows"] == 25_000


def test_the_ai_adoption_gap_interval_is_pinned(results: dict[str, object]) -> None:
    gap = results["ai_adoption_gap"]
    assert gap["point_pp"] == 2.95
    assert gap["ci_low_pp"] == 1.11
    assert gap["ci_high_pp"] == 5.01


def test_the_ai_adoption_effect_is_real_but_immaterial(results: dict[str, object]) -> None:
    """Corrects the README's "does not meaningfully affect outcomes".

    The interval excludes zero and the permutation test agrees (p = 0.004), so this is
    not a null result. It is a detectable effect whose whole range is too small to act
    on — which is a different claim, and the honest one.
    """
    gap = results["ai_adoption_gap"]
    spread = results["adoption_spread_permutation"]

    assert gap["excludes_zero"] is True
    assert spread["exceeds_null_p95"] is True
    assert spread["p_value"] < 0.05
    # And yet the entire interval sits below 5 pp on a 14% base rate.
    assert gap["ci_high_pp"] < 5.5


def test_the_adoption_spread_permutation_is_pinned(results: dict[str, object]) -> None:
    spread = results["adoption_spread_permutation"]
    assert spread["levels"] == 5
    assert spread["observed_pp"] == 2.95
    assert spread["null_median_pp"] == 1.27
    assert spread["null_p95_pp"] == 2.23
    assert spread["p_value"] == 0.004


def test_sector_and_geography_carry_no_signal_at_all(results: dict[str, object]) -> None:
    """Sharpens README finding 3 from "almost no signal" to "none".

    The observed spreads are not merely small — both sit *below* the median spread that
    shuffling the outcome produces at the same group sizes. There is nothing here to
    explain, and any feature importance a model assigns to these columns is noise.
    """
    for key in ("domain_spread_permutation", "country_spread_permutation"):
        spread = results[key]
        assert spread["levels"] == 20
        assert spread["exceeds_null_p95"] is False
        assert spread["p_value"] > 0.5
        assert spread["observed_pp"] < spread["null_median_pp"], key


def test_the_domain_and_country_spreads_are_pinned(results: dict[str, object]) -> None:
    """On the 24,467-row canonical frame, so slightly below audit.py's 4.82 / 5.83."""
    domain = results["domain_spread_permutation"]
    country = results["country_spread_permutation"]
    assert domain["observed_pp"] == 4.74
    assert domain["null_median_pp"] == 4.85
    assert domain["p_value"] == 0.5405
    assert country["observed_pp"] == 6.09
    assert country["p_value"] == 0.6603


def test_the_runway_cliff_is_the_one_large_effect(results: dict[str, object]) -> None:
    effect = results["runway_cliff_effect"]
    assert effect["point_pp"] == 12.33
    assert effect["ci_low_pp"] == 11.14
    assert effect["ci_high_pp"] == 13.51
    assert effect["excludes_zero"] is True
    # An order of magnitude past everything else in this tier, and the interval is narrow.
    assert effect["ci_low_pp"] > results["ai_adoption_gap"]["ci_high_pp"]


def test_the_blanks_depend_on_domain_beyond_chance(results: dict[str, object]) -> None:
    """The finding that rules out mode-filling ``AI_Adoption_Level``.

    Corrects the README's "consistent with missing-completely-at-random": the blank rate
    varies with ``Domain`` far more than shuffling produces, so the blanks are missing at
    random *conditional on domain*, which is a different assumption with different
    consequences for imputation.
    """
    by_domain = results["missingness_by_domain_permutation"]
    assert by_domain["observed_pp"] == 8.38
    assert by_domain["null_median_pp"] == 4.08
    assert by_domain["null_p95_pp"] == 6.43
    assert by_domain["p_value"] == 0.007
    assert by_domain["exceeds_null_p95"] is True


def test_the_blanks_do_not_depend_on_funding_stage(results: dict[str, object]) -> None:
    """Which is why the README's claim was half right — it named the wrong column."""
    by_stage = results["missingness_by_stage_permutation"]
    assert by_stage["observed_pp"] == 2.87
    assert by_stage["p_value"] == 0.3397
    assert by_stage["exceeds_null_p95"] is False


def test_the_ai_adoption_ordering_reverses_inside_tier_two(results: dict[str, object]) -> None:
    """Simpson's paradox, and the reason the workbooks insist on this check.

    Across all rows, AI-native companies close less often than non-adopters. Inside
    ``Tier 2 VC`` — 5,157 rows, not a thin slice — the sign flips: AI-native close *more*
    often. The headline is not safe to quote without this caveat.
    """
    reversal = results["adoption_reversal"]
    assert reversal["tiers_checked"] == 6
    assert reversal["tiers_against_expected_sign"] == 1
    assert reversal["strongest_reversal_tier"] == "Tier 2 VC"
    assert reversal["strongest_reversal_gap_pp"] == -0.54
    assert reversal["strongest_reversal_rows"] == 5157
    assert reversal["gap_sign_flips_somewhere"] is True


def test_the_rank_correlation_check_finds_the_same_reversal(
    results: dict[str, object],
) -> None:
    """Two independent measures of the same flip, so neither stands alone."""
    check = results["adoption_reversal_within_tier"]
    assert check["subgroups_checked"] == 6
    assert check["subgroups_reversed"] == 1
    assert check["most_reversed_level"] == "Tier 2 VC"
    assert check["ordering_flips"] is True


def test_a_reversal_is_never_reported_from_a_thin_subgroup(raw: pd.DataFrame) -> None:
    """The guard that stops this check manufacturing paradoxes."""
    data = diagnostic.canonical_frame(raw)
    check = diagnostic.reversal_check(data, "AI_Adoption_Level", "Investor_Tier", min_rows=10**6)
    assert check["subgroups_checked"] == 0
    assert check["ordering_flips"] is False


def test_the_size_measures_are_one_block_on_logs(results: dict[str, object]) -> None:
    summary = results["collinearity_summary"]
    assert summary["pairs"] == 5
    assert summary["min_pearson_log"] == 0.925
    assert summary["max_pearson_log"] == 0.988
    assert summary["all_log_pairs_above_0_9"] is True


def test_logging_is_what_makes_the_size_measures_one_block(raw: pd.DataFrame) -> None:
    """The workbooks' curved-relationship diagnostic, and what it decides here.

    On raw values, three of the five pairs sit below 0.9 — readable as "strong but
    separable", which would license interpreting their coefficients individually. On logs
    every pair clears 0.92. The relationships are monotone but curved, so Pearson on raw
    values understates them, and the collinearity decision rests on the logged figures.
    """
    table = diagnostic.correlation_comparison(diagnostic.canonical_frame(raw))
    assert bool((table["pearson_log"] >= table["pearson_raw"]).all())
    assert int((table["pearson_raw"] < 0.9).sum()) == 3
    assert int((table["pearson_log"] < 0.9).sum()) == 0


def test_group_gaps_are_negligible_once_expressed_in_sd_units(
    results: dict[str, object],
) -> None:
    """A 56.6-unit funding gap between domains sounds large and is 0.14 SD."""
    funding = results["funding_gap_by_domain"]
    assert funding["gap_absolute"] == 56.6
    assert funding["gap_in_sd_units"] == 0.141
    assert funding["effect_size_label"] == "negligible"

    runway = results["runway_gap_by_investor_tier"]
    assert runway["gap_in_sd_units"] == 0.368
    assert runway["effect_size_label"] == "small"


def test_the_drilldown_hides_cells_too_small_to_read(raw: pd.DataFrame) -> None:
    data = diagnostic.canonical_frame(raw)
    table = diagnostic.drilldown(data, "Domain", "Investor_Tier")
    assert int(table["n"].min()) >= diagnostic.MIN_DRILLDOWN_ROWS
    assert list(table.columns) == ["n", "closed_pct"]


def test_diagnostic_and_audit_agree_on_the_adoption_gap(raw: pd.DataFrame) -> None:
    """Cross-module consistency on the same quantity, computed two ways."""
    data = diagnostic.canonical_frame(raw)
    from_audit = audit.ai_adoption_effect(data)
    from_diagnostic = diagnostic.ai_adoption_gap(data)
    gap = round(from_audit["None"] - from_audit["AI-Native"], 2)
    assert gap == from_diagnostic["point_pp"]
