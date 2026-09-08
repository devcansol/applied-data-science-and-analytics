"""Tests for the top-level composer.

``report.py`` computes nothing of its own, so these check that it composes the tiers
faithfully and that the headline it extracts still says what the README says.
"""

from __future__ import annotations

import pytest

from startup_outcomes import report


@pytest.fixture(scope="module")
def results(report_results: dict[str, object]) -> dict[str, object]:
    """The entire project in one call, shared with every other tier test file."""
    return report_results


def test_every_tier_is_present(results: dict[str, object]) -> None:
    assert set(results) == {
        "audit",
        "descriptive",
        "diagnostic",
        "predictive",
        "unsupervised",
        "prescriptive",
        "model_card",
    }


def test_each_tier_declares_the_rows_it_used(results: dict[str, object]) -> None:
    """The two-frame guard, applied across the whole report."""
    assert results["descriptive"]["rows"] == 25_000
    for tier in ("diagnostic", "predictive", "unsupervised", "prescriptive"):
        assert results[tier]["rows"] == 24_467, tier


def test_the_headline_is_a_set_of_comparisons_not_bare_figures(
    results: dict[str, object],
) -> None:
    facts = report.headline(results)
    assert facts["model_pr_auc"] == 0.193
    assert facts["runway_only_pr_auc"] == 0.191
    assert facts["model_beats_runway_only"] is False
    assert facts["leakage_demo_pr_auc"] == 0.235
    assert facts["rows_flagged_at_default_threshold"] == 2


def test_the_headline_carries_the_diagnostic_corrections(
    results: dict[str, object],
) -> None:
    facts = report.headline(results)
    assert facts["ai_adoption_gap_pp"] == 2.95
    assert facts["ai_adoption_interval_excludes_zero"] is True
    assert facts["domain_signal_p_value"] > 0.5
    assert facts["blanks_depend_on_domain"] is True


def test_the_headline_carries_the_prescriptive_verdict(
    results: dict[str, object],
) -> None:
    facts = report.headline(results)
    assert facts["cost_saving_vs_blanket_policy_pct"] == 1.69
    assert facts["targeting_indistinguishable_from_random"] is True


def test_the_summary_states_the_negative_result_plainly(
    results: dict[str, object],
) -> None:
    """A reader who sees only this block must not come away thinking the model works."""
    text = report.summary_text(results)
    assert "does NOT beat" in text
    assert "The data is synthetic" in text
    assert "0.193" in text and "0.191" in text


def test_report_only_composes(results: dict[str, object]) -> None:
    """No number may enter the README through this module without a tier behind it."""
    import inspect

    source = inspect.getsource(report)
    for forbidden in ("groupby", "average_precision", "fit(", "np."):
        assert forbidden not in source, forbidden
