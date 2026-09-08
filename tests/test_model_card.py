"""Pins the model card.

The card is the artifact most likely to be read in isolation, so its limits are asserted
as strictly as its numbers: a card that states a score without stating that the score does
not beat its baseline would be misleading on its own.
"""

from __future__ import annotations

import pytest

from startup_outcomes.config import EXCLUDED_LEAKY, RANDOM_SEED
from startup_outcomes.models import card


@pytest.fixture(scope="module")
def facts(report_results: dict[str, object]) -> dict[str, object]:
    """The card's slice of the shared report; see tests/conftest.py."""
    return report_results["model_card"]


def test_the_card_states_what_it_was_trained_on(facts: dict[str, object]) -> None:
    assert facts["rows"] == 24_467
    assert facts["features"] == 13
    assert facts["rows_dropped_for_leaky_stage"] == 533
    assert facts["seed"] == RANDOM_SEED
    assert "no holdout" in str(facts["evaluation"])


def test_the_card_names_the_excluded_columns(facts: dict[str, object]) -> None:
    assert facts["excluded_to_avoid_leakage"] == EXCLUDED_LEAKY
    assert not set(facts["feature_names"]) & set(EXCLUDED_LEAKY)


def test_the_card_says_the_model_does_not_beat_its_baseline(
    facts: dict[str, object],
) -> None:
    """The single most important line on the card, and it must not be quietly droppable."""
    assert facts["pr_auc"] == 0.193
    assert facts["baseline_pr_auc"] == 0.1914
    assert facts["beats_baseline"] is False


def test_the_card_never_reports_accuracy_without_its_baseline(
    facts: dict[str, object],
) -> None:
    assert facts["accuracy_pct"] == 85.88
    assert facts["majority_baseline_accuracy_pct"] == 85.89
    assert float(facts["accuracy_pct"]) < float(facts["majority_baseline_accuracy_pct"])


def test_every_fact_appears_in_the_rendered_card(facts: dict[str, object]) -> None:
    """So the text and the dict cannot drift apart."""
    rendered = card.model_card(facts=facts)
    for value in (facts["pr_auc"], facts["roc_auc"], facts["baseline_pr_auc"]):
        assert str(value) in rendered
    assert f"{facts['rows']:,}" in rendered
    for column in EXCLUDED_LEAKY:
        assert column in rendered


def test_the_rendered_card_says_no_where_it_matters(facts: dict[str, object]) -> None:
    rendered = card.model_card(facts=facts)
    assert "Beats baseline  : NO" in rendered
    assert "KNOWN LIMITS" in rendered


def test_the_limits_are_measured_results_not_boilerplate(facts: dict[str, object]) -> None:
    """Each limit restates a number established elsewhere in the project."""
    limits = card.known_limits(facts)
    assert len(limits) == 7
    joined = " ".join(limits)
    assert "synthetic" in joined
    assert "0.235" in joined
    assert "44.3% predicted, 20.0% realised" in joined
    assert "causal" in joined


def test_the_card_is_returned_not_written(facts: dict[str, object]) -> None:
    """The workbooks write ``model_card.txt``. Nothing here touches the filesystem."""
    assert isinstance(card.model_card(facts=facts), str)
