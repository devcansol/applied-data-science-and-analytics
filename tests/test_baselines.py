"""Tests for the baselines, especially the runway threshold rule.

The rule is a real estimator rather than a hard-coded constant, so it needs real tests:
it must learn its rates from the training fold only, and it must behave like a classifier
the rest of the protocol can score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from startup_outcomes.audit import RUNWAY_CLIFF_MONTHS
from startup_outcomes.config import RANDOM_SEED
from startup_outcomes.models import baselines, protocol


def test_the_three_baselines_are_ordered_by_what_they_know() -> None:
    assert list(baselines.all_baselines()) == [
        "majority baseline",
        "stratified baseline",
        "runway threshold rule",
    ]


def test_the_stratified_baseline_is_seeded() -> None:
    assert baselines.stratified_baseline().random_state == RANDOM_SEED


def test_the_threshold_comes_from_the_audit_not_from_tuning() -> None:
    """It was located from the data before any model existed, so it is not fitted to a metric."""
    rule = baselines.runway_rule()
    assert rule.threshold == RUNWAY_CLIFF_MONTHS == 6.0
    assert rule.column == "Runway_Months_2024"


def test_the_rule_learns_its_rates_rather_than_hard_coding_them(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    features_frame, target = design
    rule = baselines.runway_rule().fit(features_frame, target)
    check_is_fitted(rule, "rate_below_")
    # The audit reports 23.41% / 10.98% on all 25,000 rows; these are the same rates on
    # the 24,467-row canonical frame, which drops 533 non-closed stage-IPO companies.
    assert round(rule.rate_below_ * 100, 2) == 23.59
    assert round(rule.rate_at_or_above_ * 100, 2) == 11.27
    assert rule.rate_below_ > rule.rate_at_or_above_


def test_the_rule_returns_only_two_distinct_probabilities(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Which is exactly why it ranks more coarsely than the runway column itself."""
    features_frame, target = design
    rule = baselines.runway_rule().fit(features_frame, target)
    probabilities = rule.predict_proba(features_frame)[:, 1]
    assert len(np.unique(probabilities)) == 2


def test_the_rule_never_crosses_the_default_threshold(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Even below six months of runway, fewer than a quarter of companies close."""
    features_frame, target = design
    rule = baselines.runway_rule().fit(features_frame, target)
    assert int(rule.predict(features_frame).sum()) == 0
    assert float(rule.predict_proba(features_frame)[:, 1].max()) < 0.5


def test_probabilities_sum_to_one(design: tuple[pd.DataFrame, pd.Series]) -> None:
    features_frame, target = design
    rule = baselines.runway_rule().fit(features_frame, target)
    probabilities = rule.predict_proba(features_frame)
    assert bool(np.allclose(probabilities.sum(axis=1), 1.0))


def test_an_unfitted_rule_refuses_to_predict(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    features_frame, _ = design
    with pytest.raises(Exception, match="fitted|is not fitted"):
        baselines.runway_rule().predict_proba(features_frame)


def test_the_rule_survives_a_fold_with_nothing_below_the_threshold() -> None:
    """The guard that keeps cross-validation from producing a NaN probability."""
    frame = pd.DataFrame({"Runway_Months_2024": [10.0, 20.0, 30.0, 40.0]})
    target = pd.Series([0, 1, 0, 1])
    rule = baselines.runway_rule().fit(frame, target)
    assert rule.rate_below_ == 0.5
    assert bool(np.isfinite(rule.predict_proba(frame)).all())


def test_the_majority_baseline_scores_the_base_rate_on_pr_auc(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """An analytic identity, so this test catches a mis-wired label vector instantly.

    A constant score gives average precision exactly equal to the prevalence.
    """
    features_frame, target = design
    result = protocol.evaluate(
        "majority baseline", baselines.majority_baseline(), features_frame, target
    )
    assert result["pr_auc"] == round(float(target.mean()), 3)
    assert result["roc_auc"] == 0.5
