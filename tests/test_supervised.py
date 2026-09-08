"""Pins the predictive findings quoted in the README.

**Two kinds of assertion live here, and the distinction is the point.**

*Structural* assertions state a claim as a relation — "no model beats the runway column",
"only leakage helps" — and are the real tests. *Value pins* exist only because the README
quotes digits; they are paired with a band test so a library upgrade that moves a third
decimal is distinguishable from a change that moves a conclusion.

If a pin fails and its band passes, the drift is cosmetic. If both fail, the finding
changed: stop and investigate. Do not widen a band to make a test pass.
"""

from __future__ import annotations

import pandas as pd

from startup_outcomes.config import SIZE_BLOCK
from startup_outcomes.models import supervised

UPGRADE_HINT = (
    "Model metrics moved. Check the installed scikit-learn against uv.lock (the README's "
    "Reproducibility section records the versions these pins were generated under). If a "
    "library changed and the structural tests in this file still pass, this is cosmetic "
    'drift: regenerate with `uv run python -c "from startup_outcomes.models.supervised '
    "import run_all; print(run_all())\"`, then update these pins and the README's "
    "predictive table in ONE commit. If a structural test also fails, the FINDING changed "
    "- investigate; do not widen the band and do not update the pin."
)

#: Half the width of the acceptable band around each pinned metric. Roughly 1.5 fold
#: standard deviations, so genuine noise passes and a real change does not.
BAND = 0.01


def test_the_documented_model_metrics_are_exact(supervised_results: dict[str, object]) -> None:
    board = supervised_results["leaderboard"]
    assert board["gradient boosting"]["pr_auc"] == 0.193, UPGRADE_HINT
    assert board["gradient boosting"]["roc_auc"] == 0.595, UPGRADE_HINT
    assert board["runway only (same estimator)"]["pr_auc"] == 0.191, UPGRADE_HINT
    assert board["runway threshold rule"]["pr_auc"] == 0.186, UPGRADE_HINT
    assert board["random forest"]["pr_auc"] == 0.194, UPGRADE_HINT
    assert board["logistic regression"]["pr_auc"] == 0.171, UPGRADE_HINT
    assert board["k-nearest neighbours"]["pr_auc"] == 0.157, UPGRADE_HINT
    assert board["majority baseline"]["pr_auc"] == 0.141, UPGRADE_HINT


def test_model_metrics_stay_inside_the_documented_band(
    supervised_results: dict[str, object],
) -> None:
    """Distinguishes cosmetic drift from a changed conclusion. See the module docstring."""
    board = supervised_results["leaderboard"]
    for model, expected in (
        ("gradient boosting", 0.193),
        ("runway only (same estimator)", 0.191),
        ("runway threshold rule", 0.186),
        ("random forest", 0.194),
        ("logistic regression", 0.171),
    ):
        assert abs(float(board[model]["pr_auc"]) - expected) < BAND, model


def test_no_model_beats_the_runway_column(supervised_results: dict[str, object]) -> None:
    """The headline, and it is an inequality rather than a pin.

    Twelve extra features buy 0.0016 average precision over the same estimator given only
    ``Runway_Months_2024``. The paired bootstrap interval contains zero and the difference
    is a quarter of one fold standard deviation, so per the baselines decision this is
    reported as not beating it.
    """
    comparison = supervised_results["beats_runway_only"]
    assert comparison["beats_baseline"] is False
    assert comparison["interval_excludes_zero"] is False
    assert comparison["difference_exceeds_fold_sd"] is False
    assert comparison["ci_low_auc"] < 0 < comparison["ci_high_auc"]
    assert abs(float(comparison["difference_auc"])) < float(comparison["fold_sd"])


def test_the_model_does_beat_the_coarse_threshold_rule(
    supervised_results: dict[str, object],
) -> None:
    """The honest nuance, kept so the headline is not overstated.

    The two-level rule ranks coarsely — it emits exactly two probabilities — so the model
    clears it by 0.0072 with an interval excluding zero. That says the runway *column*
    carries more than the six-month step alone, not that the other twelve features help.
    """
    comparison = supervised_results["beats_threshold_rule"]
    assert comparison["beats_baseline"] is True
    assert comparison["interval_excludes_zero"] is True
    assert comparison["ci_low_auc"] > 0


def test_a_single_split_moves_more_than_the_effect_being_measured(
    supervised_results: dict[str, object],
) -> None:
    """Why this project has no holdout.

    Changing only the split seed moves average precision by 0.0067 — four times the
    model-versus-runway difference of 0.0016. A holdout could not have resolved the
    comparison at all.
    """
    instability = supervised_results["seed_instability"]
    comparison = supervised_results["beats_runway_only"]
    assert instability["splits"] == 5
    assert instability["range_auc"] == 0.0067
    assert float(instability["range_auc"]) > abs(float(comparison["difference_auc"]))


def test_the_model_never_meaningfully_flags_a_closure_at_the_default_threshold(
    supervised_results: dict[str, object],
) -> None:
    """It is a ranker, not a classifier — and it is worse than guessing on accuracy.

    Two of 24,467 rows cross 0.5, and both are false positives. So accuracy lands at
    85.88% against the majority baseline's 85.89%: the model is fractionally *worse* than
    predicting "survives" for everyone, while being a better ranker than chance.
    """
    counts = supervised_results["threshold_counts"]
    assert counts["flagged_rows"] == 2
    assert counts["true_positive"] == 0
    assert float(counts["max_predicted_probability"]) < 0.55
    assert counts["balanced_accuracy_pct"] == 50.0
    assert float(counts["accuracy_pct"]) < float(counts["majority_baseline_accuracy_pct"])


def test_dropping_the_undated_columns_costs_less_than_the_fold_spread(
    supervised_results: dict[str, object],
) -> None:
    """Retires the project's largest documented judgement call.

    The README flags the undated financial columns as an assumption results should be
    checked without. Checked: dropping all six costs 0.004 average precision, under one
    fold standard deviation. The assumption does not change any conclusion.
    """
    comparison = supervised_results["feature_set_comparison"]
    assert comparison["full_features"] == 13
    assert comparison["conservative_features"] == 7
    assert comparison["cost_of_dropping_undated_auc"] == 0.004
    assert comparison["cost_is_within_one_fold_sd"] is True


def test_only_leakage_produces_a_detectable_improvement(
    supervised_results: dict[str, object],
) -> None:
    """The one intervention in this whole project that clears the noise.

    Adding the two outcome-contemporaneous columns lifts average precision by 0.042 —
    nearly seven fold standard deviations, where every legitimate change sits inside one.
    That contrast is the argument for the timing contract.
    """
    demonstration = supervised_results["leakage_demonstration"]
    assert demonstration["excluded_columns_added"] == 2
    assert demonstration["gain_auc"] == 0.042
    assert demonstration["exceeds_three_fold_sds"] is True
    assert float(demonstration["gain_in_fold_sds"]) > 6.0
    assert demonstration["leakage_controlled"] is False


def test_every_reported_model_declares_whether_it_is_leakage_controlled(
    supervised_results: dict[str, object],
) -> None:
    """A partition test, in the spirit of the timing-group contract.

    Machine-readable rather than a prose caption, because a caption is the first thing
    lost when a number is pasted somewhere else.
    """
    board = supervised_results["leaderboard"]
    assert all("leakage_controlled" in entry for entry in board.values())
    assert all(entry["leakage_controlled"] for entry in board.values())
    assert supervised_results["leakage_demonstration"]["leakage_controlled"] is False


def test_the_tree_stops_learning_almost_immediately(
    supervised_results: dict[str, object],
) -> None:
    """There is nothing beyond the one rule to find.

    An unlimited tree reaches a perfect training score and its out-of-fold score falls to
    0.145 — indistinguishable from the 0.141 base rate. The best depth is 2: one split.
    """
    summary = supervised_results["overfitting_summary"]
    assert summary["depths_tried"] == 7
    assert summary["unlimited_train_pr_auc"] == 1.0
    assert summary["depth_at_best_out_of_fold"] == "2"
    assert summary["unlimited_collapses_to_base_rate"] is True
    assert float(summary["unlimited_gap"]) > 100 * abs(float(summary["shallow_gap"]))


def test_tuning_buys_less_than_the_fold_spread(supervised_results: dict[str, object]) -> None:
    tuning = supervised_results["tuning"]
    assert tuning["grid_size"] == 8
    assert tuning["gain_auc"] == 0.007
    assert tuning["gain_within_two_fold_sds"] is True
    assert tuning["optimistic"] is True


def test_the_best_parameters_are_never_reported(
    supervised_results: dict[str, object],
) -> None:
    """With eight near-tied combinations, ``best_params_`` documents a coin toss.

    Any change to scikit-learn's tie-breaking would flip it, so it is deliberately absent
    from the returned dict and must never reach the README.
    """
    assert "best_params" not in supervised_results["tuning"]
    assert "best_params_" not in supervised_results["tuning"]


def test_runway_dominates_every_other_feature_group(
    supervised_results: dict[str, object],
) -> None:
    """Scrambling runway costs 0.033; the next-largest group — the whole size block
    of four correlated columns — costs 0.0046, seven times less."""
    summary = supervised_results["importance_summary"]
    assert summary["most_important_group"] == "Runway_Months_2024"
    assert summary["runway_drop_auc"] == 0.0327
    assert float(summary["most_important_drop_auc"]) == float(summary["runway_drop_auc"])
    assert summary["second_largest_drop_auc"] == 0.0046
    assert float(summary["runway_drop_auc"]) > 5 * float(summary["second_largest_drop_auc"])


def test_importance_is_only_ever_reported_by_block(
    supervised_results: dict[str, object],
) -> None:
    """Makes the collinearity decision structurally unbreakable.

    At r = 0.93-0.99 on logs, whichever size proxy a tree splits on first is arbitrary.
    There is no key in this output for an individual size column, so the forbidden
    comparison cannot be made from it even by accident.
    """
    names = supervised_results["importance_summary"]["group_names"]
    assert not set(names) & set(SIZE_BLOCK)
    assert "size block (4 columns)" in names


def test_every_feature_appears_in_exactly_one_importance_group(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    features_frame, _ = design
    groups = supervised.feature_groups(features_frame)
    covered = [column for columns in groups.values() for column in columns]
    assert sorted(covered) == sorted(features_frame.columns)
    assert len(covered) == len(set(covered))


def test_the_conservative_estimator_passes_the_categorical_flag_explicitly() -> None:
    """Behaviour must be identical on scikit-learn 1.6 through 1.9, not default-dependent."""
    assert supervised.chosen_estimator().categorical_features == "from_dtype"


def test_two_models_score_worse_than_a_single_column(
    supervised_results: dict[str, object],
) -> None:
    """Logistic regression and k-NN lose to runway alone, which is worth documenting."""
    board = supervised_results["leaderboard"]
    runway = float(board["runway only (same estimator)"]["pr_auc"])
    assert float(board["logistic regression"]["pr_auc"]) < runway
    assert float(board["k-nearest neighbours"]["pr_auc"]) < runway
