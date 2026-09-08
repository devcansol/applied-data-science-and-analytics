"""Tests for the shared evaluation protocol.

These describe the CODE — the splitter, the metrics, the encoding. The findings the
protocol produces are pinned in ``test_supervised.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startup_outcomes.config import RANDOM_SEED
from startup_outcomes.models import protocol


def test_the_splitter_is_seeded_from_the_project_seed() -> None:
    splitter = protocol.stratified_folds()
    assert splitter.n_splits == 5
    assert splitter.shuffle is True
    assert splitter.random_state == RANDOM_SEED


def test_the_same_folds_come_back_every_time(design: tuple[pd.DataFrame, pd.Series]) -> None:
    _, target = design
    placeholder = np.zeros(len(target))
    first = [test.tolist() for _, test in protocol.stratified_folds().split(placeholder, target)]
    second = [test.tolist() for _, test in protocol.stratified_folds().split(placeholder, target)]
    assert first == second


def test_every_row_gets_exactly_one_out_of_fold_prediction(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    """Which is what makes the pooled vector a legitimate basis for ranking all rows."""
    features_frame, _ = design
    assert len(oof_scores) == len(features_frame)
    assert bool(((oof_scores >= 0) & (oof_scores <= 1)).all())


def test_the_base_rate_travels_with_every_score(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    """Average precision has no fixed floor, so a PR-AUC without a prevalence is unreadable."""
    _, target = design
    scored = protocol.score_probabilities(target, oof_scores)
    assert set(scored) == {"roc_auc", "pr_auc", "base_rate_pct"}
    assert scored["base_rate_pct"] == 14.11


def test_the_fold_spread_brackets_the_pooled_metric(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    _, target = design
    scored = protocol.score_probabilities(target, oof_scores)
    spread = protocol.fold_spread(target, oof_scores)
    assert spread["pr_auc_fold_min"] <= scored["pr_auc"] <= spread["pr_auc_fold_max"]
    assert spread["pr_auc_fold_sd"] > 0


def test_accuracy_is_never_returned_without_its_baseline(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    """The one place accuracy appears, it appears beside the number that deflates it."""
    _, target = design
    counts = protocol.threshold_counts(target, oof_scores)
    assert "accuracy_pct" in counts
    assert "majority_baseline_accuracy_pct" in counts


def test_the_confusion_counts_add_up(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    _, target = design
    counts = protocol.threshold_counts(target, oof_scores)
    total = (
        counts["true_positive"]
        + counts["false_positive"]
        + counts["false_negative"]
        + counts["true_negative"]
    )
    assert total == 24_467
    assert counts["true_positive"] + counts["false_positive"] == counts["flagged_rows"]


def test_the_encoded_matrix_is_all_numeric(design: tuple[pd.DataFrame, pd.Series]) -> None:
    features_frame, target = design
    prepared = protocol.preprocessor(features_frame).fit_transform(features_frame, target)
    assert np.isfinite(np.asarray(prepared, dtype=float)).all()


def test_infrequent_levels_are_pooled_rather_than_exploded(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The one-hot path stays narrower than a naive encoding.

    Encoding every level of all six categoricals would give 119 dummy columns plus the 7
    numerics. ``min_frequency=0.01`` pools the levels below 245 rows, landing at 112 —
    modest here, but it is what stops a future high-cardinality column from dominating
    the matrix, and it is why an unseen level in a fold cannot raise.
    """
    features_frame, target = design
    prepared = protocol.preprocessor(features_frame).fit_transform(features_frame, target)
    naive_width = (
        sum(
            features_frame[column].cat.categories.size
            for column in protocol.categorical_columns(features_frame)
        )
        + 7
    )
    assert naive_width == 126
    assert np.asarray(prepared).shape[1] == 112


def test_the_clustering_matrix_excludes_the_one_hot_columns(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The correctness requirement behind the unsupervised tier.

    K-means on sparse dummies scores a high silhouette at k=2 purely by splitting on one
    binary column. Restricting to the continuous columns is what keeps a distance
    meaningful, so this asserts the restriction rather than trusting it.
    """
    features_frame, _ = design
    matrix = protocol.log_numeric_matrix(features_frame)
    assert matrix.shape[1] == 7
    assert not set(matrix.columns) & set(protocol.categorical_columns(features_frame))
    # Standardised, so every column is centred and unit-scaled.
    assert bool((matrix.mean().abs() < 1e-9).all())
    assert bool(((matrix.std() - 1.0).abs() < 0.01).all())


def test_the_size_columns_are_logged_and_runway_is_not(
    design: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Logging runway would blur the six-month step the generator encoded."""
    features_frame, _ = design
    assert "Runway_Months_2024" not in protocol.LOG_COLUMNS
    assert "Founding_Year" not in protocol.LOG_COLUMNS
    for column in ("Total_Funding_USD_Millions", "Valuation_USD_Millions"):
        assert column in protocol.LOG_COLUMNS


def test_the_leaderboard_is_recomputed_not_accumulated() -> None:
    """No module-level result registry, so a call cannot depend on earlier calls."""
    entries = [
        {"model": "a", "pr_auc": 0.10},
        {"model": "b", "pr_auc": 0.20},
    ]
    first = protocol.leaderboard(entries)
    second = protocol.leaderboard(entries)
    assert list(first.index) == ["b", "a"]
    assert first.equals(second)
