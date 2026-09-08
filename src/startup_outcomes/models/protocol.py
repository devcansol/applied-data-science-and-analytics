"""The shared evaluation protocol: one splitter, one metric set, one way to score.

Two decisions live here because leaving either implicit would make every number in the
predictive and prescriptive tiers unreproducible.

**Pooled out-of-fold predictions are the reporting unit.** A metric can be computed by
averaging per-fold values or by scoring a single length-n out-of-fold vector, and on this
data the two disagree in the third decimal — the same order as the model-versus-baseline
difference under study. Averaging is also the wrong operation for a ranking metric like
average precision, which does not decompose over 4,893-row folds. So every headline
number is computed once on the pooled vector, and per-fold values appear only as the
*spread* that structural comparisons are measured against.

**There is no train/test split anywhere in this project.** With 3,453 positives a 25%
holdout carries a PR-AUC standard error near 0.012 — larger than every effect being
measured, and roughly twenty times the model-versus-baseline gap. A holdout that cannot
resolve the effect adds variance and a false sense of rigour, so evaluation is 5-fold
out-of-fold over all rows. The one consequence is that a tuned score has no untouched
data behind it; :mod:`startup_outcomes.models.supervised` labels that figure optimistic
rather than pretending otherwise.

The fold spread is derived from the pooled vector using the same fold indices, so it costs
no extra fitting and cannot disagree with the headline number it qualifies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from startup_outcomes.config import RANDOM_SEED, SIZE_BLOCK

#: Folds for every cross-validated result in the project.
FOLDS = 5

#: The metric that leads. Average precision is PR-AUC: at a 14% positive rate it is the
#: sharper read, and ROC-AUC is over-optimistic under this much imbalance.
HEADLINE_METRIC = "pr_auc"

#: Heavy right-skewed non-negative columns, logged before any distance- or
#: coefficient-based model sees them. ``Runway_Months_2024`` is deliberately absent: its
#: effect is a step at six months, and logging would blur the threshold it encodes.
LOG_COLUMNS = [*SIZE_BLOCK, "Peak_Headcount_2023"]

#: Default decision threshold, kept as a constant so the prescriptive tier can show that
#: nothing about 0.5 is principled.
DEFAULT_THRESHOLD = 0.5


def stratified_folds(n_splits: int = FOLDS) -> StratifiedKFold:
    """The project's only splitter, stratified on the target."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def out_of_fold_probabilities(
    estimator: object, features: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    """Positive-class probability for every row, predicted by a model that did not see it.

    The prescriptive tier ranks on this vector rather than on in-sample predictions. That
    is the single most consequential difference from the reference workbooks, which fit on
    75% of rows and then score all of them — inflating every lift, threshold and expected
    value they report.
    """
    predicted = cross_val_predict(
        estimator,
        features,
        target,
        cv=stratified_folds(),
        method="predict_proba",
    )
    return np.asarray(predicted)[:, 1]


def score_probabilities(target: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """The two ranking metrics, plus the base rate they must be read against.

    The base rate travels with them because average precision has no fixed floor: 0.19
    means nothing until you know that a coin weighted to the prevalence scores 0.14.
    """
    labels = np.asarray(target, dtype=int)
    return {
        "roc_auc": round(float(roc_auc_score(labels, scores)), 3),
        "pr_auc": round(float(average_precision_score(labels, scores)), 3),
        "base_rate_pct": round(float(labels.mean() * 100), 2),
    }


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "roc_auc": lambda labels, scores: float(roc_auc_score(labels, scores)),
        "pr_auc": lambda labels, scores: float(average_precision_score(labels, scores)),
    }


def fold_spread(target: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """Standard deviation of each metric across the folds that produced ``scores``.

    Derived from the pooled vector with the same fold indices, so it costs no extra fits
    and is guaranteed consistent with the headline numbers. This spread is the yardstick:
    a difference between two models smaller than it is reported as no difference.
    """
    labels = np.asarray(target, dtype=int)
    placeholder = np.zeros(len(labels))
    per_fold: dict[str, list[float]] = {name: [] for name in _metric_functions()}
    for _, test_rows in stratified_folds().split(placeholder, labels):
        for name, metric in _metric_functions().items():
            per_fold[name].append(metric(labels[test_rows], scores[test_rows]))
    spread = {}
    for name, values in per_fold.items():
        spread[f"{name}_fold_sd"] = round(float(np.std(values)), 4)
        spread[f"{name}_fold_min"] = round(float(np.min(values)), 4)
        spread[f"{name}_fold_max"] = round(float(np.max(values)), 4)
    return spread


def threshold_counts(
    target: pd.Series,
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Confusion counts at a threshold, in plain language.

    ``accuracy_pct`` appears here and nowhere else in the project, and it is returned
    beside ``majority_baseline_accuracy_pct`` in the same dict so the two cannot be quoted
    apart. On this data that pairing is the whole point: a model can match the baseline's
    accuracy to two decimals while flagging almost nothing.
    """
    labels = np.asarray(target, dtype=int)
    flagged = scores >= threshold
    true_positive = int((flagged & (labels == 1)).sum())
    false_positive = int((flagged & (labels == 0)).sum())
    false_negative = int((~flagged & (labels == 1)).sum())
    true_negative = int((~flagged & (labels == 0)).sum())
    correct = true_positive + true_negative
    majority = max(float(labels.mean()), 1 - float(labels.mean()))
    return {
        "threshold": round(float(threshold), 3),
        "flagged_rows": int(flagged.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "recall_pct": round(true_positive / max(true_positive + false_negative, 1) * 100, 2),
        "precision_pct": round(true_positive / max(true_positive + false_positive, 1) * 100, 2),
        "accuracy_pct": round(correct / len(labels) * 100, 2),
        "majority_baseline_accuracy_pct": round(majority * 100, 2),
        "balanced_accuracy_pct": round(
            (
                true_positive / max(true_positive + false_negative, 1)
                + true_negative / max(true_negative + false_positive, 1)
            )
            / 2
            * 100,
            2,
        ),
        "max_predicted_probability": round(float(scores.max()), 4),
    }


def categorical_columns(features: pd.DataFrame) -> list[str]:
    """Columns carrying the ``category`` dtype."""
    return [
        str(column)
        for column in features.columns
        if isinstance(features[column].dtype, pd.CategoricalDtype)
    ]


def preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """One-hot and scaling for the models that cannot take categories natively.

    Always used inside a :class:`~sklearn.pipeline.Pipeline` so it is refit per fold —
    fitting an encoder or a scaler on all rows first is the subtle leakage the workbooks
    warn about and then commit anyway.

    ``min_frequency`` keeps the 58 city levels from becoming 58 near-empty columns, and
    ``handle_unknown="infrequent_if_exist"`` routes a level unseen in a fold into the
    infrequent bucket instead of raising.
    """
    categorical = categorical_columns(features)
    logged = [column for column in LOG_COLUMNS if column in features.columns]
    linear = [
        str(column)
        for column in features.columns
        if column not in categorical and column not in logged
    ]
    return ColumnTransformer(
        [
            (
                "logged",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                logged,
            ),
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def encoded_pipeline(estimator: object, features: pd.DataFrame) -> Pipeline:
    """Wrap an estimator that needs numeric input in the fold-fitted preprocessor."""
    return Pipeline([("prepare", preprocessor(features)), ("model", estimator)])


def log_numeric_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Scaled, logged, numeric-only view of the features, for clustering and PCA.

    Deliberately *not* the one-hot matrix. K-means on sparse dummies scores a silhouette
    near 0.6 at k=2, which looks like strong structure and is an artifact of splitting on
    a single binary column. Reporting that as a finding would have been wrong, so the
    unsupervised tier is restricted to the continuous columns where a distance means
    something.
    """
    numeric = features.select_dtypes(include=np.number).copy()
    for column in numeric.columns:
        if column in LOG_COLUMNS:
            numeric[column] = np.log1p(numeric[column])
    scaled = StandardScaler().fit_transform(numeric)
    return pd.DataFrame(scaled, columns=numeric.columns, index=numeric.index)


def evaluate(
    name: str,
    estimator: object,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    leakage_controlled: bool = True,
    scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Score one model out-of-fold and package the result.

    ``leakage_controlled`` is mandatory metadata rather than a comment: a partition test
    asserts every reported model carries it and that exactly one entry in the tier is
    ``False``. A prose caption is the first thing lost when a number is copied.

    Args:
        name: Identifies the model in the leaderboard and in the README.
        estimator: Anything with ``fit``/``predict_proba``.
        features: The design matrix.
        target: The binary label.
        leakage_controlled: ``False`` only for the labelled leakage demonstration.
        scores: Pre-computed out-of-fold probabilities, to avoid refitting.
    """
    probabilities = (
        scores if scores is not None else out_of_fold_probabilities(estimator, features, target)
    )
    return {
        "model": name,
        "features": int(features.shape[1]),
        "rows": int(features.shape[0]),
        "leakage_controlled": bool(leakage_controlled),
        **score_probabilities(target, probabilities),
        **fold_spread(target, probabilities),
    }


def leaderboard(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Rank evaluated models by the headline metric.

    Recomputed from its argument every time. The workbooks accumulate results in a
    module-level dict, which makes a notebook's output depend on which cells ran before.
    """
    table = pd.DataFrame(list(results)).set_index("model")
    return table.sort_values(HEADLINE_METRIC, ascending=False)
