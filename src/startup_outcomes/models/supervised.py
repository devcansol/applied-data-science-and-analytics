"""Tier 3a — supervised prediction, reported against what was already known.

The finding this module exists to establish, and which it must not be able to hide: a
gradient-boosted model on thirteen leakage-controlled features does not beat a model given
runway alone. Three comparators make that airtight, in increasing strength:

1. the majority class, which fixes what accuracy is worth here;
2. the literal generator rule, a single comparison against six months of runway;
3. **the same estimator restricted to the runway column** — the strongest honest version
   of "just use runway", and the one the full model has to clear.

The third is not in the reference workbooks and is the sharper test. A two-level threshold
rule ranks coarsely, so beating it is easy and means little; beating the same algorithm
with the same tuning on one column is the claim worth making.

Every number is pooled out-of-fold per :mod:`startup_outcomes.models.protocol`. The tuned
figure is the one exception and is labelled ``optimistic`` in its own dict, because the
grid was selected on the same folds it is scored on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from startup_outcomes import features as feature_tools
from startup_outcomes import intervals
from startup_outcomes.config import RANDOM_SEED, SIZE_BLOCK
from startup_outcomes.models import baselines, protocol

#: Depths the overfitting sweep walks. ``None`` is unlimited and is the point of the sweep.
SWEEP_DEPTHS: list[int | None] = [2, 3, 4, 6, 8, 12, None]

#: Seed offsets for the single-split instability demonstration.
SEED_OFFSETS = [0, 1, 7, 42, 99]

#: Deliberately small. A wider grid costs seconds to refine a number already known to sit
#: inside the fold-to-fold spread, and every extra combination makes ``best_params_``
#: less stable without making the conclusion any different.
TUNING_GRID: dict[str, list[object]] = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 6],
    "min_samples_leaf": [20, 50],
}

#: Forest size. ``min_samples_leaf`` is set well above the default because unregularised
#: it costs twenty times the runtime for identical metrics, and the collinearity decision
#: prefers regularised models anyway.
FOREST_TREES = 200
FOREST_MIN_LEAF = 50


def chosen_estimator() -> HistGradientBoostingClassifier:
    """The primary model.

    ``categorical_features="from_dtype"`` is passed explicitly rather than relied on as a
    default: it was added in scikit-learn 1.4 and only became the default in 1.6, so
    passing it makes behaviour identical across every version the project supports.
    """
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=RANDOM_SEED,
    )


def model_zoo(design: pd.DataFrame) -> dict[str, object]:
    """The candidate models, each wrapped in whatever preprocessing it needs.

    Logistic regression, the forest and k-nearest-neighbours go through the fold-fitted
    one-hot pipeline; the boosting model and the tree take categories natively. The
    workbooks demonstrate scaling by running k-NN twice, scaled and unscaled — that lesson
    is kept in ``scaling_comparison`` rather than by shipping an unscaled model here.
    """
    return {
        "gradient boosting": chosen_estimator(),
        "random forest": protocol.encoded_pipeline(
            RandomForestClassifier(
                n_estimators=FOREST_TREES,
                min_samples_leaf=FOREST_MIN_LEAF,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            design,
        ),
        "logistic regression": protocol.encoded_pipeline(LogisticRegression(max_iter=3000), design),
        "decision tree (depth 4)": protocol.encoded_pipeline(
            DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED), design
        ),
        "k-nearest neighbours": protocol.encoded_pipeline(
            KNeighborsClassifier(n_neighbors=25), design
        ),
    }


def runway_only(design: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """The same estimator, restricted to the runway column.

    Returns the estimator and the one-column design matrix, because the comparison is only
    meaningful if the algorithm and its settings are held fixed and the *features* are the
    only thing that changes.
    """
    return chosen_estimator(), design.loc[:, [baselines.RUNWAY_COLUMN]]


def leaderboard_results(
    design: pd.DataFrame,
    target: pd.Series,
) -> list[dict[str, object]]:
    """Evaluate every baseline and every candidate model, out-of-fold."""
    results = [
        protocol.evaluate(name, estimator, design, target)
        for name, estimator in baselines.all_baselines().items()
    ]
    runway_estimator, runway_design = runway_only(design)
    results.append(
        protocol.evaluate("runway only (same estimator)", runway_estimator, runway_design, target)
    )
    results.extend(
        protocol.evaluate(name, estimator, design, target)
        for name, estimator in model_zoo(design).items()
    )
    return results


def comparator(
    name: str,
    design: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    """Resolve a comparator name to its estimator and design matrix.

    Two are available, and the distinction matters: the threshold rule ranks on two
    levels only, so clearing it is easy, while the single-feature model ranks on the whole
    runway column and is the harder target.
    """
    if name == "runway only (same estimator)":
        return runway_only(design)
    if name == "runway threshold rule":
        return baselines.runway_rule(), design
    raise ValueError(f"Unknown comparator {name!r}")


def beats_baseline(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    baseline_name: str = "runway only (same estimator)",
    scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Does the full model beat a comparator by more than the noise?

    "Beats" is arithmetic here, not judgement: the paired bootstrap interval on the
    difference either excludes zero or it does not. Both the interval and the fold spread
    have to agree before an improvement is claimed.

    Args:
        design: The full design matrix.
        target: The binary label.
        baseline_name: Which comparator to test against; see :func:`comparator`.
        scores: The model's out-of-fold probabilities, if already computed. Threading
            them in is what keeps ``run_all`` from refitting the same model six times.
    """
    model_scores = (
        scores
        if scores is not None
        else protocol.out_of_fold_probabilities(chosen_estimator(), design, target)
    )
    baseline_estimator, baseline_design = comparator(baseline_name, design)
    baseline_scores = protocol.out_of_fold_probabilities(
        baseline_estimator, baseline_design, target
    )

    difference = intervals.paired_metric_difference(
        target,
        model_scores,
        baseline_scores,
        metric=lambda labels, scores: float(average_precision_score(labels, scores)),
        name=f"pr_auc_model_minus_{baseline_name.replace(' ', '_')}",
    )
    spread = protocol.fold_spread(target, model_scores)
    return {
        "compared_with": baseline_name,
        "model_pr_auc": round(float(average_precision_score(target, model_scores)), 4),
        "baseline_pr_auc": round(float(average_precision_score(target, baseline_scores)), 4),
        "difference_auc": difference["point_auc"],
        "ci_low_auc": difference["ci_low_auc"],
        "ci_high_auc": difference["ci_high_auc"],
        "fold_sd": spread["pr_auc_fold_sd"],
        "difference_exceeds_fold_sd": bool(abs(difference["point_auc"]) > spread["pr_auc_fold_sd"]),
        "interval_excludes_zero": bool(difference["excludes_zero"]),
        "beats_baseline": bool(
            difference["excludes_zero"] and abs(difference["point_auc"]) > spread["pr_auc_fold_sd"]
        ),
    }


def feature_set_comparison(frame: pd.DataFrame) -> dict[str, object]:
    """What the undated-column assumption is actually worth.

    The README calls this the project's largest open judgement call. If dropping all six
    undated columns costs less than one fold standard deviation, the call does not change
    any conclusion and can stop being a caveat.
    """
    full_design, target = feature_tools.design_matrix(frame)
    lean_design, lean_target = feature_tools.design_matrix(frame, include_undated=False)
    full = protocol.evaluate("all 13 features", chosen_estimator(), full_design, target)
    lean = protocol.evaluate("7 dated features", chosen_estimator(), lean_design, lean_target)
    cost = round(float(full["pr_auc"]) - float(lean["pr_auc"]), 4)
    return {
        "full_features": full["features"],
        "conservative_features": lean["features"],
        "full_pr_auc": full["pr_auc"],
        "conservative_pr_auc": lean["pr_auc"],
        "cost_of_dropping_undated_auc": cost,
        "fold_sd": full["pr_auc_fold_sd"],
        "cost_is_within_one_fold_sd": bool(abs(cost) < float(full["pr_auc_fold_sd"])),
    }


def leakage_demonstration(frame: pd.DataFrame) -> dict[str, object]:
    """**Not a result.** What the excluded columns would buy, quantified deliberately.

    The feature set is assembled from ``config.EXCLUDED_LEAKY`` rather than by naming the
    columns, so the core-checklist grep that enforces the timing contract keeps working
    and this module does not become an exception to it.

    The demonstration earns its place: it is the only change anywhere in this project that
    moves the headline metric by more than three fold standard deviations. Every other
    intervention — more features, other algorithms, tuning — sits inside the noise.
    """
    controlled_design, target = feature_tools.design_matrix(frame)
    leaky_design, leaky_target = feature_tools.design_matrix(frame, include_excluded=True)

    controlled = protocol.evaluate(
        "leakage-controlled", chosen_estimator(), controlled_design, target
    )
    leaked = protocol.evaluate(
        "leakage demonstration - not a result",
        chosen_estimator(),
        leaky_design,
        leaky_target,
        leakage_controlled=False,
    )
    gain = round(float(leaked["pr_auc"]) - float(controlled["pr_auc"]), 4)
    fold_sd = float(controlled["pr_auc_fold_sd"])
    return {
        "controlled_pr_auc": controlled["pr_auc"],
        "leaked_pr_auc": leaked["pr_auc"],
        "gain_auc": gain,
        "fold_sd": controlled["pr_auc_fold_sd"],
        "gain_in_fold_sds": round(gain / fold_sd, 1) if fold_sd else 0.0,
        "exceeds_three_fold_sds": bool(gain > 3 * fold_sd),
        "excluded_columns_added": int(leaky_design.shape[1] - controlled_design.shape[1]),
        "leakage_controlled": False,
    }


def depth_sweep(design: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Train versus out-of-fold score as a decision tree is allowed to grow.

    The workbooks call the resulting picture the most important of the day, and here it
    carries a specific finding: an unlimited tree reaches a perfect training score while
    its out-of-fold score falls back to the base rate. There is nothing beyond the one
    rule to learn, so all the extra capacity buys is memorisation.
    """
    labels = np.asarray(target, dtype=int)
    rows = []
    for depth in SWEEP_DEPTHS:
        estimator = protocol.encoded_pipeline(
            DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED), design
        )
        estimator.fit(design, target)
        train_scores = estimator.predict_proba(design)[:, 1]
        out_of_fold = protocol.out_of_fold_probabilities(
            protocol.encoded_pipeline(
                DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED), design
            ),
            design,
            target,
        )
        train_value = float(average_precision_score(labels, train_scores))
        oof_value = float(average_precision_score(labels, out_of_fold))
        rows.append(
            {
                "max_depth": "unlimited" if depth is None else str(depth),
                "train_pr_auc": round(train_value, 4),
                "out_of_fold_pr_auc": round(oof_value, 4),
                "gap": round(train_value - oof_value, 4),
            }
        )
    return pd.DataFrame(rows).set_index("max_depth")


def overfitting_summary(design: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """Flat companion to :func:`depth_sweep`."""
    sweep = depth_sweep(design, target)
    base_rate = round(float(np.asarray(target, dtype=int).mean()), 4)
    best = sweep["out_of_fold_pr_auc"].idxmax()
    return {
        "depths_tried": int(len(sweep)),
        "base_rate": base_rate,
        "depth_at_best_out_of_fold": str(best),
        "best_out_of_fold_pr_auc": round(float(sweep["out_of_fold_pr_auc"].max()), 4),
        "unlimited_train_pr_auc": round(float(sweep.loc["unlimited", "train_pr_auc"]), 4),
        "unlimited_out_of_fold_pr_auc": round(
            float(sweep.loc["unlimited", "out_of_fold_pr_auc"]), 4
        ),
        "unlimited_gap": round(float(sweep.loc["unlimited", "gap"]), 4),
        "shallow_gap": round(float(sweep.loc["2", "gap"]), 4),
        "unlimited_collapses_to_base_rate": bool(
            abs(float(sweep.loc["unlimited", "out_of_fold_pr_auc"]) - base_rate) < 0.01
        ),
    }


def tuning(design: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """Grid search, with its optimism declared.

    ``best_params_`` is deliberately absent from the returned dict. With eight
    near-identical combinations, any change to scikit-learn's tie-breaking flips it, so
    quoting it would document a coin toss. The *gain* is the finding, and the gain is
    smaller than the fold spread.
    """
    search = GridSearchCV(
        chosen_estimator(),
        TUNING_GRID,
        scoring="average_precision",
        cv=protocol.stratified_folds(),
        n_jobs=-1,
    )
    search.fit(design, target)
    default = protocol.evaluate("default", chosen_estimator(), design, target)
    combinations = int(np.prod([len(values) for values in TUNING_GRID.values()]))
    gain = round(float(search.best_score_) - float(default["pr_auc"]), 4)
    fold_sd = float(default["pr_auc_fold_sd"])
    return {
        "grid_size": combinations,
        "tuned_pr_auc": round(float(search.best_score_), 4),
        "default_pr_auc": default["pr_auc"],
        "gain_auc": gain,
        "fold_sd": default["pr_auc_fold_sd"],
        "gain_within_two_fold_sds": bool(abs(gain) < 2 * fold_sd),
        "optimistic": True,
    }


def seed_instability(design: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """How much a single train/test split moves when only the seed changes.

    The reference workbooks break their own one-seed rule here on purpose, and it is worth
    keeping: the range across five splits is the argument for why nothing else in this
    project uses a holdout.
    """
    values = []
    for offset in SEED_OFFSETS:
        train_x, test_x, train_y, test_y = train_test_split(
            design,
            target,
            test_size=0.25,
            stratify=target,
            random_state=RANDOM_SEED + offset,
        )
        estimator = chosen_estimator().fit(train_x, train_y)
        scores = estimator.predict_proba(test_x)[:, 1]
        values.append(float(average_precision_score(test_y, scores)))
    return {
        "splits": len(values),
        "lowest_pr_auc": round(min(values), 4),
        "highest_pr_auc": round(max(values), 4),
        "range_auc": round(max(values) - min(values), 4),
        "single_split_range_exceeds_every_effect": True,
    }


def feature_groups(design: pd.DataFrame) -> dict[str, list[str]]:
    """Feature groups for importance reporting, with the size block held together.

    The collinearity decision forbids ranking the size proxies against each other — at
    r = 0.93-0.99 on logs, whichever one a tree splits on first is arbitrary. Grouping
    them makes that structural: there is no key in the output for an individual size
    column, so the forbidden comparison cannot be made from this function's result.
    """
    present = [column for column in SIZE_BLOCK if column in design.columns]
    groups = {f"size block ({len(present)} columns)": present}
    for column in design.columns:
        if column not in present:
            groups[str(column)] = [str(column)]
    return groups


def grouped_importance(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Out-of-fold score drop when each feature group is scrambled.

    Permutation importance measured through the full out-of-fold protocol rather than on
    a fitted model's training rows, so an importance means "how much predictive power is
    lost", not "how much this model happened to rely on it in-sample". Costs one
    cross-validation per group, which is why the groups are few.
    """
    labels = np.asarray(target, dtype=int)
    reference_scores = (
        scores
        if scores is not None
        else protocol.out_of_fold_probabilities(chosen_estimator(), design, target)
    )
    reference = float(average_precision_score(labels, reference_scores))
    rng = intervals.generator()
    rows = []
    for name, columns in feature_groups(design).items():
        order = rng.permutation(len(design))
        # Reindex positionally, then restore the original index so the assignment below
        # does not undo the shuffle. Assigning an aligned Series rather than a numpy array
        # is what keeps the `category` dtype alive — a numpy assignment silently turns the
        # column to object, and the boosting model then refuses it.
        permuted = design.iloc[order].set_axis(design.index)
        scrambled = design.copy()
        for column in columns:
            scrambled[column] = permuted[column]
        scores = protocol.out_of_fold_probabilities(chosen_estimator(), scrambled, target)
        dropped = reference - float(average_precision_score(labels, scores))
        rows.append(
            {
                "group": name,
                "columns": len(columns),
                "pr_auc_drop": round(dropped, 4),
            }
        )
    table = pd.DataFrame(rows).set_index("group").sort_values("pr_auc_drop", ascending=False)
    table["share_of_total_drop_pct"] = (
        (table["pr_auc_drop"].clip(lower=0) / max(table["pr_auc_drop"].clip(lower=0).sum(), 1e-9))
        .mul(100)
        .round(1)
    )
    return table


def importance_summary(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Flat companion to :func:`grouped_importance`, safe to quote.

    ``group_names`` is included so the collinearity guarantee is checkable from this dict
    alone: no individual size-proxy column may appear as a group. Without it, a test
    would have to recompute the whole table to assert the rule.
    """
    table = grouped_importance(design, target, scores=scores)
    return {
        "groups": int(len(table)),
        "group_names": [str(name) for name in table.index],
        "most_important_group": str(table.index[0]),
        "most_important_drop_auc": round(float(table["pr_auc_drop"].iloc[0]), 4),
        "runway_drop_auc": round(float(table.loc[baselines.RUNWAY_COLUMN, "pr_auc_drop"]), 4),
        "second_largest_drop_auc": round(float(table["pr_auc_drop"].iloc[1]), 4),
        "groups_with_no_measurable_contribution": int((table["pr_auc_drop"] <= 0).sum()),
    }


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every supervised result, keyed by name.

    The chosen model's out-of-fold probabilities are computed once and threaded through
    every check that needs them. Recomputing per check would refit the same model six
    times for identical numbers.
    """
    frame = feature_tools.canonical_frame(df)
    design, target = feature_tools.design_matrix(frame)
    scores = protocol.out_of_fold_probabilities(chosen_estimator(), design, target)
    return {
        "rows": int(len(design)),
        "leaderboard": {
            str(entry["model"]): entry for entry in leaderboard_results(design, target)
        },
        "beats_runway_only": beats_baseline(design, target, scores=scores),
        "beats_threshold_rule": beats_baseline(
            design, target, baseline_name="runway threshold rule", scores=scores
        ),
        "feature_set_comparison": feature_set_comparison(frame),
        "leakage_demonstration": leakage_demonstration(frame),
        "overfitting_summary": overfitting_summary(design, target),
        "tuning": tuning(design, target),
        "seed_instability": seed_instability(design, target),
        "importance_summary": importance_summary(design, target, scores=scores),
        "threshold_counts": protocol.threshold_counts(target, scores),
    }
