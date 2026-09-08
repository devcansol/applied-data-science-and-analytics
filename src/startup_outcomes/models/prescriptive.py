"""Tier 4 — prescriptive: turning a score into a decision, and pricing that decision.

**Read this before quoting anything from this module.** The data is synthetic. No company
in it exists, no cost or value here was measured, and nothing this module returns is a
recommendation about anything. What it demonstrates is the *arithmetic* that converts a
ranking into a decision — and, applied honestly to this ranking, that arithmetic returns a
negative verdict, which is the actual finding.

Four rules keep the tier from reading as business advice:

* **No currency.** Costs and values are unitless relative weights, named ``_cost_units``
  and ``_value_units``. A number with a currency symbol gets quoted as a result.
* **Every output ships with its trivial comparator.** A cheapest threshold means nothing
  without the cost of flagging everyone and flagging nobody beside it. Under the
  illustrative weights the cost-optimal policy here turns out to be very nearly a blanket
  policy, which is the honest headline.
* **Hit rates are realised, never predicted.** The workbooks compute the success rate of a
  targeted group as the mean of the model's own scores — the model grading its own
  homework. Both numbers are reported here, and the gap between them is a calibration
  finding.
* **Every returned dict carries ``basis``.** Machine-readable, because a prose caveat is
  the first thing lost when a number is pasted into a slide.

Scores come from :func:`~startup_outcomes.models.protocol.out_of_fold_probabilities`, so
no company is ever ranked by a model that trained on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from startup_outcomes import features as feature_tools
from startup_outcomes import intervals
from startup_outcomes.config import ID_COLUMN
from startup_outcomes.models import protocol, supervised

#: Marks every number this module returns as a methodology demonstration on simulated
#: data. Asserted by a partition test; never remove it to make an output tidier.
BASIS = "synthetic-data-methodology-demo"

#: Relative cost of missing a closure against raising a false alarm. Unitless: the ratio
#: is the only thing that matters, and it is the reader's to change.
COST_MISS_UNITS = 10.0
COST_FALSE_ALARM_UNITS = 1.0

#: Illustrative intervention economics, all unitless.
CAPACITY = 30
COST_PER_ACTION_UNITS = 100.0
VALUE_PER_SUCCESS_UNITS = 900.0
SUCCESS_RATE = 0.40

#: Thresholds the cost sweep walks.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

#: Success rates the worth-doing table walks.
SENSITIVITY_RATES = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

#: Fraction of the ranking treated as "the top" in the targeting mix.
TOP_FRACTION = 0.20

#: Points in the one-way response curve, and the percentile range it sweeps.
CURVE_POINTS = 25
CURVE_LOW_PERCENTILE, CURVE_HIGH_PERCENTILE = 5, 95


def response_curve(
    design: pd.DataFrame,
    target: pd.Series,
    feature: str,
    *,
    points: int = CURVE_POINTS,
) -> pd.DataFrame:
    """How the model's predicted probability moves as one feature is swept.

    Averaged over all rows via ``partial_dependence`` rather than swept on one arbitrary
    row as the workbooks do — a single row's curve is a property of that row.

    **This is a property of the model, not of the world.** It says how the prediction
    changes if the input changes, which is not what would happen if a company's runway
    actually changed. Every serious analyst is eventually asked to forget that distinction.
    """
    estimator = supervised.chosen_estimator().fit(design, target)
    low, high = np.percentile(
        design[feature].to_numpy(dtype=float),
        [CURVE_LOW_PERCENTILE, CURVE_HIGH_PERCENTILE],
    )
    grid = np.linspace(low, high, points)
    computed = partial_dependence(
        estimator,
        design,
        features=[design.columns.get_loc(feature)],
        grid_resolution=points,
        custom_values={design.columns.get_loc(feature): grid},
        method="brute",
        response_method="predict_proba",
    )
    return pd.DataFrame(
        {
            feature: np.round(np.asarray(computed["grid_values"][0], dtype=float), 3),
            "predicted_closed_pct": np.round(np.asarray(computed["average"][0]) * 100, 2),
        }
    ).set_index(feature)


def response_summary(design: pd.DataFrame, target: pd.Series, feature: str) -> dict[str, object]:
    """Flat companion to :func:`response_curve`: the swing across a realistic range."""
    curve = response_curve(design, target, feature)
    values = curve["predicted_closed_pct"]
    return {
        "feature": feature,
        "low_end_pct": round(float(values.iloc[0]), 2),
        "high_end_pct": round(float(values.iloc[-1]), 2),
        "swing_pp": round(float(values.iloc[0] - values.iloc[-1]), 2),
        "points": int(len(curve)),
        "is_model_behaviour_not_causal_effect": True,
        "basis": BASIS,
    }


def ranked_actions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    top: int = 15,
) -> pd.DataFrame:
    """The top of the ranking, with identifiers.

    Not a target list. This is the top of a ranking produced by a model that does not beat
    a single column, over companies that do not exist.
    """
    ranked = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].to_numpy(),
            "score": np.round(scores, 4),
            "runway_months": frame["Runway_Months_2024"].to_numpy(),
            "domain": frame["Domain"].astype(str).to_numpy(),
        }
    )
    ranked["rank"] = ranked["score"].rank(ascending=False, method="first").astype(int)
    return ranked.nlargest(top, "score").set_index("rank")


def targeting_mix(
    frame: pd.DataFrame,
    scores: np.ndarray,
    column: str,
    *,
    top_fraction: float = TOP_FRACTION,
) -> pd.DataFrame:
    """Which categories are over-represented at the top of the ranking.

    The targeting insight the workbooks reach for. Here it mostly restates the runway
    finding through a different lens, which is worth seeing.
    """
    cutoff = int(len(frame) * top_fraction)
    top_rows = np.argsort(-scores)[:cutoff]
    overall = frame[column].value_counts(normalize=True) * 100
    selected = frame.iloc[top_rows][column].value_counts(normalize=True) * 100
    table = pd.DataFrame(
        {
            "top_pct": selected.round(2),
            "everyone_pct": overall.round(2),
        }
    ).fillna(0.0)
    table["over_under_pp"] = (table["top_pct"] - table["everyone_pct"]).round(2)
    return table.sort_values("over_under_pp", ascending=False)


def cost_threshold_sweep(
    target: pd.Series,
    scores: np.ndarray,
    *,
    cost_miss: float = COST_MISS_UNITS,
    cost_false_alarm: float = COST_FALSE_ALARM_UNITS,
) -> pd.DataFrame:
    """Total cost at each threshold, given a relative price for each kind of error.

    The threshold is a business decision, not a technical one, and this is the table that
    shows it: nothing about 0.5 is principled, and the cheapest cut moves as soon as the
    cost ratio does.
    """
    labels = np.asarray(target, dtype=int)
    rows = []
    for threshold in THRESHOLD_GRID:
        flagged = scores >= threshold
        false_positive = int((flagged & (labels == 0)).sum())
        false_negative = int((~flagged & (labels == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "flagged_rows": int(flagged.sum()),
                "false_alarms": false_positive,
                "missed_closures": false_negative,
                "total_cost_units": round(
                    false_positive * cost_false_alarm + false_negative * cost_miss, 1
                ),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def cheapest_threshold(
    target: pd.Series,
    scores: np.ndarray,
    *,
    cost_miss: float = COST_MISS_UNITS,
    cost_false_alarm: float = COST_FALSE_ALARM_UNITS,
) -> dict[str, object]:
    """The cost-minimising threshold, beside the policies that need no model at all.

    The comparators are what make this reportable. A cheapest threshold quoted alone
    implies the model is steering the decision; quoted against "flag everyone" it shows
    how little steering is actually happening.
    """
    sweep = cost_threshold_sweep(
        target, scores, cost_miss=cost_miss, cost_false_alarm=cost_false_alarm
    )
    labels = np.asarray(target, dtype=int)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    flag_everyone = negatives * cost_false_alarm
    flag_nobody = positives * cost_miss

    best_threshold = float(sweep["total_cost_units"].idxmin())
    best_cost = float(sweep.loc[best_threshold, "total_cost_units"])
    default_cost = float(sweep.loc[protocol.DEFAULT_THRESHOLD, "total_cost_units"])
    # Where a perfectly calibrated probability would sit: flag when p * cost_miss exceeds
    # (1 - p) * cost_false_alarm.
    indifference = cost_false_alarm / (cost_false_alarm + cost_miss)
    cheapest_trivial = min(flag_everyone, flag_nobody)
    return {
        "cost_miss_units": cost_miss,
        "cost_false_alarm_units": cost_false_alarm,
        "analytic_indifference_threshold": round(indifference, 4),
        "cheapest_threshold": round(best_threshold, 2),
        "cheapest_cost_units": round(best_cost, 1),
        "flagged_at_cheapest": int(sweep.loc[best_threshold, "flagged_rows"]),
        "flagged_share_pct": round(
            float(sweep.loc[best_threshold, "flagged_rows"]) / len(labels) * 100, 1
        ),
        "cost_flag_everyone_units": round(float(flag_everyone), 1),
        "cost_flag_nobody_units": round(float(flag_nobody), 1),
        "cost_at_default_threshold_units": round(default_cost, 1),
        "saving_vs_best_trivial_policy_pct": round(
            (cheapest_trivial - best_cost) / cheapest_trivial * 100, 2
        ),
        "basis": BASIS,
    }


def capacity_expected_value(
    target: pd.Series,
    scores: np.ndarray,
    *,
    capacity: int = CAPACITY,
    cost_per_action: float = COST_PER_ACTION_UNITS,
    value_per_success: float = VALUE_PER_SUCCESS_UNITS,
    success_rate: float = SUCCESS_RATE,
) -> dict[str, object]:
    """Expected value of acting on the top of the ranking under a capacity limit.

    Reports the *realised* closure rate among the targeted rows from their true labels,
    and the model's own average score separately. The workbooks use the latter as the hit
    rate, which lets an over-confident model inflate its own business case.

    The realised rate carries a bootstrap interval, because at a capacity of thirty it
    rests on a handful of events and a point estimate would be indefensible.
    """
    labels = np.asarray(target, dtype=int)
    acted = np.argsort(-scores)[:capacity]
    hits = int(labels[acted].sum())
    realised = float(labels[acted].mean())
    base_rate = float(labels.mean())

    acted_labels = labels[acted].astype(float)
    interval = intervals.bootstrap_rows(
        capacity,
        lambda rows: float(acted_labels[rows].mean() * 100),
        name="realised_closure_rate_in_targeted_group",
    )

    spend = capacity * cost_per_action
    expected_return = capacity * realised * success_rate * value_per_success
    break_even = cost_per_action / (realised * value_per_success) if realised > 0 else float("inf")
    return {
        "capacity": int(capacity),
        "cost_per_action_units": cost_per_action,
        "value_per_success_units": value_per_success,
        "assumed_success_rate": success_rate,
        "hits_in_targeted_group": hits,
        "realised_closure_rate_pct": round(realised * 100, 2),
        "realised_rate_ci_low_pct": interval["ci_low_pp"],
        "realised_rate_ci_high_pct": interval["ci_high_pp"],
        "model_predicted_rate_pct": round(float(scores[acted].mean() * 100), 2),
        "base_rate_pct": round(base_rate * 100, 2),
        "lift_vs_random": round(realised / base_rate, 2) if base_rate else 0.0,
        "spend_units": round(spend, 1),
        "expected_return_units": round(expected_return, 1),
        "net_units": round(expected_return - spend, 1),
        "break_even_success_rate": round(break_even, 3),
        # The decisive honesty check: if the interval on the realised rate spans the base
        # rate, targeting by this model is indistinguishable from picking at random.
        "indistinguishable_from_random": bool(
            interval["ci_low_pp"] <= base_rate * 100 <= interval["ci_high_pp"]
        ),
        "model_overstates_its_own_hit_rate": bool(scores[acted].mean() > realised),
        "basis": BASIS,
    }


def value_sensitivity(
    target: pd.Series,
    scores: np.ndarray,
    *,
    capacity: int = CAPACITY,
    cost_per_action: float = COST_PER_ACTION_UNITS,
    value_per_success: float = VALUE_PER_SUCCESS_UNITS,
) -> pd.DataFrame:
    """Net value across a range of assumed success rates.

    The column is ``net_positive`` — a statement about arithmetic — rather than "worth
    doing", which would be advice this data cannot support.
    """
    labels = np.asarray(target, dtype=int)
    acted = np.argsort(-scores)[:capacity]
    realised = float(labels[acted].mean())
    spend = capacity * cost_per_action
    rows = []
    for rate in SENSITIVITY_RATES:
        expected_return = capacity * realised * rate * value_per_success
        rows.append(
            {
                "assumed_success_rate": rate,
                "expected_return_units": round(expected_return, 1),
                "net_units": round(expected_return - spend, 1),
                "net_positive": bool(expected_return > spend),
            }
        )
    return pd.DataFrame(rows).set_index("assumed_success_rate")


def lift_curve(target: pd.Series, scores: np.ndarray, *, points: int = 20) -> pd.DataFrame:
    """Hit rate and lift as a function of how far down the ranking you act."""
    labels = np.asarray(target, dtype=int)
    order = np.argsort(-scores)
    base_rate = float(labels.mean())
    rows = []
    for step in range(1, points + 1):
        cutoff = max(int(len(labels) * step / points), 1)
        hit_rate = float(labels[order[:cutoff]].mean())
        rows.append(
            {
                "targeted_pct": round(step / points * 100, 1),
                "rows": cutoff,
                "hit_rate_pct": round(hit_rate * 100, 2),
                "lift": round(hit_rate / base_rate, 3) if base_rate else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("targeted_pct")


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every prescriptive result, keyed by name.

    Scores are computed once out-of-fold and threaded through every check.
    """
    frame = feature_tools.canonical_frame(df)
    design, target = feature_tools.design_matrix(frame)
    scores = protocol.out_of_fold_probabilities(supervised.chosen_estimator(), design, target)
    curve = lift_curve(target, scores)
    return {
        "rows": int(len(design)),
        "basis": BASIS,
        "cheapest_threshold": cheapest_threshold(target, scores),
        "capacity_expected_value": capacity_expected_value(target, scores),
        "top_decile_lift": float(curve.loc[10.0, "lift"]),
        "top_fifth_lift": float(curve.loc[20.0, "lift"]),
        "response_to_runway": response_summary(design, target, "Runway_Months_2024"),
    }
