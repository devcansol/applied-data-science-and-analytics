"""The model card: what the model is, what it scores, and what it must not be used for.

The reference workbooks write this to ``model_card.txt`` with hand-typed fields. Here it
is a rendering of :func:`card_facts`, a flat dict every value of which is pinned by a
test. That difference matters: a hand-typed card drifts from the model the moment either
changes, and nothing catches it. Nothing is written to disk — the card is returned, the
notebook prints it, and the README quotes it.

The ``KNOWN LIMITS`` block is not boilerplate. Every line in it is a measured result from
elsewhere in this project, and the card would be misleading without them.
"""

from __future__ import annotations

import pandas as pd

from startup_outcomes import features as feature_tools
from startup_outcomes.config import EXCLUDED_LEAKY, RANDOM_SEED, TARGET, model_features
from startup_outcomes.models import protocol, supervised

#: Width of the rendered card's rule lines.
CARD_WIDTH = 78


def card_facts(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every fact the card states, as a flat dict."""
    frame = feature_tools.canonical_frame(df)
    design, target = feature_tools.design_matrix(frame)
    scores = protocol.out_of_fold_probabilities(supervised.chosen_estimator(), design, target)
    scored = protocol.score_probabilities(target, scores)
    spread = protocol.fold_spread(target, scores)
    counts = protocol.threshold_counts(target, scores)
    comparison = supervised.beats_baseline(design, target, scores=scores)

    return {
        "predicts": f"{TARGET} == 'Closed' within the 2026 snapshot",
        "task": "binary classification, reported as a ranking",
        "algorithm": "HistGradientBoostingClassifier (categorical_features='from_dtype')",
        "rows": int(len(design)),
        "features": int(design.shape[1]),
        "feature_names": list(model_features()),
        "excluded_to_avoid_leakage": list(EXCLUDED_LEAKY),
        "rows_dropped_for_leaky_stage": 533,
        "evaluation": "5-fold stratified, pooled out-of-fold predictions; no holdout",
        "seed": int(RANDOM_SEED),
        "pr_auc": scored["pr_auc"],
        "roc_auc": scored["roc_auc"],
        "base_rate_pct": scored["base_rate_pct"],
        "pr_auc_fold_sd": spread["pr_auc_fold_sd"],
        "baseline_name": comparison["compared_with"],
        "baseline_pr_auc": comparison["baseline_pr_auc"],
        "beats_baseline": comparison["beats_baseline"],
        "flagged_at_default_threshold": counts["flagged_rows"],
        "accuracy_pct": counts["accuracy_pct"],
        "majority_baseline_accuracy_pct": counts["majority_baseline_accuracy_pct"],
    }


def known_limits(facts: dict[str, object]) -> list[str]:
    """The limits that must travel with any quotation of this model's score."""
    return [
        "The data is synthetic. Nothing this model learns transfers to real companies.",
        (
            f"It does not beat its baseline: PR-AUC {facts['pr_auc']} against "
            f"{facts['baseline_pr_auc']} for the same estimator given runway alone."
        ),
        (
            f"It is not usable as a classifier. At threshold 0.5 it flags "
            f"{facts['flagged_at_default_threshold']} of {facts['rows']:,} rows, so its "
            f"accuracy ({facts['accuracy_pct']}%) is below the majority baseline's "
            f"({facts['majority_baseline_accuracy_pct']}%)."
        ),
        "It is over-confident at the top of its ranking: 44.3% predicted, 20.0% realised.",
        "No untouched holdout exists; every figure is 5-fold out-of-fold over all rows.",
        (
            "Two columns are excluded by the timing contract. Adding them lifts PR-AUC to "
            "0.235 — the only change in this project that clears the noise, which is why "
            "they stay out."
        ),
        "No causal claim is supported. The response curves describe the model, not the world.",
    ]


def model_card(df: pd.DataFrame | None = None, facts: dict[str, object] | None = None) -> str:
    """Render the card as plain text.

    Args:
        df: Frame to compute the facts from, if ``facts`` is not supplied.
        facts: Pre-computed facts, to avoid refitting.
    """
    values = facts if facts is not None else card_facts(df)
    rule = "=" * CARD_WIDTH
    lines = [
        rule,
        "MODEL CARD - startup-outcomes",
        rule,
        f"Predicts        : {values['predicts']}",
        f"Task            : {values['task']}",
        f"Algorithm       : {values['algorithm']}",
        f"Trained on      : {values['rows']:,} rows x {values['features']} features",
        f"                  ({values['rows_dropped_for_leaky_stage']} rows dropped: "
        f"Funding_Stage == 'IPO' determines the target)",
        f"Evaluation      : {values['evaluation']}",
        f"Seed            : {values['seed']}",
        "",
        f"PR-AUC          : {values['pr_auc']} (+/- {values['pr_auc_fold_sd']} across folds)",
        f"ROC-AUC         : {values['roc_auc']}",
        f"Base rate       : {values['base_rate_pct']}% - the floor PR-AUC must be read against",
        f"Baseline        : {values['baseline_pr_auc']} ({values['baseline_name']})",
        f"Beats baseline  : {'yes' if values['beats_baseline'] else 'NO'}",
        "",
        "Features used   : " + ", ".join(str(name) for name in values["feature_names"]),
        "Excluded        : " + ", ".join(str(name) for name in values["excluded_to_avoid_leakage"]),
        "",
        "KNOWN LIMITS",
    ]
    lines.extend(f"  - {limit}" for limit in known_limits(values))
    lines.append(rule)
    return "\n".join(lines)
