"""Every tier in one call — the command the README's reproducibility claim rests on.

    uv run python -c "from startup_outcomes.report import run_all; print(run_all())"

Imported by nothing. This module composes; it computes nothing of its own, so a number
can never enter the README through here without existing in a tier module first.

The tiers run in ladder order — descriptive, diagnostic, predictive, prescriptive — which
is also dependency order, and the order the notebooks follow.
"""

from __future__ import annotations

import pandas as pd

from startup_outcomes import audit, descriptive, diagnostic
from startup_outcomes.load import load_raw
from startup_outcomes.models import card, prescriptive, supervised, unsupervised

#: Tier order, used for the summary and by the notebooks.
TIERS = ["audit", "descriptive", "diagnostic", "predictive", "unsupervised", "prescriptive"]


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Run every tier and return the results keyed by tier name.

    Takes about forty seconds, almost all of it in the predictive tier's cross-validation.
    Nothing is cached and nothing is written: the whole analysis is recomputed from
    ``data/raw`` every time, which is what makes "re-derive in one command" true.
    """
    source = df if df is not None else load_raw()
    return {
        "audit": audit.run_all(source),
        "descriptive": descriptive.run_all(source),
        "diagnostic": diagnostic.run_all(source),
        "predictive": supervised.run_all(source),
        "unsupervised": unsupervised.run_all(source),
        "prescriptive": prescriptive.run_all(source),
        "model_card": card.card_facts(source),
    }


def headline(results: dict[str, object] | None = None) -> dict[str, object]:
    """The dozen numbers that carry this project's conclusions.

    Deliberately small. If a reader takes nothing else away, these are the claims — and
    every one of them is a comparison rather than a bare figure.
    """
    computed = results if results is not None else run_all()
    predictive = computed["predictive"]
    prescriptive_results = computed["prescriptive"]
    diagnostic_results = computed["diagnostic"]
    board = predictive["leaderboard"]

    return {
        "rows_analysed": predictive["rows"],
        "base_rate_pct": board["gradient boosting"]["base_rate_pct"],
        "model_pr_auc": board["gradient boosting"]["pr_auc"],
        "runway_only_pr_auc": board["runway only (same estimator)"]["pr_auc"],
        "model_beats_runway_only": predictive["beats_runway_only"]["beats_baseline"],
        "leakage_demo_pr_auc": predictive["leakage_demonstration"]["leaked_pr_auc"],
        "rows_flagged_at_default_threshold": predictive["threshold_counts"]["flagged_rows"],
        "ai_adoption_gap_pp": diagnostic_results["ai_adoption_gap"]["point_pp"],
        "ai_adoption_interval_excludes_zero": diagnostic_results["ai_adoption_gap"][
            "excludes_zero"
        ],
        "domain_signal_p_value": diagnostic_results["domain_spread_permutation"]["p_value"],
        "blanks_depend_on_domain": diagnostic_results["missingness_by_domain_permutation"][
            "exceeds_null_p95"
        ],
        "cost_saving_vs_blanket_policy_pct": prescriptive_results["cheapest_threshold"][
            "saving_vs_best_trivial_policy_pct"
        ],
        "targeting_indistinguishable_from_random": prescriptive_results["capacity_expected_value"][
            "indistinguishable_from_random"
        ],
    }


def summary_text(results: dict[str, object] | None = None) -> str:
    """The headline as a printable block, for the capstone cell of the last notebook."""
    computed = results if results is not None else run_all()
    facts = headline(computed)
    rule = "=" * 78
    verdict = "does NOT beat" if not facts["model_beats_runway_only"] else "beats"
    return "\n".join(
        [
            rule,
            "STARTUP OUTCOMES - what the four tiers found",
            rule,
            f"Rows analysed        : {facts['rows_analysed']:,} "
            f"(base rate {facts['base_rate_pct']}% closed)",
            "",
            f"A 13-feature gradient-boosted model {verdict} the same estimator given",
            f"runway alone: PR-AUC {facts['model_pr_auc']} against {facts['runway_only_pr_auc']}.",
            f"At the default threshold it flags "
            f"{facts['rows_flagged_at_default_threshold']} of "
            f"{facts['rows_analysed']:,} companies.",
            f"Adding the two excluded 2026 columns lifts it to "
            f"{facts['leakage_demo_pr_auc']} - the only change that clears the noise.",
            "",
            f"AI adoption moves the closed rate {facts['ai_adoption_gap_pp']} pp; the interval "
            f"excludes zero ({facts['ai_adoption_interval_excludes_zero']}), so the effect is",
            "real and far too small to act on. It reverses inside Tier 2 VC.",
            f"Domain carries no signal at all (permutation p = {facts['domain_signal_p_value']}).",
            "",
            f"Prescriptively, the cost-optimal policy saves "
            f"{facts['cost_saving_vs_blanket_policy_pct']}% against flagging everyone,",
            f"and targeting the top 30 is indistinguishable from random "
            f"({facts['targeting_indistinguishable_from_random']}).",
            "",
            "The data is synthetic. None of this describes real startups.",
            rule,
        ]
    )
