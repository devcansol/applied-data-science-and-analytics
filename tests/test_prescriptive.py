"""Pins the prescriptive findings quoted in the README.

The verdict this tier reaches is negative, and these tests are written so it cannot
quietly turn positive: the comparators, the calibration gap and the interval on the
realised rate are all asserted as claims, not just as digits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startup_outcomes.models import prescriptive


@pytest.fixture(scope="module")
def results(report_results: dict[str, object]) -> dict[str, object]:
    """This tier's slice of the shared report; see tests/conftest.py."""
    return report_results["prescriptive"]


def test_every_prescriptive_output_declares_its_basis(results: dict[str, object]) -> None:
    """A partition test. The disclaimer is a dict key so it survives being copied."""
    assert results["basis"] == prescriptive.BASIS
    for key in ("cheapest_threshold", "capacity_expected_value", "response_to_runway"):
        assert results[key]["basis"] == prescriptive.BASIS, key


def test_no_output_carries_a_currency(results: dict[str, object]) -> None:
    """Costs and values are unitless weights; a currency symbol would invite quotation."""
    for value in results.values():
        if isinstance(value, dict):
            for name in value:
                assert "usd" not in name.lower()
                assert "$" not in name
    money_keys = [name for name in results["capacity_expected_value"] if "units" in name]
    assert len(money_keys) >= 4


def test_the_cost_optimal_policy_is_very_nearly_a_blanket_policy(
    results: dict[str, object],
) -> None:
    """The finding that makes this tier a result rather than fake advice.

    At a 10:1 cost ratio the cheapest threshold is the lowest one on the grid, flagging
    97.1% of companies and saving 1.69% against simply flagging everyone. The model is
    steering almost nothing.
    """
    threshold = results["cheapest_threshold"]
    assert threshold["cheapest_threshold"] == 0.05
    assert threshold["flagged_share_pct"] == 97.1
    assert threshold["saving_vs_best_trivial_policy_pct"] == 1.69
    assert float(threshold["saving_vs_best_trivial_policy_pct"]) < 5.0


def test_the_cheapest_threshold_is_reported_beside_its_trivial_comparators(
    results: dict[str, object],
) -> None:
    """A cheapest cost quoted alone implies the model is making the decision."""
    threshold = results["cheapest_threshold"]
    for key in (
        "cost_flag_everyone_units",
        "cost_flag_nobody_units",
        "cost_at_default_threshold_units",
    ):
        assert key in threshold
    assert float(threshold["cheapest_cost_units"]) < float(threshold["cost_flag_everyone_units"])
    assert float(threshold["cheapest_cost_units"]) < float(threshold["cost_flag_nobody_units"])


def test_the_default_threshold_is_as_bad_as_doing_nothing(
    results: dict[str, object],
) -> None:
    """Because the model almost never crosses 0.5, its default-threshold cost is the
    cost of flagging nobody — nothing about 0.5 is principled."""
    threshold = results["cheapest_threshold"]
    assert float(threshold["cost_at_default_threshold_units"]) >= float(
        threshold["cost_flag_nobody_units"]
    )


def test_the_analytic_indifference_point_explains_the_cheap_threshold(
    results: dict[str, object],
) -> None:
    """cost_fa / (cost_fa + cost_miss) = 1/11. A calibrated model should flag anything
    above 9.1%, and most of this population sits there."""
    assert results["cheapest_threshold"]["analytic_indifference_threshold"] == 0.0909


def test_the_model_overstates_its_own_hit_rate_by_more_than_double(
    results: dict[str, object],
) -> None:
    """The calibration finding the workbooks' formulation would have hidden.

    Using the mean predicted score as the hit rate — as the reference notebook does —
    would claim 44.3% of the targeted group closes. The true labels say 20.0%.
    """
    value = results["capacity_expected_value"]
    assert value["model_predicted_rate_pct"] == 44.26
    assert value["realised_closure_rate_pct"] == 20.0
    assert value["model_overstates_its_own_hit_rate"] is True
    assert float(value["model_predicted_rate_pct"]) > 2 * float(value["realised_closure_rate_pct"])


def test_targeting_thirty_companies_is_indistinguishable_from_random(
    results: dict[str, object],
) -> None:
    """Six events cannot support a targeting claim, and the interval says so.

    The realised rate is 20.0% with a 95% interval of [6.67, 36.67], which contains the
    14.11% base rate. The apparent 1.42x lift is not resolvable at this capacity.
    """
    value = results["capacity_expected_value"]
    assert value["hits_in_targeted_group"] == 6
    assert value["realised_rate_ci_low_pct"] == 6.67
    assert value["realised_rate_ci_high_pct"] == 36.67
    assert value["indistinguishable_from_random"] is True
    assert (
        float(value["realised_rate_ci_low_pct"])
        <= float(value["base_rate_pct"])
        <= float(value["realised_rate_ci_high_pct"])
    )


def test_the_intervention_does_not_pay_at_the_assumed_success_rate(
    results: dict[str, object],
) -> None:
    """Net is negative, and the break-even rate is the number to quote.

    Breaking even needs the intervention to work 55.6% of the time. Whether that is
    plausible is not something this data can inform, which is the honest closing line.
    """
    value = results["capacity_expected_value"]
    assert value["net_units"] == -840.0
    assert value["break_even_success_rate"] == 0.556
    assert float(value["net_units"]) < 0


def test_the_sensitivity_table_states_a_fact_not_a_recommendation(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    _, target = design
    table = prescriptive.value_sensitivity(target, oof_scores)
    assert list(table.columns) == ["expected_return_units", "net_units", "net_positive"]
    # The sign flips between a 40% and a 60% success rate — the margin of safety.
    assert bool(table.loc[0.4, "net_positive"]) is False
    assert bool(table.loc[0.6, "net_positive"]) is True


def test_lift_is_modest_and_falls_as_you_target_more(
    results: dict[str, object],
) -> None:
    assert results["top_decile_lift"] == 1.68
    assert results["top_fifth_lift"] == 1.6
    assert results["top_decile_lift"] > results["top_fifth_lift"]


def test_the_ranked_list_is_scored_out_of_fold(
    analysis: pd.DataFrame, oof_scores: np.ndarray
) -> None:
    """No company is ranked by a model that trained on it."""
    ranked = prescriptive.ranked_actions(analysis, oof_scores, top=15)
    assert len(ranked) == 15
    assert list(ranked.index) == list(range(1, 16))
    assert bool(ranked["score"].is_monotonic_decreasing)
    # Every top-ranked company is short of runway, which is the model restating the rule.
    assert float(ranked["runway_months"].max()) < 6.0


def test_the_targeting_mix_shows_no_meaningful_domain_concentration(
    analysis: pd.DataFrame, oof_scores: np.ndarray
) -> None:
    """Consistent with the diagnostic finding that domain carries no signal."""
    table = prescriptive.targeting_mix(analysis, oof_scores, "Domain")
    assert float(table["over_under_pp"].abs().max()) < 3.0


def test_the_response_curve_rediscovers_the_runway_cliff(
    results: dict[str, object],
) -> None:
    """And it is labelled as model behaviour, not a causal effect."""
    response = results["response_to_runway"]
    assert response["feature"] == "Runway_Months_2024"
    assert response["low_end_pct"] == 23.44
    assert response["high_end_pct"] == 11.7
    assert response["swing_pp"] == 11.74
    assert response["is_model_behaviour_not_causal_effect"] is True


def test_the_cost_sweep_covers_the_whole_threshold_range(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    _, target = design
    sweep = prescriptive.cost_threshold_sweep(target, oof_scores)
    assert len(sweep) == 19
    assert bool(sweep["flagged_rows"].is_monotonic_decreasing)
    assert int(sweep["flagged_rows"].iloc[-1]) == 0


def test_changing_the_cost_ratio_moves_the_answer(
    design: tuple[pd.DataFrame, pd.Series], oof_scores: np.ndarray
) -> None:
    """Prescriptive analytics in one assertion: the threshold is a business input.

    At 10:1 the cheapest cut is the lowest on the grid; at 1:1, where a false alarm costs
    as much as a miss, it is cheapest to flag nobody.
    """
    _, target = design
    expensive_misses = prescriptive.cheapest_threshold(target, oof_scores, cost_miss=10.0)
    equal_costs = prescriptive.cheapest_threshold(target, oof_scores, cost_miss=1.0)
    assert expensive_misses["cheapest_threshold"] == 0.05
    assert equal_costs["cheapest_threshold"] > expensive_misses["cheapest_threshold"]
    assert int(equal_costs["flagged_at_cheapest"]) < int(expensive_misses["flagged_at_cheapest"])
