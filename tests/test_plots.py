"""Tests for the plotting module.

Headless, with no backend configuration, because ``plots.py`` never touches pyplot. The
substantive assertion is that no plot function computes a statistic — that is what keeps
an unpinned number from reaching a chart.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from startup_outcomes import descriptive, plots


def test_no_plot_function_computes_a_statistic() -> None:
    """A chart that calculates its own group means is an unpinned number with a picture.

    ``polyfit`` is the one permitted exception — a fitted line is a drawing instruction,
    not a reported number, and nothing quotes its coefficients.
    """
    source = inspect.getsource(plots)
    for forbidden in (".groupby(", ".mean()", ".median()", ".corr(", ".value_counts("):
        assert forbidden not in source, forbidden


def test_the_module_never_imports_pyplot() -> None:
    import sys

    assert "matplotlib.pyplot" not in sys.modules


def test_distribution_returns_a_figure(raw: pd.DataFrame) -> None:
    shape = descriptive.distribution_shape(raw, "Runway_Months_2024")
    figure = plots.distribution(
        raw["Runway_Months_2024"],
        mean=float(shape["mean"]),
        median=float(shape["median"]),
        title="runway",
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "runway"


def test_the_descriptive_grid_has_six_panels(raw: pd.DataFrame) -> None:
    summary = descriptive.group_summary(raw, "Runway_Months_2024", "Investor_Tier")
    shape = descriptive.distribution_shape(raw, "Runway_Months_2024")
    figure = plots.descriptive_grid(
        values=raw["Runway_Months_2024"],
        mean=float(shape["mean"]),
        median=float(shape["median"]),
        summary=summary,
        frequencies=raw["Investor_Tier"].value_counts(),
        correlations=raw[["Runway_Months_2024", "Peak_Headcount_2023"]].corr(),
        scatter_x=raw["Peak_Headcount_2023"],
        scatter_y=raw["Runway_Months_2024"],
    )
    assert len(figure.axes) == 6


def test_the_lift_chart_draws_the_random_line() -> None:
    """A lift curve without its reference line cannot be read."""
    curve = pd.DataFrame({"lift": [1.7, 1.6, 1.4]}, index=[10.0, 20.0, 30.0])
    figure = plots.lift_curves({"model": curve, "runway rule": curve})
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert "random" in labels


def test_the_precision_recall_chart_draws_the_prevalence_line() -> None:
    figure = plots.precision_recall_curve(
        np.linspace(0, 1, 10), np.linspace(0.3, 0.1, 10), base_rate=0.1411
    )
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("base rate" in label for label in labels)


def test_the_silhouette_panel_draws_both_rule_of_thumb_lines() -> None:
    scan = pd.DataFrame({"inertia": [90.0, 70.0], "silhouette": [0.42, 0.34]}, index=[2, 3])
    figure = plots.elbow_and_silhouette(scan)
    assert len(figure.axes) == 2
    labels = [text.get_text() for text in figure.axes[1].get_legend().get_texts()]
    assert "clear structure" in labels
    assert "weak structure" in labels


def test_the_cost_curve_marks_the_cheapest_and_the_default() -> None:
    costs = pd.DataFrame({"total_cost_units": [20659.0, 31610.0]}, index=[0.05, 0.25])
    figure = plots.threshold_cost_curve(costs, cheapest_threshold=0.05, flag_everyone_cost=21014.0)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("cheapest" in label for label in labels)
    assert any("default" in label for label in labels)
    assert "flag everyone" in labels


def test_the_response_curve_is_labelled_as_model_behaviour() -> None:
    curve = pd.DataFrame({"predicted_closed_pct": [23.4, 11.7]}, index=[2.0, 30.0])
    curve.index.name = "Runway_Months_2024"
    figure = plots.response_curve(curve, base_rate_pct=14.11)
    assert "not a causal effect" in figure.axes[0].get_title()


def test_the_reversal_plot_draws_one_line_per_subgroup() -> None:
    table = pd.DataFrame(
        {"None": [17.9, 13.8], "AI-Native": [12.8, 14.3]},
        index=pd.Index(["Accelerator", "Tier 2 VC"], name="Investor_Tier"),
    )
    table.columns.name = "AI_Adoption_Level"
    figure = plots.reversal_plot(table)
    assert len(figure.axes[0].lines) == 2
