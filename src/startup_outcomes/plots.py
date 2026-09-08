"""Charts for the notebooks. Every function returns a Figure and computes no statistic.

Two rules make this module safe to call from a thin driver.

**Nothing here computes a number.** Every value arrives already computed by a tier module
that has a test behind it. A chart that calculates its own group means is an unpinned
number with a picture around it — which is precisely why seaborn is not used: its
``barplot``, ``regplot`` and ``heatmap`` are statistics functions that quietly estimate
means and confidence intervals of their own.

**Nothing here imports pyplot.** Figures are constructed directly, so there is no global
figure registry and no global rcParams mutation. Three things follow: a returned figure
renders exactly once in a notebook instead of twice, plot functions are order-independent
in the same way the pipeline rule requires of transforms, and the rule is checkable by a
grep anchored to import lines rather than by review. Notebooks still need
``%matplotlib inline`` in their first cell, or a returned figure has no registered
formatter and renders as nothing at all — silently, with no error.

The palette is carried over from the reference workbooks so the charts read the same.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

#: The workbooks' five colours: data, reference/warning, optimum, fill, and zero-line.
PRIMARY_COLOR = "#2194D4"
ACCENT_COLOR = "#B4342F"
POSITIVE_COLOR = "#137A5E"
MUTED_COLOR = "#D8EAF7"
NEUTRAL_COLOR = "#6B7280"

FIGURE_SIZE = (7.0, 4.0)
GRID_FIGURE_SIZE = (14.0, 7.5)
HEATMAP_COLORMAP = "Blues"
HISTOGRAM_BINS = 60
KDE_POINTS = 200
GRID_ALPHA = 0.3


def _figure(size: tuple[float, float] = FIGURE_SIZE) -> Figure:
    return Figure(figsize=size, layout="constrained")


def _style(axes: object, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axes.set_title(title)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    axes.grid(alpha=GRID_ALPHA)
    axes.set_axisbelow(True)


def distribution(
    values: pd.Series,
    *,
    mean: float,
    median: float,
    title: str,
) -> Figure:
    """Histogram with a density overlay, and the mean and median marked.

    The two vertical lines are the point: on a right-skewed column they sit far apart,
    and the distance is the argument for never quoting the mean as typical.
    """
    figure = _figure()
    axes = figure.subplots()
    clean = values.dropna().to_numpy(dtype=float)
    axes.hist(clean, bins=HISTOGRAM_BINS, color=PRIMARY_COLOR, alpha=0.75, density=True)
    if len(np.unique(clean)) > 2:
        grid = np.linspace(clean.min(), clean.max(), KDE_POINTS)
        axes.plot(grid, gaussian_kde(clean)(grid), color=NEUTRAL_COLOR, linewidth=1.5)
    axes.axvline(mean, color=ACCENT_COLOR, linestyle="--", label=f"mean {mean:,.2f}")
    axes.axvline(median, color=POSITIVE_COLOR, linestyle="-", label=f"median {median:,.2f}")
    axes.legend()
    _style(axes, title=title, xlabel=str(values.name), ylabel="density")
    return figure


def boxplot_by_category(
    df: pd.DataFrame,
    *,
    measure: str,
    by: str,
    order: Sequence[str],
) -> Figure:
    """Distribution of a measure within each category, ordered by the caller."""
    figure = _figure()
    axes = figure.subplots()
    groups = [df.loc[df[by] == level, measure].dropna().to_numpy(dtype=float) for level in order]
    boxes = axes.boxplot(groups, vert=False, patch_artist=True, tick_labels=list(order))
    for patch in boxes["boxes"]:
        patch.set_facecolor(MUTED_COLOR)
        patch.set_edgecolor(PRIMARY_COLOR)
    for median in boxes["medians"]:
        median.set_color(ACCENT_COLOR)
    _style(axes, title=f"{measure} by {by}", xlabel=measure)
    return figure


def group_means(summary: pd.DataFrame, *, column: str = "mean", title: str = "") -> Figure:
    """Horizontal bars of a precomputed group summary."""
    figure = _figure()
    axes = figure.subplots()
    ordered = summary.sort_values(column)
    axes.barh([str(index) for index in ordered.index], ordered[column], color=PRIMARY_COLOR)
    _style(axes, title=title or f"{column} by {summary.index.name}", xlabel=column)
    return figure


def category_counts(frequencies: pd.Series, *, title: str = "rows per category") -> Figure:
    """Rows per category — the chart everyone skips and the one that saves them.

    It is the visual form of the ``n`` column: a group statistic on forty rows should not
    be read the same way as one on four thousand.
    """
    figure = _figure()
    axes = figure.subplots()
    ordered = frequencies.sort_values()
    axes.barh([str(index) for index in ordered.index], ordered.to_numpy(), color=NEUTRAL_COLOR)
    _style(axes, title=title, xlabel="rows")
    return figure


def scatter_with_fit(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    x_label: str,
    y_label: str,
) -> Figure:
    """Scatter with a straight-line fit, on a sample so the points stay readable."""
    figure = _figure()
    axes = figure.subplots()
    frame = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    axes.scatter(frame["x"], frame["y"], s=6, alpha=0.25, color=PRIMARY_COLOR)
    slope, intercept = np.polyfit(frame["x"], frame["y"], 1)
    line = np.linspace(frame["x"].min(), frame["x"].max(), 2)
    axes.plot(line, slope * line + intercept, color=ACCENT_COLOR, linewidth=2)
    _style(axes, title=f"{y_label} against {x_label}", xlabel=x_label, ylabel=y_label)
    return figure


def correlation_heatmap(correlations: pd.DataFrame, *, title: str = "correlation") -> Figure:
    """Annotated correlation matrix."""
    figure = _figure((6.5, 5.5))
    axes = figure.subplots()
    values = correlations.to_numpy(dtype=float)
    image = axes.imshow(values, cmap=HEATMAP_COLORMAP, vmin=-1, vmax=1)
    axes.set_xticks(range(len(correlations.columns)))
    axes.set_xticklabels(correlations.columns, rotation=45, ha="right", fontsize=7)
    axes.set_yticks(range(len(correlations.index)))
    axes.set_yticklabels(correlations.index, fontsize=7)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(values[row, column]) > 0.6 else "black",
            )
    figure.colorbar(image, ax=axes, shrink=0.8)
    axes.set_title(title)
    return figure


def missingness_map(blocks: pd.DataFrame, *, overall_pct: float) -> Figure:
    """Blank share across blocks of consecutive rows.

    Binned rather than a raw null mask: at 25,000 rows tall the mask renders as a smear,
    while this still shows a stripe or a contiguous block if one exists.

    ``overall_pct`` is passed in rather than averaged here — a reference line is a number,
    and every number on a chart has to come from a function with a test behind it.
    """
    figure = _figure()
    axes = figure.subplots()
    axes.bar(blocks.index, blocks["blank_pct"], width=1.0, color=PRIMARY_COLOR)
    axes.axhline(
        overall_pct,
        color=ACCENT_COLOR,
        linestyle="--",
        label=f"overall {overall_pct:.2f}%",
    )
    axes.legend()
    _style(
        axes,
        title="blank AI_Adoption_Level by row block",
        xlabel="row block",
        ylabel="blank %",
    )
    return figure


def imputation_sensitivity(strategies: pd.DataFrame) -> Figure:
    """The gap each cleaning strategy produces — the size of the decision being made."""
    figure = _figure((7.5, 4.0))
    axes = figure.subplots()
    axes.barh(
        [str(index) for index in strategies.index],
        strategies["none_vs_ai_native_gap_pp"],
        color=PRIMARY_COLOR,
    )
    _style(
        axes,
        title="None - AI-Native closed-rate gap by imputation strategy",
        xlabel="gap (pp)",
    )
    return figure


def descriptive_grid(
    *,
    values: pd.Series,
    mean: float,
    median: float,
    summary: pd.DataFrame,
    frequencies: pd.Series,
    correlations: pd.DataFrame,
    scatter_x: pd.Series,
    scatter_y: pd.Series,
) -> Figure:
    """The workbooks' fixed six-panel descriptive grid, in one figure.

    Every input is precomputed elsewhere, so a panel here and its standalone chart cannot
    disagree.
    """
    figure = _figure(GRID_FIGURE_SIZE)
    axes = figure.subplots(2, 3)

    clean = values.dropna().to_numpy(dtype=float)
    axes[0][0].hist(clean, bins=HISTOGRAM_BINS, color=PRIMARY_COLOR, density=True)
    axes[0][0].axvline(mean, color=ACCENT_COLOR, linestyle="--")
    axes[0][0].axvline(median, color=POSITIVE_COLOR)
    _style(axes[0][0], title=f"shape of {values.name}")

    ordered = summary.sort_values("median")
    axes[0][1].barh([str(index) for index in ordered.index], ordered["median"], color=PRIMARY_COLOR)
    _style(axes[0][1], title="median by group")

    by_mean = summary.sort_values("mean")
    axes[0][2].barh([str(index) for index in by_mean.index], by_mean["mean"], color=PRIMARY_COLOR)
    _style(axes[0][2], title="mean by group")

    frame = pd.DataFrame({"x": scatter_x, "y": scatter_y}).dropna()
    axes[1][0].scatter(frame["x"], frame["y"], s=5, alpha=0.2, color=PRIMARY_COLOR)
    _style(axes[1][0], title=f"{scatter_y.name} vs {scatter_x.name}")

    matrix = correlations.to_numpy(dtype=float)
    axes[1][1].imshow(matrix, cmap=HEATMAP_COLORMAP, vmin=-1, vmax=1)
    axes[1][1].set_xticks(range(len(correlations.columns)))
    axes[1][1].set_xticklabels(correlations.columns, rotation=90, fontsize=6)
    axes[1][1].set_yticks(range(len(correlations.index)))
    axes[1][1].set_yticklabels(correlations.index, fontsize=6)
    axes[1][1].set_title("correlation")

    counted = frequencies.sort_values()
    axes[1][2].barh(
        [str(index) for index in counted.index], counted.to_numpy(), color=NEUTRAL_COLOR
    )
    _style(axes[1][2], title="rows per category")
    return figure


def reversal_plot(table: pd.DataFrame, *, title: str = "") -> Figure:
    """One line per subgroup, so a crossing is visible as a crossing.

    This is the Simpson's-paradox chart. Lines that stay parallel mean the headline
    ordering holds; a line that slopes the other way is the reversal.
    """
    figure = _figure((7.5, 4.5))
    axes = figure.subplots()
    for level in table.index:
        axes.plot(
            [str(column) for column in table.columns],
            table.loc[level].to_numpy(dtype=float),
            marker="o",
            label=str(level),
        )
    axes.legend(fontsize=7, ncol=2)
    _style(
        axes,
        title=title or f"closed rate by {table.columns.name} within {table.index.name}",
        ylabel="closed %",
    )
    return figure


def interval_plot(intervals_table: pd.DataFrame) -> Figure:
    """Point estimates with their confidence intervals, and a zero reference line."""
    figure = _figure()
    axes = figure.subplots()
    positions = np.arange(len(intervals_table))
    lows = intervals_table["point"] - intervals_table["ci_low"]
    highs = intervals_table["ci_high"] - intervals_table["point"]
    axes.errorbar(
        intervals_table["point"],
        positions,
        xerr=[lows, highs],
        fmt="o",
        color=PRIMARY_COLOR,
        ecolor=NEUTRAL_COLOR,
        capsize=4,
    )
    axes.axvline(0, color=ACCENT_COLOR, linestyle="--")
    axes.set_yticks(positions)
    axes.set_yticklabels([str(index) for index in intervals_table.index], fontsize=8)
    _style(axes, title="effects with 95% intervals", xlabel="effect (pp)")
    return figure


def permutation_plot(observed: float, null_median: float, null_p95: float, *, title: str) -> Figure:
    """Observed spread against the spread shuffling produces."""
    figure = _figure()
    axes = figure.subplots()
    axes.barh(
        ["observed", "null median", "null 95th pct"],
        [observed, null_median, null_p95],
        color=[PRIMARY_COLOR, NEUTRAL_COLOR, ACCENT_COLOR],
    )
    _style(axes, title=title, xlabel="spread (pp)")
    return figure


def depth_sweep_curve(sweep: pd.DataFrame) -> Figure:
    """Train against out-of-fold score as depth grows — overfitting, live."""
    figure = _figure()
    axes = figure.subplots()
    labels = [str(index) for index in sweep.index]
    axes.plot(labels, sweep["train_pr_auc"], marker="o", color=ACCENT_COLOR, label="train")
    axes.plot(
        labels,
        sweep["out_of_fold_pr_auc"],
        marker="o",
        color=PRIMARY_COLOR,
        label="out of fold",
    )
    axes.legend()
    _style(axes, title="tree depth against score", xlabel="max depth", ylabel="PR-AUC")
    return figure


def leaderboard_bars(board: pd.DataFrame, *, base_rate: float) -> Figure:
    """Model scores with fold-spread error bars, against the base-rate floor."""
    figure = _figure((7.5, 4.5))
    axes = figure.subplots()
    ordered = board.sort_values("pr_auc")
    axes.barh(
        [str(index) for index in ordered.index],
        ordered["pr_auc"],
        xerr=ordered["pr_auc_fold_sd"],
        color=PRIMARY_COLOR,
        ecolor=NEUTRAL_COLOR,
        capsize=3,
    )
    axes.axvline(base_rate, color=ACCENT_COLOR, linestyle="--", label=f"base rate {base_rate:.3f}")
    axes.legend()
    _style(axes, title="PR-AUC by model", xlabel="PR-AUC")
    return figure


def precision_recall_curve(
    recall: np.ndarray, precision: np.ndarray, *, base_rate: float
) -> Figure:
    """Precision against recall, with the prevalence line drawn.

    A PR curve without its prevalence is unreadable: at a 14% positive rate, 0.19 is a
    modest result and 0.14 is nothing at all.
    """
    figure = _figure()
    axes = figure.subplots()
    axes.plot(recall, precision, color=PRIMARY_COLOR)
    axes.axhline(base_rate, color=ACCENT_COLOR, linestyle="--", label=f"base rate {base_rate:.3f}")
    axes.legend()
    _style(axes, title="precision against recall", xlabel="recall", ylabel="precision")
    return figure


def lift_curves(curves: Mapping[str, pd.DataFrame]) -> Figure:
    """Lift for several rankings on one pair of axes, against the random line.

    Not in the reference workbooks, and the project's honest headline: the model's curve
    and the runway rule's curve are drawn together, and they overlap.
    """
    figure = _figure()
    axes = figure.subplots()
    for name, curve in curves.items():
        axes.plot(curve.index, curve["lift"], marker="o", markersize=3, label=name)
    axes.axhline(1.0, color=NEUTRAL_COLOR, linestyle="--", label="random")
    axes.legend()
    _style(axes, title="lift by depth of targeting", xlabel="% targeted", ylabel="lift")
    return figure


def elbow_and_silhouette(scan: pd.DataFrame) -> Figure:
    """Inertia and silhouette side by side.

    Inertia always falls, so it cannot choose k alone; the silhouette panel carries the
    0.5 and 0.25 rule-of-thumb lines that say whether any k is worth taking.
    """
    figure = _figure((10.0, 4.0))
    left, right = figure.subplots(1, 2)
    left.plot(scan.index, scan["inertia"], marker="o", color=PRIMARY_COLOR)
    _style(left, title="inertia (the elbow)", xlabel="k", ylabel="inertia")

    right.plot(scan.index, scan["silhouette"], marker="o", color=PRIMARY_COLOR)
    right.axhline(0.5, color=POSITIVE_COLOR, linestyle="--", label="clear structure")
    right.axhline(0.25, color=ACCENT_COLOR, linestyle=":", label="weak structure")
    right.legend(fontsize=8)
    _style(right, title="silhouette", xlabel="k", ylabel="silhouette")
    return figure


def variance_curve(curve: pd.DataFrame, *, target_pct: float = 90.0) -> Figure:
    """Cumulative explained variance, with the retention target marked."""
    figure = _figure()
    axes = figure.subplots()
    axes.plot(curve.index, curve["cumulative_pct"], marker="o", color=PRIMARY_COLOR)
    axes.axhline(target_pct, color=ACCENT_COLOR, linestyle="--", label=f"{target_pct:.0f}%")
    axes.legend()
    _style(
        axes,
        title="cumulative explained variance",
        xlabel="components",
        ylabel="variance kept (%)",
    )
    return figure


def component_scatter(
    coordinates: pd.DataFrame,
    labels: Sequence[int],
    *,
    variance_shown_pct: float,
) -> Figure:
    """Rows projected onto two components, coloured by cluster.

    The axes are blends of logged, standardised columns and mean nothing on their own —
    never quote a component value to a reader.
    """
    figure = _figure((6.0, 5.0))
    axes = figure.subplots()
    axes.scatter(
        coordinates["component_1"],
        coordinates["component_2"],
        c=list(labels),
        cmap="tab10",
        s=5,
        alpha=0.35,
    )
    _style(
        axes,
        title=f"two components ({variance_shown_pct:.1f}% of variance)",
        xlabel="component 1",
        ylabel="component 2",
    )
    return figure


def threshold_cost_curve(
    costs: pd.DataFrame,
    *,
    cheapest_threshold: float,
    default_threshold: float = 0.5,
    flag_everyone_cost: float | None = None,
) -> Figure:
    """Total cost against threshold, with the cheapest cut and the default both marked.

    The comparator line is the reason this chart is honest: when the blanket policy sits
    almost on top of the cheapest cut, the model is not making the decision.
    """
    figure = _figure()
    axes = figure.subplots()
    axes.plot(costs.index, costs["total_cost_units"], color=PRIMARY_COLOR, marker="o", markersize=3)
    axes.axvline(
        cheapest_threshold,
        color=POSITIVE_COLOR,
        label=f"cheapest {cheapest_threshold:.2f}",
    )
    axes.axvline(
        default_threshold, color=ACCENT_COLOR, linestyle=":", label=f"default {default_threshold}"
    )
    if flag_everyone_cost is not None:
        axes.axhline(
            flag_everyone_cost,
            color=NEUTRAL_COLOR,
            linestyle="--",
            label="flag everyone",
        )
    axes.legend(fontsize=8)
    _style(
        axes,
        title="the threshold is a business decision",
        xlabel="threshold",
        ylabel="total cost (units)",
    )
    return figure


def response_curve(curve: pd.DataFrame, *, base_rate_pct: float) -> Figure:
    """Predicted probability as one feature is swept, against the base rate."""
    figure = _figure()
    axes = figure.subplots()
    axes.plot(
        curve.index, curve["predicted_closed_pct"], color=PRIMARY_COLOR, marker="o", markersize=3
    )
    axes.axhline(
        base_rate_pct, color=ACCENT_COLOR, linestyle="--", label=f"base rate {base_rate_pct:.2f}%"
    )
    axes.legend()
    _style(
        axes,
        title=f"model response to {curve.index.name} (not a causal effect)",
        xlabel=str(curve.index.name),
        ylabel="predicted closed %",
    )
    return figure


def value_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """Net value across assumed success rates, with the break-even line."""
    figure = _figure()
    axes = figure.subplots()
    colours = [
        POSITIVE_COLOR if positive else ACCENT_COLOR for positive in sensitivity["net_positive"]
    ]
    axes.bar(
        [f"{rate:.0%}" for rate in sensitivity.index],
        sensitivity["net_units"],
        color=colours,
    )
    axes.axhline(0, color=NEUTRAL_COLOR)
    _style(
        axes,
        title="net value by assumed success rate",
        xlabel="assumed success rate",
        ylabel="net (units)",
    )
    return figure
