"""Bootstrap intervals and permutation tests.

``arch-modeling.md`` requires an interval on every reported effect, and the reference
workbooks report none — they substitute effect size in SD units plus an ``n``. That is
readable but it cannot answer the one question this dataset keeps raising: is a 3 pp
spread across 25,000 rows distinguishable from a constant base rate? So intervals are
computed here and quoted everywhere.

Two conventions make the numbers reproducible:

* **Every generator is built inside the function that uses it**, on a line naming
  ``RANDOM_SEED``. A module-level generator would make results depend on call order —
  the same defect the pure-transform rule exists to prevent — and it would also hide
  from ``grep -rn "np.random" src/ | grep -v RANDOM_SEED``.
* **Draws are shared, not independent, within one comparison.** ``offset`` defaults to 0
  so two statistics resampled together see the same row draw. That is what makes a
  difference interval a *paired* bootstrap; drawing independently would report the sum of
  two marginal widths and overstate the uncertainty on the difference.

The percentile method is used throughout. At n = 24,467 with these near-symmetric
statistics, BCa buys nothing it would not also cost in machinery to test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.random import Generator

from startup_outcomes.config import BOOTSTRAP_RESAMPLES, RANDOM_SEED

#: Percentile bounds of a 95% interval.
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5

#: Shuffles behind a permutation p-value. Coarser than the bootstrap because a p-value is
#: only ever read against 0.05, and the smallest resolvable value is 1 / PERMUTATIONS.
PERMUTATIONS = 1000

#: A statistic takes an array of row positions and returns one number. Taking positions
#: rather than values lets one draw serve several columns, which is what pairs a
#: difference interval.
Statistic = Callable[[np.ndarray], float]


def generator(offset: int = 0) -> Generator:
    """A freshly seeded random generator.

    Args:
        offset: Added to the project seed. Defaults to 0; pass a non-zero value only for
            deliberate multi-draw replication, never to work around a result you dislike.
    """
    return np.random.default_rng(RANDOM_SEED + offset)


def _summarise(
    point: float,
    draws: np.ndarray,
    *,
    name: str,
    unit: str,
    digits: int,
) -> dict[str, object]:
    """Package a point estimate and its resampled draws as a flat, JSON-clean dict."""
    low, high = np.percentile(draws, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    return {
        "statistic": name,
        f"point_{unit}": round(float(point), digits),
        f"ci_low_{unit}": round(float(low), digits),
        f"ci_high_{unit}": round(float(high), digits),
        f"ci_width_{unit}": round(float(high - low), digits),
        "resamples": int(len(draws)),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def bootstrap_rows(
    row_count: int,
    statistic: Statistic,
    *,
    name: str,
    unit: str = "pp",
    digits: int = 2,
    resamples: int = BOOTSTRAP_RESAMPLES,
    offset: int = 0,
) -> dict[str, object]:
    """Percentile bootstrap of ``statistic`` over a resampled row index.

    The loop is over draws, not over rows: each ``statistic`` call is itself vectorized,
    so this does not reintroduce row-wise iteration. The index is drawn per iteration
    rather than pre-allocated as a matrix — at 1,000 x 24,467 that matrix would be 196 MB
    for no benefit.

    Args:
        row_count: Number of rows the statistic may address.
        statistic: Called with an array of row positions; returns one number.
        name: Identifies the quantity in the returned dict and in test failures.
        unit: Suffix for the value keys, matching the ``audit.py`` naming convention
            (``pp`` for percentage points, ``auc`` for a metric difference).
        digits: Rounding applied to every returned value.
        resamples: Draw count; defaults to the project-wide constant.
        offset: Seed offset, forwarded to :func:`generator`.
    """
    rng = generator(offset)
    positions = np.arange(row_count)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draws[index] = statistic(rng.integers(0, row_count, size=row_count))
    return _summarise(statistic(positions), draws, name=name, unit=unit, digits=digits)


def _rate_by_code(codes: np.ndarray, flags: np.ndarray, level_count: int) -> np.ndarray:
    """Percentage of ``flags`` that are true within each integer group code.

    ``bincount`` rather than a groupby: this runs once per bootstrap draw, where a pandas
    groupby costs roughly twenty times as much.

    ``factorize`` codes a missing group as ``-1``. Those rows are dropped: a blank
    ``AI_Adoption_Level`` is the absence of a level, not a level of its own, so it belongs
    in neither side of a between-level comparison.
    """
    present = codes >= 0
    codes, flags = codes[present], flags[present]
    totals = np.bincount(codes, minlength=level_count).astype(float)
    hits = np.bincount(codes, weights=flags, minlength=level_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, hits / totals * 100, np.nan)


def rate_gap_interval(
    groups: pd.Series,
    flags: pd.Series,
    *,
    high: str,
    low: str,
    name: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Interval on the gap in true-rate between two levels of a categorical column.

    Args:
        groups: The categorical column.
        flags: Boolean outcome, aligned with ``groups``.
        high: Level whose rate is expected to be larger; it is the left side of the gap.
        low: Level subtracted from it.
        name: Identifies the quantity.
        resamples: Draw count.
    """
    codes, levels = pd.factorize(groups, use_na_sentinel=True)
    level_list = list(levels)
    for level in (high, low):
        if level not in level_list:
            raise ValueError(f"Level {level!r} absent from {groups.name}: {level_list}")
    high_code, low_code = level_list.index(high), level_list.index(low)
    flag_values = flags.to_numpy(dtype=float)
    level_count = len(level_list)

    def gap(rows: np.ndarray) -> float:
        rates = _rate_by_code(codes[rows], flag_values[rows], level_count)
        return float(rates[high_code] - rates[low_code])

    return bootstrap_rows(len(groups), gap, name=name, resamples=resamples)


def rate_difference_interval(
    mask: pd.Series,
    flags: pd.Series,
    *,
    name: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Interval on the true-rate difference between the masked rows and the rest."""
    mask_values = mask.to_numpy(dtype=bool)
    flag_values = flags.to_numpy(dtype=float)

    def difference(rows: np.ndarray) -> float:
        selected, outcome = mask_values[rows], flag_values[rows]
        if selected.all() or not selected.any():
            return float("nan")
        return float(outcome[selected].mean() - outcome[~selected].mean()) * 100

    return bootstrap_rows(len(mask), difference, name=name, resamples=resamples)


def paired_metric_difference(
    y_true: pd.Series | np.ndarray,
    scores_left: pd.Series | np.ndarray,
    scores_right: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    name: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Interval on ``metric(left) - metric(right)``, both scored on the same draw.

    Pairing is the point: two ranking metrics computed on independent draws would produce
    an interval roughly twice as wide, and the comparison this project turns on — a model
    against a one-line rule — lives well inside that difference.

    A draw whose rows are all one class cannot be scored; such draws are dropped, and at
    a 14% positive rate over 24,467 rows they do not occur in practice.
    """
    truth = np.asarray(y_true, dtype=int)
    left = np.asarray(scores_left, dtype=float)
    right = np.asarray(scores_right, dtype=float)

    def difference(rows: np.ndarray) -> float:
        labels = truth[rows]
        if labels.min() == labels.max():
            return float("nan")
        return float(metric(labels, left[rows]) - metric(labels, right[rows]))

    return bootstrap_rows(
        len(truth), difference, name=name, unit="auc", digits=4, resamples=resamples
    )


def permutation_spread(
    groups: pd.Series,
    flags: pd.Series,
    *,
    name: str,
    permutations: int = PERMUTATIONS,
    offset: int = 0,
) -> dict[str, object]:
    """Whether the spread of true-rate across a column's levels exceeds chance.

    This is the honest test of a "no signal" claim. A 4.82 pp closed-rate spread across
    20 domains sounds small but means nothing until compared with the spread that
    shuffling the outcome produces at the same group sizes — which is what this returns.

    Args:
        groups: The categorical column whose levels are compared.
        flags: Boolean outcome, aligned with ``groups``.
        name: Identifies the quantity.
        permutations: Shuffle count; the smallest resolvable p-value is its reciprocal.
        offset: Seed offset, forwarded to :func:`generator`.
    """
    codes, levels = pd.factorize(groups, use_na_sentinel=True)
    level_count = len(levels)
    # Drop rows with no group before shuffling, so the null is built over the same rows
    # the observed statistic used.
    present = codes >= 0
    codes = codes[present]
    flag_values = flags.to_numpy(dtype=float)[present]

    def spread(values: np.ndarray) -> float:
        rates = _rate_by_code(codes, values, level_count)
        return float(np.nanmax(rates) - np.nanmin(rates))

    rng = generator(offset)
    observed = spread(flag_values)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        null[index] = spread(rng.permutation(flag_values))

    # +1 in both terms so a p-value is never exactly zero: with 1,000 shuffles the
    # evidence cannot distinguish "rare" from "impossible".
    p_value = float((np.sum(null >= observed) + 1) / (permutations + 1))
    return {
        "statistic": name,
        "levels": int(level_count),
        "observed_pp": round(observed, 2),
        "null_median_pp": round(float(np.median(null)), 2),
        "null_p95_pp": round(float(np.percentile(null, 95)), 2),
        "p_value": round(p_value, 4),
        "permutations": int(permutations),
        "exceeds_null_p95": bool(observed > np.percentile(null, 95)),
    }
