"""Unit tests for the interval machinery.

These describe the CODE, not the dataset: every input is synthetic and constructed so the
right answer is known in advance. The dataset's own intervals are pinned in
``test_diagnostic.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startup_outcomes import intervals
from startup_outcomes.config import RANDOM_SEED


@pytest.fixture(scope="module")
def two_level_frame() -> pd.DataFrame:
    """4,000 rows where the true rate gap between levels is exactly 20 pp."""
    generator = intervals.generator()
    groups = pd.Series(["high"] * 2000 + ["low"] * 2000, name="group")
    flags = pd.Series(
        np.concatenate(
            [
                generator.random(2000) < 0.40,
                generator.random(2000) < 0.20,
            ]
        ),
        name="flag",
    )
    return pd.DataFrame({"group": groups, "flag": flags})


def test_the_generator_is_seeded_from_the_project_seed() -> None:
    expected = np.random.default_rng(RANDOM_SEED).integers(0, 1_000_000, size=5)
    assert intervals.generator().integers(0, 1_000_000, size=5).tolist() == expected.tolist()


def test_an_offset_changes_the_draw() -> None:
    base = intervals.generator().integers(0, 1_000_000, size=5).tolist()
    shifted = intervals.generator(offset=1).integers(0, 1_000_000, size=5).tolist()
    assert base != shifted


def test_the_same_interval_comes_back_on_a_second_call(two_level_frame: pd.DataFrame) -> None:
    """Guards against a module-level generator being introduced later.

    A shared generator would make every interval depend on how many intervals ran before
    it, which is unreproducible in exactly the way that is hardest to notice.
    """
    first = intervals.rate_gap_interval(
        two_level_frame["group"], two_level_frame["flag"], high="high", low="low", name="gap"
    )
    second = intervals.rate_gap_interval(
        two_level_frame["group"], two_level_frame["flag"], high="high", low="low", name="gap"
    )
    assert first == second


def test_a_known_gap_falls_inside_its_interval(two_level_frame: pd.DataFrame) -> None:
    result = intervals.rate_gap_interval(
        two_level_frame["group"], two_level_frame["flag"], high="high", low="low", name="gap"
    )
    assert result["ci_low_pp"] < 20.0 < result["ci_high_pp"]
    assert result["excludes_zero"] is True


def test_interval_endpoints_bracket_the_point_estimate(two_level_frame: pd.DataFrame) -> None:
    result = intervals.rate_gap_interval(
        two_level_frame["group"], two_level_frame["flag"], high="high", low="low", name="gap"
    )
    assert result["ci_low_pp"] < result["point_pp"] < result["ci_high_pp"]
    assert result["ci_width_pp"] == round(result["ci_high_pp"] - result["ci_low_pp"], 2)


def test_every_returned_value_key_carries_a_unit_suffix(two_level_frame: pd.DataFrame) -> None:
    result = intervals.rate_gap_interval(
        two_level_frame["group"], two_level_frame["flag"], high="high", low="low", name="gap"
    )
    assert set(result) == {
        "statistic",
        "point_pp",
        "ci_low_pp",
        "ci_high_pp",
        "ci_width_pp",
        "resamples",
        "excludes_zero",
    }
    assert result["resamples"] == 1000


def test_a_metric_difference_uses_the_auc_unit_and_four_digits() -> None:
    truth = pd.Series([0, 1] * 500)
    scores = pd.Series(intervals.generator().random(1000))
    result = intervals.paired_metric_difference(
        truth,
        scores,
        scores,
        metric=lambda labels, values: float(np.mean(labels * values)),
        name="identical",
        resamples=100,
    )
    # Identical score vectors must give exactly zero difference on every draw.
    assert result["point_auc"] == 0.0
    assert result["ci_low_auc"] == 0.0
    assert result["ci_high_auc"] == 0.0
    assert result["excludes_zero"] is False


def test_a_rate_difference_interval_recovers_a_planted_step() -> None:
    generator = intervals.generator()
    mask = pd.Series([True] * 1500 + [False] * 1500)
    flags = pd.Series(
        np.concatenate([generator.random(1500) < 0.50, generator.random(1500) < 0.20])
    )
    result = intervals.rate_difference_interval(mask, flags, name="step")
    assert result["ci_low_pp"] < 30.0 < result["ci_high_pp"]
    assert result["excludes_zero"] is True


def test_an_absent_level_is_rejected_rather_than_silently_skipped(
    two_level_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="middle"):
        intervals.rate_gap_interval(
            two_level_frame["group"],
            two_level_frame["flag"],
            high="middle",
            low="low",
            name="gap",
        )


def test_a_permutation_test_finds_nothing_when_there_is_nothing() -> None:
    generator = intervals.generator()
    groups = pd.Series(np.repeat(list("abcde"), 1000))
    flags = pd.Series(generator.random(5000) < 0.3)
    result = intervals.permutation_spread(groups, flags, name="null", permutations=200)
    assert result["p_value"] > 0.05
    assert result["exceeds_null_p95"] is False
    assert result["levels"] == 5


def test_a_permutation_test_finds_a_planted_signal() -> None:
    generator = intervals.generator()
    groups = pd.Series(np.repeat(list("abcde"), 1000))
    rates = np.repeat([0.10, 0.20, 0.30, 0.40, 0.80], 1000)
    flags = pd.Series(generator.random(5000) < rates)
    result = intervals.permutation_spread(groups, flags, name="planted", permutations=200)
    assert result["p_value"] < 0.01
    assert result["exceeds_null_p95"] is True
    assert result["observed_pp"] > result["null_p95_pp"]


def test_a_permutation_p_value_is_never_exactly_zero() -> None:
    """With N shuffles the evidence cannot separate "rare" from "impossible"."""
    generator = intervals.generator()
    groups = pd.Series(np.repeat(list("ab"), 1000))
    flags = pd.Series(np.concatenate([generator.random(1000) < 0.01, np.ones(1000, dtype=bool)]))
    result = intervals.permutation_spread(groups, flags, name="extreme", permutations=100)
    assert result["p_value"] > 0.0
    assert result["p_value"] == round(1 / 101, 4)
