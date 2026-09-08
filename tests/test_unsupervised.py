"""Pins the unsupervised findings quoted in the README."""

from __future__ import annotations

import pandas as pd
import pytest

from startup_outcomes import features
from startup_outcomes.models import protocol, unsupervised


@pytest.fixture(scope="module")
def matrix(design: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    features_frame, _ = design
    return protocol.log_numeric_matrix(features_frame)


@pytest.fixture(scope="module")
def results(report_results: dict[str, object]) -> dict[str, object]:
    """This tier's slice of the shared report; see tests/conftest.py."""
    return report_results["unsupervised"]


def test_clustering_runs_on_the_continuous_columns_only(matrix: pd.DataFrame) -> None:
    """The correctness requirement, asserted rather than trusted.

    On the 112-column one-hot matrix, k=2 scores a silhouette near 0.6 — "clear
    structure" by the usual rule of thumb, and an artifact of splitting on one sparse
    binary. Restricting to the seven standardised continuous columns is what makes a
    Euclidean distance mean anything here.
    """
    assert matrix.shape == (24_467, 7)


def test_no_clear_structure_exists(results: dict[str, object]) -> None:
    """``best_k`` is computed from the silhouette, never hand-edited into a cell as the
    workbooks' ``K = 3`` is — and it is reported beside the verdict so nobody reads a
    chosen k as evidence that clusters exist."""
    summary = results["clustering_summary"]
    assert summary["k_values_tried"] == 7
    assert summary["best_k"] == 2
    assert summary["best_silhouette"] == 0.4195
    assert summary["structure_verdict"] == "weak structure"
    assert summary["any_clear_structure"] is False


def test_inertia_alone_cannot_choose_k(results: dict[str, object]) -> None:
    """It falls monotonically by construction, which is why the silhouette decides."""
    assert results["clustering_summary"]["inertia_falls_monotonically"] is True


def test_the_only_grouping_found_is_a_size_split(
    analysis: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    """A cluster is useless until you can name it, and the only available name is "big".

    The two clusters differ forty-fold in median funding and thirty-fold in median
    headcount, with the same top domain in both. That is one continuous size gradient cut
    in half, not two kinds of company.
    """
    profile = unsupervised.cluster_profile(analysis, matrix, 2)
    small, large = profile.loc[0], profile.loc[1]
    assert float(small["median_funding"]) == 3.9
    assert float(large["median_funding"]) == 158.25
    assert float(large["median_funding"]) > 30 * float(small["median_funding"])
    assert small["top_domain"] == large["top_domain"] == "Generative AI"


def test_the_clusters_barely_separate_the_outcome(results: dict[str, object]) -> None:
    """3.41 pp between clusters — the same order as the AI-adoption effect, and no basis
    for a segmentation anyone should act on."""
    separation = results["cluster_separation"]
    assert separation["clusters"] == 2
    assert separation["closed_pct_spread"] == 3.41
    assert separation["closed_pct_min"] == 11.9
    assert separation["closed_pct_max"] == 15.31


def test_the_numeric_space_is_essentially_one_dimension(results: dict[str, object]) -> None:
    """Which restates the collinearity finding from a different direction.

    A single component carries 68% of the variance across seven columns, because five of
    them are size proxies correlated at r > 0.92 on logs. There is no rich structure for
    a model to exploit — there is size, and there is runway.
    """
    summary = results["pca_summary"]
    assert summary["columns"] == 7
    assert summary["pc1_explained_pct"] == 68.47
    assert summary["first_two_explained_pct"] == 82.76
    assert summary["components_for_90pct"] == 3


def test_the_variance_curve_is_cumulative_and_complete(matrix: pd.DataFrame) -> None:
    curve = unsupervised.variance_curve(matrix)
    assert len(curve) == 7
    assert round(float(curve["cumulative_pct"].iloc[-1]), 1) == 100.0
    assert bool(curve["cumulative_pct"].is_monotonic_increasing)


def test_the_projection_keeps_two_components(matrix: pd.DataFrame) -> None:
    coordinates = unsupervised.pca_coordinates(matrix)
    assert list(coordinates.columns) == ["component_1", "component_2"]
    assert len(coordinates) == len(matrix)


def test_the_silhouette_is_sampled_reproducibly(matrix: pd.DataFrame) -> None:
    """The full distance matrix would be 4.5 GB, so the estimate must still be stable."""
    assert unsupervised.SILHOUETTE_SAMPLE == 5000
    first = unsupervised.cluster_scan(matrix.head(2000))
    second = unsupervised.cluster_scan(matrix.head(2000))
    assert first.equals(second)


def test_the_design_matrix_feeding_clustering_is_leakage_controlled(
    analysis: pd.DataFrame,
) -> None:
    design_frame, _ = features.design_matrix(analysis)
    matrix = protocol.log_numeric_matrix(design_frame)
    from startup_outcomes import config

    config.assert_no_leakage(matrix.columns)
