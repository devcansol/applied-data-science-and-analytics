"""Tier 3b — unsupervised: is there structure nobody labelled?

The workbooks' framing is "not everything has a target", and the honest answer here turns
out to be that there is nothing to find. That is worth establishing properly rather than
asserting, because a clustering can always be *run* and will always return clusters.

One methodological point dominates this module. Clustering the one-hot matrix scores a
silhouette near 0.6 at k=2 — comfortably in the "clear structure" band — and it is an
artifact: with 112 sparse columns, k-means splits on a single binary and calls it a
grouping. Reporting that would have been a false finding. So every method here runs on
:func:`~startup_outcomes.models.protocol.log_numeric_matrix`, the standardised continuous
columns, where a Euclidean distance means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from startup_outcomes import features as feature_tools
from startup_outcomes.config import RANDOM_SEED, TARGET
from startup_outcomes.models import protocol

#: Cluster counts the scan walks. Two is the smallest meaningful split; beyond eight the
#: profile table stops being readable, which is the workbooks' own stopping rule.
CLUSTER_RANGE = range(2, 9)

#: Rows sampled for the silhouette. The full 24,467 x 24,467 distance matrix is 4.5 GB and
#: the estimate is stable well below that.
SILHOUETTE_SAMPLE = 5000

#: Silhouette bands from the workbooks: above 0.5 is clear structure, above 0.25 is weak,
#: below is no meaningful grouping.
CLEAR_STRUCTURE, WEAK_STRUCTURE = 0.5, 0.25

#: Variance a principal-component subspace must retain to be called a summary of the data.
VARIANCE_TARGET = 0.90

#: Restarts per k. Fixed rather than left at the default so the result is reproducible.
KMEANS_RESTARTS = 10


def cluster_scan(matrix: pd.DataFrame) -> pd.DataFrame:
    """Inertia and silhouette at each candidate ``k`` — the elbow and silhouette curves.

    Inertia always falls as ``k`` rises, so it cannot choose ``k`` on its own; the
    silhouette is what says whether any split is worth making.
    """
    rows = []
    values = matrix.to_numpy()
    for count in CLUSTER_RANGE:
        model = KMeans(n_clusters=count, n_init=KMEANS_RESTARTS, random_state=RANDOM_SEED)
        labels = model.fit_predict(values)
        rows.append(
            {
                "k": int(count),
                "inertia": round(float(model.inertia_), 1),
                "silhouette": round(
                    float(
                        silhouette_score(
                            values,
                            labels,
                            sample_size=SILHOUETTE_SAMPLE,
                            random_state=RANDOM_SEED,
                        )
                    ),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def _structure_label(silhouette: float) -> str:
    if silhouette >= CLEAR_STRUCTURE:
        return "clear structure"
    if silhouette >= WEAK_STRUCTURE:
        return "weak structure"
    return "no meaningful grouping"


def clustering_summary(matrix: pd.DataFrame) -> dict[str, object]:
    """Flat companion to :func:`cluster_scan`, and the verdict it supports."""
    scan = cluster_scan(matrix)
    best_k = int(scan["silhouette"].idxmax())
    best = float(scan.loc[best_k, "silhouette"])
    return {
        "k_values_tried": int(len(scan)),
        "columns_clustered": int(matrix.shape[1]),
        "best_k": best_k,
        "best_silhouette": round(best, 4),
        "structure_verdict": _structure_label(best),
        "any_clear_structure": bool(best >= CLEAR_STRUCTURE),
        "inertia_falls_monotonically": bool(bool(scan["inertia"].is_monotonic_decreasing)),
    }


def chosen_k(matrix: pd.DataFrame) -> int:
    """The ``k`` with the best silhouette.

    Computed, not hand-edited into a notebook cell. It is reported alongside the verdict
    so nobody reads a chosen ``k`` as evidence that clusters exist.
    """
    return int(cluster_scan(matrix)["silhouette"].idxmax())


def cluster_profile(frame: pd.DataFrame, matrix: pd.DataFrame, clusters: int) -> pd.DataFrame:
    """What the clusters actually contain.

    The workbooks' rule is that a cluster is useless until it can be named. This table is
    what a name would have to come from — and here it is what shows the clusters are
    slices of a size gradient rather than kinds of company.
    """
    model = KMeans(n_clusters=clusters, n_init=KMEANS_RESTARTS, random_state=RANDOM_SEED)
    labels = model.fit_predict(matrix.to_numpy())
    described = frame.assign(cluster=labels)
    profile = described.groupby("cluster", observed=True).agg(
        rows=("Company_ID", "count"),
        median_funding=("Total_Funding_USD_Millions", "median"),
        median_runway=("Runway_Months_2024", "median"),
        median_headcount=("Peak_Headcount_2023", "median"),
        closed_pct=(TARGET, lambda outcomes: (outcomes == "Closed").mean() * 100),
    )
    profile["top_domain"] = described.groupby("cluster", observed=True)["Domain"].agg(
        lambda values: values.value_counts().index[0]
    )
    profile["share_of_rows_pct"] = (profile["rows"] / profile["rows"].sum() * 100).round(1)
    return profile.round(2)


def cluster_separation(
    frame: pd.DataFrame, matrix: pd.DataFrame, clusters: int
) -> dict[str, object]:
    """Whether the clusters separate the outcome at all.

    The question a stakeholder would ask of any segmentation. If the closed rate is flat
    across clusters, the grouping carries no information about survival however tidy it
    looks in two dimensions.
    """
    profile = cluster_profile(frame, matrix, clusters)
    rates = profile["closed_pct"]
    return {
        "clusters": int(clusters),
        "closed_pct_min": round(float(rates.min()), 2),
        "closed_pct_max": round(float(rates.max()), 2),
        "closed_pct_spread": round(float(rates.max() - rates.min()), 2),
        "smallest_cluster_share_pct": round(float(profile["share_of_rows_pct"].min()), 1),
    }


def variance_curve(matrix: pd.DataFrame) -> pd.DataFrame:
    """Explained variance and its cumulative total, per principal component."""
    model = PCA(random_state=RANDOM_SEED).fit(matrix.to_numpy())
    ratios = model.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1),
            "explained_pct": np.round(ratios * 100, 2),
            "cumulative_pct": np.round(np.cumsum(ratios) * 100, 2),
        }
    ).set_index("component")


def pca_summary(matrix: pd.DataFrame) -> dict[str, object]:
    """How much of the data a low-dimensional view would actually keep.

    The workbooks insist that a PCA report states what was thrown away. Here that is the
    finding: the components needed for 90% of the variance are most of the columns, so
    there is no low-dimensional structure to summarise.
    """
    curve = variance_curve(matrix)
    cumulative = curve["cumulative_pct"].to_numpy()
    needed = int(np.argmax(cumulative >= VARIANCE_TARGET * 100) + 1)
    return {
        "columns": int(matrix.shape[1]),
        "pc1_explained_pct": round(float(curve.loc[1, "explained_pct"]), 2),
        "first_two_explained_pct": round(float(curve.loc[2, "cumulative_pct"]), 2),
        "components_for_90pct": needed,
        "share_of_columns_needed_for_90pct": round(needed / matrix.shape[1] * 100, 1),
    }


def pca_coordinates(matrix: pd.DataFrame) -> pd.DataFrame:
    """The first two components, for the scatter plot.

    The axes are linear blends of logged, standardised columns and mean nothing on their
    own — never quote a component value to a reader.
    """
    model = PCA(n_components=2, random_state=RANDOM_SEED)
    projected = model.fit_transform(matrix.to_numpy())
    return pd.DataFrame(projected, columns=["component_1", "component_2"], index=matrix.index)


def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Every unsupervised result, keyed by name."""
    frame = feature_tools.canonical_frame(df)
    design, _ = feature_tools.design_matrix(frame)
    matrix = protocol.log_numeric_matrix(design)
    clusters = chosen_k(matrix)
    return {
        "rows": int(len(matrix)),
        "clustering_summary": clustering_summary(matrix),
        "cluster_separation": cluster_separation(frame, matrix, clusters),
        "pca_summary": pca_summary(matrix),
    }
