"""Session-scoped fixtures for the expensive shared objects.

The two original test files each declared their own module-scoped ``df`` fixture, which
was right when loading was the only cost. The modelling tiers changed that: fitting the
same out-of-fold score vector once per test file would dominate the suite's runtime.

Caching lives here rather than inside ``src/`` deliberately. A memo inside a tier module
would be module-level mutable state, which ``arch-data-pipeline.md`` rules out because it
makes results depend on call order. A pytest fixture is scoped, visible, and thrown away
when the session ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startup_outcomes import features, report
from startup_outcomes.load import load_raw
from startup_outcomes.models import protocol, supervised


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """All 25,000 rows, as loaded. What the README's "Dataset at a glance" describes."""
    return load_raw()


@pytest.fixture(scope="session")
def analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """The 24,467 rows the diagnostic and modelling tiers run on.

    Distinct from ``raw`` because ``Funding_Stage == 'IPO'`` determines the target, so any
    tier that uses that column must drop those rows first.
    """
    return features.canonical_frame(raw)


@pytest.fixture(scope="session")
def design(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` for the leakage-controlled 13-feature binary problem."""
    return features.design_matrix(analysis)


@pytest.fixture(scope="session")
def oof_scores(design: tuple[pd.DataFrame, pd.Series]) -> np.ndarray:
    """Out-of-fold probabilities from the chosen model — about one second, reused widely."""
    features_frame, target = design
    return protocol.out_of_fold_probabilities(supervised.chosen_estimator(), features_frame, target)


@pytest.fixture(scope="session")
def report_results(raw: pd.DataFrame) -> dict[str, object]:
    """Every tier, computed once for the whole suite.

    This is the single largest cost (~45s) and it is deliberately the *only* place the
    tiers run. Every tier-specific fixture below slices this rather than recomputing:
    running ``supervised.run_all`` in its own file as well would double the suite's
    runtime for identical numbers.

    The blast radius is intended. A broken ``run_all`` invalidates the documentation of
    every tier at once, so failing broadly is the correct signal.
    """
    return report.run_all(raw)


@pytest.fixture(scope="session")
def supervised_results(report_results: dict[str, object]) -> dict[str, object]:
    """The supervised tier's slice of the report."""
    return report_results["predictive"]
