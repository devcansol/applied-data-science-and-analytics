"""The scores a model has to beat before it has learned anything.

Three baselines, in increasing order of how much they already know:

1. **Majority class** — predicts "survives" for every company. It scores 85.89% accuracy,
   which is why accuracy is not the headline anywhere in this project.
2. **Stratified random** — guesses at the base rate. It calibrates what a ranking metric
   looks like when the ranking is pure noise.
3. **The runway threshold rule** — one comparison against six months of runway, which the
   audit identified as a hand-authored rule in the data generator before any model
   existed. This is the honest comparator: a gradient-boosted model that cannot beat it
   has rediscovered the generator's scaffolding and nothing else.

The third is the one that matters, and it is the reason this module exists rather than
just calling ``DummyClassifier`` twice. A baseline that encodes what is already known is
what turns "the model scores 0.19" into a statement with content.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from startup_outcomes.audit import RUNWAY_CLIFF_MONTHS
from startup_outcomes.config import RANDOM_SEED

#: The column the threshold rule reads. Named as a default so the rule is reusable if the
#: audit ever locates a second threshold.
RUNWAY_COLUMN = "Runway_Months_2024"


class RunwayThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Predicts closure from a single comparison: is runway below the threshold?

    Fitted rather than hard-coded — it learns the closure rate on each side of the
    threshold from the training fold, so it returns calibrated probabilities and can be
    ranked by the same metrics as any other model. Hard-coding the two rates would make it
    unfittable and would quietly let test-fold information in.

    The threshold itself comes from ``audit.RUNWAY_CLIFF_MONTHS``, established from the
    data before any model was built, so it is not tuned against the metric it is judged by.
    """

    def __init__(
        self,
        column: str = RUNWAY_COLUMN,
        threshold: float = RUNWAY_CLIFF_MONTHS,
    ) -> None:
        self.column = column
        self.threshold = threshold

    def fit(self, features: pd.DataFrame, target: pd.Series) -> RunwayThresholdClassifier:
        """Learn the closure rate either side of the threshold."""
        labels = np.asarray(target, dtype=float)
        below = self._below(features)
        self.classes_ = np.array([0, 1])
        # Fall back to the pooled rate if a fold happens to have nothing on one side.
        overall = float(labels.mean())
        self.rate_below_ = float(labels[below].mean()) if below.any() else overall
        self.rate_at_or_above_ = float(labels[~below].mean()) if (~below).any() else overall
        return self

    def _below(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(features[self.column].to_numpy(dtype=float) < self.threshold)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of closure: one of two learned rates."""
        check_is_fitted(self, "rate_below_")
        positive = np.where(self._below(features), self.rate_below_, self.rate_at_or_above_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Hard prediction at the usual 0.5 cut, which this rule never crosses.

        Kept for interface compatibility. It always returns zero on this data, because
        even below six months of runway only 23% of companies close — which is itself the
        argument for reading the rule as a ranker rather than a classifier.
        """
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def majority_baseline() -> DummyClassifier:
    """Always predicts the majority class."""
    return DummyClassifier(strategy="most_frequent")


def stratified_baseline() -> DummyClassifier:
    """Guesses at the observed base rate."""
    return DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)


def runway_rule() -> RunwayThresholdClassifier:
    """The known generator rule, as a fitted estimator."""
    return RunwayThresholdClassifier()


def all_baselines() -> dict[str, object]:
    """Every baseline, in the order they should be read."""
    return {
        "majority baseline": majority_baseline(),
        "stratified baseline": stratified_baseline(),
        "runway threshold rule": runway_rule(),
    }
