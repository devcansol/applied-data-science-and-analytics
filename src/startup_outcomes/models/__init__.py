"""Tier 3 and 4 — modelling, evaluation, and decision analysis.

A package rather than a module because ``arch-modeling.md`` globs ``models/**``: code
under this path is where the leakage, baseline, and collinearity mandates apply, and the
glob is what loads them for a reviewer. The descriptive and diagnostic tiers sit outside
it deliberately — they must never fit an estimator.

``protocol`` is the shared layer every sibling depends on: it owns the splitter, the
preprocessing, the metrics, and the out-of-fold convention. Nothing here defines a second
way to score a model.
"""

__all__ = ["baselines", "card", "prescriptive", "protocol", "supervised", "unsupervised"]
