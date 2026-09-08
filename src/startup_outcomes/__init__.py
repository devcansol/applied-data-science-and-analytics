"""Leakage-controlled analysis of a synthetic global tech/AI startup outcomes dataset.

The package is layered, and the layering is the leakage control:

* ``config``     — paths, the one seed, and the feature-timing contract
* ``load``       — typed loading, checksum verification, leaky-stage removal
* ``intervals``  — bootstrap intervals and permutation tests
* ``audit``      — data-integrity checks
* ``features``   — engineered columns and the only design-matrix constructor
* ``descriptive``/``diagnostic`` — tiers 1 and 2, pandas only, no estimator
* ``models``     — tiers 3 and 4, the only place an estimator is fitted
* ``plots``      — figures; computes no statistic
* ``report``     — composes every tier, imported by nothing
"""

__all__ = [
    "audit",
    "config",
    "descriptive",
    "diagnostic",
    "features",
    "intervals",
    "load",
    "models",
    "plots",
    "report",
]
