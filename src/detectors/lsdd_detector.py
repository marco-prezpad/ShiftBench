"""
lsdd_detector.py

Least-Squares Density Difference drift detector using Alibi Detect.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
from alibi_detect.cd import LSDDDrift

from .base import BaseDetector
from .factory import DetectorFactory


@DetectorFactory.register("lsdd")
class LSDDDetector(BaseDetector):
    """Drift detector based on Least-Squares Density Difference."""

    def __init__(self, p_val: float = 0.05):
        self.p_val = p_val
        self._detector = None

    def fit(self, X_ref: np.ndarray) -> None:
        self._detector = LSDDDrift(
            X_ref,
            backend="pytorch",
            p_val=self.p_val,
        )

    def score(self, X_test: np.ndarray) -> float:
        preds = self._detector.predict(X_test, return_p_val=True)
        return float(1.0 - preds["data"]["p_val"])