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

    def __init__(self, p_val: float = 0.05, device: str | None = None):
        self.p_val = p_val
        self.device = device
        self._detector = None

    def fit(self, X_ref: np.ndarray) -> None:
        detector_kwargs = dict(
            backend="pytorch",
            p_val=self.p_val,
        )
        if self.device is not None:
            detector_kwargs["device"] = self.device
        self._detector = LSDDDrift(X_ref, **detector_kwargs)

    def score(self, X_test: np.ndarray) -> float:
        prediction = self._detector.predict(X_test, return_p_val=True)
        return float(1.0 - prediction["data"]["p_val"])
