"""
mmd_detector.py

Maximum Mean Discrepancy drift detector using Alibi Detect.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
import numpy as np
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.pytorch.kernels import GaussianRBF

from .base import BaseDetector
from .factory import DetectorFactory


@DetectorFactory.register("mmd")
class MMDDetector(BaseDetector):
    """Drift detector based on Maximum Mean Discrepancy."""

    def __init__(self, p_val: float = 0.05):
        self.p_val = p_val
        self._detector = None

    def fit(self, X_ref: np.ndarray) -> None:
        self._detector = MMDDrift(
            X_ref,
            backend="pytorch",
            p_val=self.p_val,
            kernel=GaussianRBF,
            configure_kernel_from_x_ref=True,
        )

    def score(self, X_test: np.ndarray) -> float:
        preds = self._detector.predict(X_test, return_p_val=True)
        return float(1.0 - preds["data"]["p_val"])