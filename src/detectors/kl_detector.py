"""
kl_detector.py

Kullback-Leibler divergence drift detector using k-NN estimation.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma

from .base import BaseDetector
from .factory import DetectorFactory


@DetectorFactory.register("kl")
class KLDetector(BaseDetector):
    """Drift detector based on k-NN estimation of KL divergence.

    Uses the method from Wang et al. (2009) "Divergence estimation for
    multidimensional densities via k-nearest-neighbor distances".
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._X_ref = None

    def fit(self, X_ref: np.ndarray) -> None:
        self._X_ref = X_ref.astype(np.float64)

    def score(self, X_test: np.ndarray) -> float:
        X_test = X_test.astype(np.float64)
        d = X_test.shape[1]
        n = len(self._X_ref)
        m = len(X_test)

        tree_ref = KDTree(self._X_ref)
        dist_ref, _ = tree_ref.query(X_test, k=self.n_neighbors)
        if dist_ref.ndim > 1:
            dist_ref = dist_ref[:, -1]
        dist_ref = np.maximum(dist_ref, 1e-12)

        tree_test = KDTree(X_test)
        dist_test, _ = tree_test.query(X_test, k=self.n_neighbors + 1)
        if dist_test.ndim > 1:
            dist_test = dist_test[:, -1]
        dist_test = np.maximum(dist_test, 1e-12)

        kl = d * np.mean(np.log(dist_ref / dist_test)) + np.log(n / (m - 1))
        return float(max(0.0, kl))