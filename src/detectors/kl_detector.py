"""
kl_detector.py

Kullback-Leibler divergence drift detector using k-NN estimation.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
from scipy.spatial import KDTree

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
        n_features = X_test.shape[1]
        n_ref = len(self._X_ref)
        n_test = len(X_test)

        reference_tree = KDTree(self._X_ref)
        dist_to_ref, _ = reference_tree.query(X_test, k=self.n_neighbors)
        if dist_to_ref.ndim > 1:
            dist_to_ref = dist_to_ref[:, -1]
        dist_to_ref = np.maximum(dist_to_ref, 1e-12)

        test_tree = KDTree(X_test)
        dist_to_test, _ = test_tree.query(X_test, k=self.n_neighbors + 1)
        if dist_to_test.ndim > 1:
            dist_to_test = dist_to_test[:, -1]
        dist_to_test = np.maximum(dist_to_test, 1e-12)

        kl_divergence = n_features * np.mean(np.log(dist_to_ref / dist_to_test)) + np.log(
            n_ref / (n_test - 1)
        )
        return float(max(0.0, kl_divergence))
