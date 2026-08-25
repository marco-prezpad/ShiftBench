"""
embedding_drift_detector.py

Drift detection via dimensionality reduction + two-sample test.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from .base import BaseDetector
from .factory import DetectorFactory


@DetectorFactory.register("embedding")
class EmbeddingDriftDetector(BaseDetector):
    """Drift detector that projects data with PCA and compares distributions.

    Uses the maximum mean discrepancy (MMD) approximation on the projected
    space to measure drift.
    """

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self._pca = None
        self._X_ref_proj = None
        self._rng = None

    def fit(self, X_ref: np.ndarray) -> None:
        n_components = min(self.n_components, X_ref.shape[1], len(X_ref))
        self._pca = PCA(n_components=n_components, random_state=42)
        self._X_ref_proj = self._pca.fit_transform(X_ref)
        self._rng = np.random.default_rng(42)

    def score(self, X_test: np.ndarray) -> float:
        X_test_proj = self._pca.transform(X_test)
        X_test_proj = X_test_proj + self._rng.normal(0, 1e-6, size=X_test_proj.shape)

        cross_distance = np.mean(cdist(self._X_ref_proj, X_test_proj))
        reference_self_distance = np.mean(cdist(self._X_ref_proj, self._X_ref_proj))
        return float(max(0.0, cross_distance - reference_self_distance))
