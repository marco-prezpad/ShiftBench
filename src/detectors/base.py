"""
base.py

Abstract base class for all drift detectors.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for unsupervised drift detectors."""

    @abstractmethod
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Args:
            X_ref: Reference dataset of shape (n_samples, n_features).
        """

    @abstractmethod
    def score(self, X_test: np.ndarray) -> float:
        """Compute a drift score for test data.

        Args:
            X_test: Test dataset of shape (n_samples, n_features).

        Returns:
            Drift score (higher values indicate more drift).
        """
