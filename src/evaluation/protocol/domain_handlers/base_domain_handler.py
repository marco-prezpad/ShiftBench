"""
base_domain_handler.py

Strategy interface for per-domain data handling in ShiftBench.

Each domain (tabular, image, time series, text) implements this interface
so BenchmarkProtocol can prepare data, generate shifts, and build detector
inputs the same way regardless of domain. Adding a new domain means
writing a new handler class + registering it here, not editing
BenchmarkProtocol itself.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseDomainHandler(ABC):
    """Encapsulates all domain-specific logic needed by BenchmarkProtocol.

    Subclasses are expected to set `self.X_ref_numeric` and
    `self.X_pool_numeric` inside `prepare_reference_and_pool`, since the
    default `reference_kl` / `build_kl_test_view` implementations below
    reuse them.
    """

    X_ref_numeric: np.ndarray
    X_pool_numeric: np.ndarray

    @abstractmethod
    def prepare_reference_and_pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Load reference/pool data and return (X_ref_numeric, X_pool_numeric).

        Also responsible for storing whatever internal state the other
        methods need (reference views for other detector families, the
        pool used to build shifted test sets, pool labels, etc.).
        """

    @abstractmethod
    def generate_shifted_test_set(self, alpha: float, seed: int):
        """Return a (X_test_shifted, y_test_shifted) pair at shift intensity alpha."""

    @abstractmethod
    def build_numeric_test_view(self, X_test_shifted) -> np.ndarray:
        """Representation of the shifted test set for the MMD/LSDD/embedding detectors."""

    @abstractmethod
    def build_evidently_test_view(self, X_test_shifted):
        """Representation of the shifted test set for the Evidently detector."""

    @property
    @abstractmethod
    def reference_evidently(self):
        """Reference data for the Evidently detector's fit()."""

    @property
    def reference_kl(self) -> np.ndarray:
        """Reference data for the KL detector's fit().

        Defaults to the same representation used by MMD/LSDD/embedding.
        Override alongside `build_kl_test_view` for domains where KL
        needs a PCA-reduced representation instead (see the image and
        text domain handlers).
        """
        return self.X_ref_numeric

    def build_kl_test_view(self, X_test_shifted) -> np.ndarray:
        """Representation of the shifted test set for the KL detector.

        Defaults to `build_numeric_test_view`; override alongside
        `reference_kl` when a domain needs a different representation.
        """
        return self.build_numeric_test_view(X_test_shifted)
