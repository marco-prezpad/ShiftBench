"""
timeseries_domain_handler.py

Domain handler for synthetic time series (sine vs. square wave), with a
synthetic class-mixture shift.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
import numpy as np

from src.data.shifts.class_mixture_shift import generate_class_mixture_shift
from src.data.shifts.series_generator import generate_synthetic_series

from .base_domain_handler import BaseDomainHandler
from .factory import DomainHandlerFactory


@DomainHandlerFactory.register("timeseries")
class TimeseriesDomainHandler(BaseDomainHandler):
    """Prepares reference/pool data and shifts for the synthetic time series domain."""

    def __init__(
        self,
        series_n_samples: int = 1000,
        series_length: int = 50,
        max_kernel_ref_size: int = 1000,
        random_state: int = 42,
        **_unused_kwargs,
    ):
        self.series_n_samples = series_n_samples
        self.series_length = series_length
        self.max_kernel_ref_size = max_kernel_ref_size
        self.random_state = random_state

        self._X_ref_evidently = None
        self._X_pool_for_shift = None
        self._y_pool = None

    def prepare_reference_and_pool(self) -> tuple[np.ndarray, np.ndarray]:
        X_all, y_all = generate_synthetic_series(
            n_samples=self.series_n_samples,
            length=self.series_length,
            random_state=self.random_state,
        )

        rng = np.random.default_rng(self.random_state)

        idx_class_a_all = np.where(y_all == 0)[0]
        ref_size = min(len(idx_class_a_all), self.max_kernel_ref_size)
        ref_idx = rng.choice(idx_class_a_all, size=ref_size, replace=False)
        X_ref = X_all[ref_idx].astype(np.float32)

        X_pool = X_all.astype(np.float32)
        y_pool = y_all

        self._X_ref_evidently = X_ref
        self.X_ref_numeric = X_ref
        self.X_pool_numeric = X_pool
        self._X_pool_for_shift = X_pool
        self._y_pool = y_pool
        return X_ref, X_pool

    def generate_shifted_test_set(self, alpha: float, seed: int):
        return generate_class_mixture_shift(
            self._X_pool_for_shift,
            self._y_pool,
            alpha=alpha,
            class_a=0,
            class_b=1,
            random_state=seed,
        )

    def build_numeric_test_view(self, X_test_shifted: np.ndarray) -> np.ndarray:
        return X_test_shifted.astype(np.float32)

    def build_evidently_test_view(self, X_test_shifted: np.ndarray) -> np.ndarray:
        return X_test_shifted

    @property
    def reference_evidently(self) -> np.ndarray:
        return self._X_ref_evidently

    # reference_kl and build_kl_test_view use BaseDomainHandler's defaults:
    # this domain never applied PCA before the KL detector either.
