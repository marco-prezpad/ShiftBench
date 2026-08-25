"""
text_domain_handler.py

Domain handler for TF-IDF embeddings of 20 Newsgroups (two categories),
with a synthetic class-mixture shift.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from src.data.shifts.class_mixture_shift import generate_class_mixture_shift

from .base_domain_handler import BaseDomainHandler
from .factory import DomainHandlerFactory


@DomainHandlerFactory.register("text")
class TextDomainHandler(BaseDomainHandler):
    """Prepares reference/pool data and shifts for the TF-IDF text domain."""

    def __init__(
        self,
        embeddings_dir: str | Path = "embeddings/newsgroups",
        max_kernel_ref_size: int = 1000,
        text_test_size: int = 3000,
        text_kl_n_components: int = 50,
        random_state: int = 42,
        **_unused_kwargs,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_kernel_ref_size = max_kernel_ref_size
        self.text_test_size = text_test_size
        self.text_kl_n_components = text_kl_n_components
        self.random_state = random_state

        self._pca_kl: PCA | None = None
        self._X_ref_evidently = None
        self._X_ref_kl = None
        self._X_pool_for_shift = None
        self._y_pool = None

    def prepare_reference_and_pool(self) -> tuple[np.ndarray, np.ndarray]:
        X_train = np.load(self.embeddings_dir / "train_embeddings.npy")
        y_train = np.load(self.embeddings_dir / "train_labels.npy")

        # Drop features with near-zero variance across the whole training set.
        variance_selector = VarianceThreshold(threshold=1e-8)
        X_train = variance_selector.fit_transform(X_train)

        class_mask = (y_train == 0) | (y_train == 1)
        X_filtered = X_train[class_mask]
        y_filtered = y_train[class_mask]

        rng = np.random.default_rng(self.random_state)

        idx_class_a = np.where(y_filtered == 0)[0]
        ref_size = min(len(idx_class_a), self.max_kernel_ref_size)
        ref_idx = rng.choice(idx_class_a, size=ref_size, replace=False)
        X_ref = X_filtered[ref_idx].astype(np.float32)

        # Also drop features that are constant within the reference itself.
        reference_variances = X_ref.var(axis=0)
        keep_columns = reference_variances > 1e-8
        X_ref = X_ref[:, keep_columns]

        total_pool = min(len(X_filtered), self.text_test_size)
        pool_idx = rng.choice(len(X_filtered), size=total_pool, replace=False)
        X_pool = X_filtered[pool_idx].astype(np.float32)
        y_pool = y_filtered[pool_idx]
        X_pool = X_pool[:, keep_columns]

        self._keep_columns = keep_columns
        self._X_ref_evidently = X_ref[:, :50]

        self._pca_kl = PCA(n_components=self.text_kl_n_components, random_state=self.random_state)
        self._pca_kl.fit(X_ref)
        self._X_ref_kl = self._pca_kl.transform(X_ref).astype(np.float32)

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
        return X_test_shifted[:, :50]

    def build_kl_test_view(self, X_test_shifted: np.ndarray) -> np.ndarray:
        return self._pca_kl.transform(X_test_shifted).astype(np.float32)

    @property
    def reference_evidently(self) -> np.ndarray:
        return self._X_ref_evidently

    @property
    def reference_kl(self) -> np.ndarray:
        return self._X_ref_kl
