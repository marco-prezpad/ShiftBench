"""
cifar10c_domain_handler.py

Domain handler for CIFAR-10 ResNet18 embeddings, with a synthetic
class-mixture shift between two classes.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.data.loaders.cifar10_loader import load_cifar10_embeddings
from src.data.shifts.class_mixture_shift import generate_class_mixture_shift

from .base_domain_handler import BaseDomainHandler
from .factory import DomainHandlerFactory


@DomainHandlerFactory.register("cifar10")
class Cifar10cDomainHandler(BaseDomainHandler):
    """Prepares reference/pool data and shifts for the CIFAR-10 image domain."""

    def __init__(
        self,
        embeddings_dir: str | Path = "embeddings/cifar10",
        class_a: int = 0,
        class_b: int = 1,
        max_kernel_ref_size: int = 1000,
        cifar_test_size: int = 3000,
        cifar_kl_n_components: int = 50,
        random_state: int = 42,
        **_unused_kwargs,
    ):
        self.embeddings_dir = embeddings_dir
        self.class_a = class_a
        self.class_b = class_b
        self.max_kernel_ref_size = max_kernel_ref_size
        self.cifar_test_size = cifar_test_size
        self.cifar_kl_n_components = cifar_kl_n_components
        self.random_state = random_state

        self._pca_kl: PCA | None = None
        self._X_ref_evidently = None
        self._X_ref_kl = None
        self._X_pool_for_shift = None
        self._y_pool = None

    def prepare_reference_and_pool(self) -> tuple[np.ndarray, np.ndarray]:
        X_train, y_train, _, _ = load_cifar10_embeddings(self.embeddings_dir)

        class_mask = (y_train == self.class_a) | (y_train == self.class_b)
        X_filtered = X_train[class_mask]
        y_filtered = y_train[class_mask]

        rng = np.random.default_rng(self.random_state)

        idx_class_a_all = np.where(y_filtered == self.class_a)[0]
        ref_size = min(len(idx_class_a_all), self.max_kernel_ref_size)
        ref_idx = rng.choice(idx_class_a_all, size=ref_size, replace=False)
        X_ref = X_filtered[ref_idx].astype(np.float32)

        total_pool = min(len(X_filtered), self.cifar_test_size)
        pool_idx = rng.choice(len(X_filtered), size=total_pool, replace=False)
        X_pool = X_filtered[pool_idx].astype(np.float32)
        y_pool = y_filtered[pool_idx]

        self._X_ref_evidently = X_ref[:, :50]

        self._pca_kl = PCA(n_components=self.cifar_kl_n_components, random_state=self.random_state)
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
            class_a=self.class_a,
            class_b=self.class_b,
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
