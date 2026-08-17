"""
protocol.py

Generic benchmark protocol for ShiftBench.
Supports domains: 'adult' (tabular) and 'cifar10c' (image embeddings).

Author: Marco Pérez Padilla
Date:   16-08-2026
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.data.load import CATEGORICAL_COLS, load_adult, load_cifar10_embeddings
from src.data.synthetic_shifts import (
    generate_adult_demographic_shift,
    generate_class_mixture_shift,
    generate_synthetic_series,
)

from src.detectors.factory import DetectorFactory
from src.evaluation.metrics import compute_detection_metrics
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging

# Import detectors to trigger registration
import src.detectors.mmd_detector  # noqa: F401
import src.detectors.lsdd_detector  # noqa: F401
import src.detectors.kl_detector  # noqa: F401
import src.detectors.embedding_drift_detector  # noqa: F401
import src.detectors.evidently_detector  # noqa: F401

logger = setup_logging()


class TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


alibi_logger = logging.getLogger("alibi_detect")
alibi_logger.addHandler(TqdmHandler())
alibi_logger.setLevel(logging.INFO)
alibi_logger.propagate = False


class BenchmarkProtocol:
    """Run distribution shift benchmark for different data domains."""

    def __init__(
        self,
        domain: str = "adult",
        data_path: str | Path = "data/adult.data",
        embeddings_dir: str | Path = "embeddings/cifar10",
        results_dir: str | Path = "results",
        random_state: int = 42,
        reference_frac: float = 0.5,
        subgroup_col: str = "race",
        subgroup_value: str = "Black",
        noise_std: float = 0.5,
        alphas: list[float] | None = None,
        n_bootstrap: int = 30,
        significance_level: float = 0.05,
        max_kernel_ref_size: int = 1000,
        force: bool = False,
        cifar_test_size: int = 3000,
        cifar_kl_n_components: int = 50,
        class_a: int = 0,
        class_b: int = 1,
        series_n_samples: int = 1000,
        series_length: int = 50,
    ):
        self.domain = domain
        self.data_path = Path(data_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        self.reference_frac = reference_frac
        self.subgroup_col = subgroup_col
        self.subgroup_value = subgroup_value
        self.noise_std = noise_std
        self.alphas = alphas or [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ]
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.max_kernel_ref_size = max_kernel_ref_size
        self.force = force
        self.cifar_test_size = cifar_test_size
        self.cifar_kl_n_components = cifar_kl_n_components
        self.class_a = class_a
        self.class_b = class_b
        self.series_n_samples = series_n_samples
        self.series_length = series_length
        self.scores: dict[str, dict[float, list[float]]] = {}
        self._encoder: OneHotEncoder | None = None
        self._checkpoint_path = self.results_dir / "scores_partial.json"

        # Domain-specific data attributes
        self._encode_test = False
        self._X_ref_evidently = None
        self._X_pool_evidently = None
        self._X_pool_for_shift = None
        self._y_pool = None
        self._shift_fn = None
        self._shift_kwargs = {}
        self._pca_kl = None
        self._X_ref_kl = None
        self._X_pool_kl = None


    def _load_and_prepare_data(self):
        """Load and prepare reference/pool data for the selected domain."""
        if self.domain == "adult":
            df = load_adult(self.data_path)
            df_ref, df_pool = train_test_split(
                df, train_size=self.reference_frac, random_state=self.random_state
            )
            X_ref_raw = df_ref.drop(columns=["income"])
            y_pool = df_pool["income"]
            X_pool_raw = df_pool.drop(columns=["income"])

            self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self._encoder.fit(
                pd.concat([X_ref_raw[CATEGORICAL_COLS], X_pool_raw[CATEGORICAL_COLS]])
            )
            X_ref_num = self._encode_dataframe(X_ref_raw)
            X_pool_num = self._encode_dataframe(X_pool_raw)

            self._X_ref_evidently = X_ref_raw
            self._X_pool_evidently = X_pool_raw
            self._X_pool_for_shift = X_pool_raw
            self._y_pool = y_pool
            self._shift_fn = generate_adult_demographic_shift
            self._shift_kwargs = {
                "subgroup_col": self.subgroup_col,
                "subgroup_value": self.subgroup_value,
            }
            self._encode_test = True
            self._pca_kl = None
            self._X_ref_kl = None
            self._X_pool_kl = None
            return X_ref_num, X_pool_num

        elif self.domain == "cifar10c":
            X_train, y_train, _, _ = load_cifar10_embeddings(self.embeddings_dir)

            mask = (y_train == self.class_a) | (y_train == self.class_b)
            X_filtered = X_train[mask]
            y_filtered = y_train[mask]

            rng = np.random.default_rng(self.random_state)

            idx_a_all = np.where(y_filtered == self.class_a)[0]
            ref_size = min(len(idx_a_all), self.max_kernel_ref_size)
            ref_idx = rng.choice(idx_a_all, size=ref_size, replace=False)
            X_ref = X_filtered[ref_idx].astype(np.float32)
            y_ref = y_filtered[ref_idx]

            total_pool = min(len(X_filtered), self.cifar_test_size)
            pool_idx = rng.choice(len(X_filtered), size=total_pool, replace=False)
            X_pool = X_filtered[pool_idx].astype(np.float32)
            y_pool = y_filtered[pool_idx]

            self._X_ref_evidently = X_ref[:, :50]
            self._X_pool_evidently = X_pool[:, :50]

            self._pca_kl = PCA(
                n_components=self.cifar_kl_n_components,
                random_state=self.random_state,
            )
            self._pca_kl.fit(X_ref)
            self._X_ref_kl = self._pca_kl.transform(X_ref).astype(np.float32)
            self._X_pool_kl = self._pca_kl.transform(X_pool).astype(np.float32)

            self._X_pool_for_shift = X_pool
            self._y_pool = y_pool
            self._shift_fn = generate_class_mixture_shift
            self._shift_kwargs = {
                "class_a": self.class_a,
                "class_b": self.class_b,
            }
            self._encode_test = False
            return X_ref, X_pool

        elif self.domain == "timeseries":
            X_all, y_all = generate_synthetic_series(
                n_samples=self.series_n_samples,
                length=self.series_length,
                random_state=self.random_state,
            )

            rng = np.random.default_rng(self.random_state)

            idx_a_all = np.where(y_all == 0)[0]
            ref_size = min(len(idx_a_all), self.max_kernel_ref_size)
            ref_idx = rng.choice(idx_a_all, size=ref_size, replace=False)
            X_ref = X_all[ref_idx].astype(np.float32)

            X_pool = X_all.astype(np.float32)
            y_pool = y_all

            self._X_ref_evidently = X_ref
            self._X_pool_evidently = X_pool

            self._pca_kl = None
            self._X_ref_kl = X_ref
            self._X_pool_kl = X_pool

            self._X_pool_for_shift = X_pool
            self._y_pool = y_pool
            self._shift_fn = generate_class_mixture_shift
            self._shift_kwargs = {"class_a": 0, "class_b": 1}
            self._encode_test = False
            return X_ref, X_pool
        
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """One‑hot encode categorical columns and return float32 array."""
        num = df.select_dtypes(include=[np.number])
        cat_encoded = self._encoder.transform(df[CATEGORICAL_COLS])
        return np.hstack([num.values, cat_encoded]).astype(np.float32)

    def _generate_shift(self, X_pool, y_pool, alpha, seed):
        """Generate a shifted test set for the current domain."""
        return self._shift_fn(
            X_pool, y_pool, alpha=alpha, random_state=seed, **self._shift_kwargs
        )


    def _init_detectors(self) -> dict:
        """Create detector instances with appropriate device."""
        names = ["mmd", "lsdd", "kl", "embedding", "evidently"]
        detectors = {}
        for name in names:
            kwargs = {}
            if name in ("mmd", "lsdd"):
                kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if name == "embedding" and self.domain == "cifar10c":
                kwargs["n_components"] = 64
            detectors[name] = DetectorFactory.create(name, **kwargs)
        return detectors


    def run(self) -> None:
        logger.info(f"Starting benchmark protocol for domain '{self.domain}'...")
        ensure_dir(self.results_dir)
        ensure_dir(self.results_dir / "figures")

        final_metrics = self.results_dir / "metrics.csv"
        if final_metrics.exists() and not self.force:
            logger.info("metrics.csv already exists. Skipping (use --force to re‑run).")
            return

        completed_alphas: set[float] = set()
        if self._checkpoint_path.exists() and not self.force:
            with open(self._checkpoint_path) as f:
                raw = json.load(f)
            self.scores = {
                det: {float(a): v for a, v in alpha_dict.items()}
                for det, alpha_dict in raw.items()
            }
            first_det = next(iter(self.scores.keys()), None)
            if first_det:
                for alpha, sc in self.scores[first_det].items():
                    if len(sc) >= self.n_bootstrap:
                        completed_alphas.add(float(alpha))
            if completed_alphas:
                logger.info(
                    f"Resuming from checkpoint. {len(completed_alphas)} alpha(s) already done."
                )
        elif self.force:
            logger.info("Force flag set – starting from scratch.")

        remaining_alphas = [a for a in self.alphas if a not in completed_alphas]
        if not remaining_alphas:
            logger.info("All alphas already completed.")
            return

        X_ref_num, X_pool_num = self._load_and_prepare_data()
        logger.info(f"Reference shape: {X_ref_num.shape}, Pool shape: {X_pool_num.shape}")

        detectors = self._init_detectors()

        rng = np.random.default_rng(self.random_state)
        for name, det in detectors.items():
            try:
                if name == "kl" and self.domain in ("cifar10c", "timeseries"):
                    det.fit(self._X_ref_kl)
                elif name in ("mmd", "lsdd"):
                    if len(X_ref_num) > self.max_kernel_ref_size:
                        idx = rng.choice(
                            len(X_ref_num), size=self.max_kernel_ref_size, replace=False
                        )
                        det.fit(X_ref_num[idx])
                    else:
                        det.fit(X_ref_num)
                elif name == "evidently":
                    det.fit(self._X_ref_evidently)
                else:
                    det.fit(X_ref_num)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM for {name}, falling back to CPU")
                    detectors[name] = DetectorFactory.create(name, device="cpu")
                    if name == "kl" and self.domain in ("cifar10c", "timeseries"):
                        detectors[name].fit(self._X_ref_kl)
                    elif name in ("mmd", "lsdd"):
                        if len(X_ref_num) > self.max_kernel_ref_size:
                            idx = rng.choice(
                                len(X_ref_num), size=self.max_kernel_ref_size, replace=False
                            )
                            detectors[name].fit(X_ref_num[idx])
                        else:
                            detectors[name].fit(X_ref_num)
                    elif name == "evidently":
                        detectors[name].fit(self._X_ref_evidently)
                    else:
                        detectors[name].fit(X_ref_num)
                else:
                    raise

        for alpha in tqdm(remaining_alphas, desc="Alphas"):
            tqdm.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} Running alpha={alpha}")
            for detector_name in detectors:
                self.scores.setdefault(detector_name, {}).setdefault(alpha, [])

            for run_id in tqdm(range(self.n_bootstrap), desc=f"  α={alpha}", leave=False):
                seed = self.random_state + run_id

                X_test_shifted, _ = self._generate_shift(
                    self._X_pool_for_shift, self._y_pool, alpha=alpha, seed=seed
                )

                if self.domain == "adult":
                    X_test_num = self._encode_dataframe(X_test_shifted)
                    X_test_evidently = X_test_shifted
                    X_test_kl = X_test_num
                elif self.domain == "cifar10c":
                    X_test_num = X_test_shifted.astype(np.float32)
                    X_test_evidently = X_test_shifted[:, :50]
                    X_test_kl = self._pca_kl.transform(X_test_shifted).astype(np.float32)
                else:
                    X_test_num = X_test_shifted.astype(np.float32)
                    X_test_evidently = X_test_shifted
                    X_test_kl = X_test_shifted.astype(np.float32)

                for name, det in detectors.items():
                    try:
                        if name == "evidently":
                            score = det.score(X_test_evidently)
                        elif name == "kl" and self.domain in ("cifar10c", "timeseries"):
                            score = det.score(X_test_kl)
                        else:
                            score = det.score(X_test_num)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(
                                f"GPU OOM during score for {name}. Switching to CPU for this detector."
                            )
                            torch.cuda.empty_cache()
                            detectors[name] = DetectorFactory.create(name, device="cpu")
                            if name == "kl" and self.domain in ("cifar10c", "timeseries"):
                                detectors[name].fit(self._X_ref_kl)
                            elif name in ("mmd", "lsdd"):
                                if len(X_ref_num) > self.max_kernel_ref_size:
                                    idx = rng.choice(
                                        len(X_ref_num),
                                        size=self.max_kernel_ref_size,
                                        replace=False,
                                    )
                                    detectors[name].fit(X_ref_num[idx])
                                else:
                                    detectors[name].fit(X_ref_num)
                            elif name == "evidently":
                                detectors[name].fit(self._X_ref_evidently)
                            else:
                                detectors[name].fit(X_ref_num)
                            if name == "evidently":
                                score = detectors[name].score(X_test_evidently)
                            elif name == "kl" and self.domain in ("cifar10c", "timeseries"):
                                score = detectors[name].score(X_test_kl)
                            else:
                                score = detectors[name].score(X_test_num)
                        else:
                            raise
                    self.scores[name][alpha].append(score)

            alpha_dir = self.results_dir / "alpha_scores"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            alpha_file = alpha_dir / f"alpha_{alpha:.2f}.json"
            alpha_scores = {det: {alpha: self.scores[det][alpha]} for det in self.scores}
            with open(alpha_file, "w") as f:
                json.dump(alpha_scores, f, indent=2)

            first_det = next(iter(self.scores), None)
            if first_det and 0.0 in self.scores[first_det]:
                completed = [a for a in self.alphas if a in self.scores[first_det]]
                partial_metrics = compute_detection_metrics(
                    self.scores,
                    completed,
                    self.n_bootstrap,
                    threshold_percentile=(1 - self.significance_level) * 100,
                )
                partial_metrics.to_csv(self.results_dir / "metrics_partial.csv", index=False)

            with open(self._checkpoint_path, "w") as f:
                json.dump(self.scores, f, indent=2)

        with open(self.results_dir / "scores.json", "w") as f:
            json.dump(self.scores, f, indent=2)

        metrics_df = compute_detection_metrics(
            self.scores,
            self.alphas,
            self.n_bootstrap,
            threshold_percentile=(1 - self.significance_level) * 100,
        )
        metrics_df.to_csv(self.results_dir / "metrics.csv", index=False)

        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

        logger.info("Benchmark finished.")
        logger.info(f"Results saved to {self.results_dir}")