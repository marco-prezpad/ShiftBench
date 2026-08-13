"""
protocol.py

Benchmark protocol to evaluate drift detectors on UCI Adult.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.data.load import CATEGORICAL_COLS, load_adult
from src.data.synthetic_shifts import generate_adult_demographic_shift
from src.detectors.factory import DetectorFactory
from src.evaluation.metrics import compute_detection_metrics
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging

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
    def __init__(
        self,
        data_path: str | Path = "data/adult.data",
        results_dir: str | Path = "results",
        random_state: int = 42,
        reference_frac: float = 0.5,
        subgroup_col: str = "race",
        subgroup_value: str = "Black",
        alphas: list[float] | None = None,
        n_bootstrap: int = 30,
        significance_level: float = 0.05,
        max_kernel_ref_size: int = 1000,
        force: bool = False,
    ):
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        self.reference_frac = reference_frac
        self.subgroup_col = subgroup_col
        self.subgroup_value = subgroup_value
        self.alphas = alphas or [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ]
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.max_kernel_ref_size = max_kernel_ref_size
        self.force = force
        self.scores: dict[str, dict[float, list[float]]] = {}
        self._encoder: OneHotEncoder | None = None
        self._checkpoint_path = self.results_dir / "scores_partial.json"

    def run(self) -> None:
        logger.info("Starting benchmark protocol...")
        ensure_dir(self.results_dir)
        ensure_dir(self.results_dir / "figures")

        # Early exit
        final_metrics = self.results_dir / "metrics.csv"
        if final_metrics.exists() and not self.force:
            logger.info("metrics.csv already exists. Skipping (use --force to re‑run).")
            return
        
        # Checkpoint
        completed_alphas: set[float] = set()
        if self._checkpoint_path.exists() and not self.force:
            with open(self._checkpoint_path) as f:
                raw = json.load(f)
            self.scores = {
                det: {float(a): v for a, v in alpha_dict.items()}
                for det, alpha_dict in raw.items()
            }
            first_detector = next(iter(self.scores.keys()), None)
            if first_detector:
                for alpha, sc in self.scores[first_detector].items():
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

        # 1. Load and split the data
        df = load_adult(self.data_path)
        df_ref, df_pool = train_test_split(
            df, train_size=self.reference_frac, random_state=self.random_state
        )
        logger.info(f"Reference size: {len(df_ref)}, Pool size: {len(df_pool)}")

        X_ref_raw = df_ref.drop(columns=["income"])
        y_pool = df_pool["income"]
        X_pool_raw = df_pool.drop(columns=["income"])

        # 2. One‑hot encoder
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._encoder.fit(
            pd.concat([X_ref_raw[CATEGORICAL_COLS], X_pool_raw[CATEGORICAL_COLS]])
        )

        X_ref_num = self._encode_dataframe(X_ref_raw)
        X_pool_num = self._encode_dataframe(X_pool_raw)

        # 3. Initialize detectors
        detectors = self._init_detectors(X_ref_num.shape)

        # 4. Train detectors
        rng = np.random.default_rng(self.random_state)
        for name, det in detectors.items():
            try:
                if name in ("mmd", "lsdd"):
                    if len(X_ref_num) > self.max_kernel_ref_size:
                        idx = rng.choice(len(X_ref_num), size=self.max_kernel_ref_size, replace=False)
                        det.fit(X_ref_num[idx])
                    else:
                        det.fit(X_ref_num)
                elif name == "evidently":
                    det.fit(X_ref_raw)
                else:
                    det.fit(X_ref_num)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM for {name}, falling back to CPU")
                    detectors[name] = DetectorFactory.create(name, device="cpu")
                    if name in ("mmd", "lsdd"):
                        if len(X_ref_num) > self.max_kernel_ref_size:
                            idx = rng.choice(len(X_ref_num), size=self.max_kernel_ref_size, replace=False)
                            detectors[name].fit(X_ref_num[idx])
                        else:
                            detectors[name].fit(X_ref_num)
                    elif name == "evidently":
                        detectors[name].fit(X_ref_raw)
                    else:
                        detectors[name].fit(X_ref_num)
                else:
                    raise

        # 5. Evaluation loop
        for alpha in tqdm(remaining_alphas, desc="Alphas"):
            tqdm.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} Running alpha={alpha}")
            for detector_name in detectors:
                self.scores.setdefault(detector_name, {}).setdefault(alpha, [])

            for run_id in tqdm(range(self.n_bootstrap), desc=f"  α={alpha}", leave=False):
                seed = self.random_state + run_id
                X_test_raw, _ = generate_adult_demographic_shift(
                    X_pool_raw, y_pool,
                    subgroup_col=self.subgroup_col,
                    subgroup_value=self.subgroup_value,
                    alpha=alpha,
                    random_state=seed,
                )
                X_test_num = self._encode_dataframe(X_test_raw)

                for name, det in detectors.items():
                    try:
                        if name == "evidently":
                            score = det.score(X_test_raw)
                        else:
                            score = det.score(X_test_num)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"GPU OOM during score for {name}. Switching to CPU for this detector.")
                            torch.cuda.empty_cache()
                            detectors[name] = DetectorFactory.create(name, device="cpu")
                            if name in ("mmd", "lsdd") and len(X_ref_num) > self.max_kernel_ref_size:
                                idx = rng.choice(len(X_ref_num), size=self.max_kernel_ref_size, replace=False)
                                detectors[name].fit(X_ref_num[idx])
                            elif name == "evidently":
                                detectors[name].fit(X_ref_raw)
                            else:
                                detectors[name].fit(X_ref_num)
                            if name == "evidently":
                                score = detectors[name].score(X_test_raw)
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
                completed_alphas = [a for a in self.alphas if a in self.scores[first_det]]
                partial_metrics = compute_detection_metrics(
                    self.scores,
                    completed_alphas,
                    self.n_bootstrap,
                    threshold_percentile=(1 - self.significance_level) * 100,
                )
                partial_metrics.to_csv(self.results_dir / "metrics_partial.csv", index=False)

            with open(self._checkpoint_path, "w") as f:
                json.dump(self.scores, f, indent=2)

        # 6. Save final scores and compute metrics
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

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        num = df.select_dtypes(include=[np.number])
        cat_encoded = self._encoder.transform(df[CATEGORICAL_COLS])
        return np.hstack([num.values, cat_encoded]).astype(np.float32)

    def _init_detectors(self, X_ref_num_shape: tuple) -> dict:
        names = ["mmd", "lsdd", "kl", "embedding", "evidently"]
        detectors = {}
        for name in names:
            kwargs = {}
            if name in ("mmd", "lsdd") and torch.cuda.is_available():
                kwargs["device"] = "cuda"  
            elif name in ("mmd", "lsdd"):
                kwargs["device"] = "cpu"
            detectors[name] = DetectorFactory.create(name, **kwargs)
        return detectors