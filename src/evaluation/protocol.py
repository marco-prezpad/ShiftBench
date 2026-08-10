"""
protocol.py

Benchmark protocol to evaluate drift detectors on UCI Adult.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data.load import CATEGORICAL_COLS, load_adult
from src.data.synthetic_shifts import generate_adult_demographic_shift
from src.detectors.factory import DetectorFactory
from src.evaluation.metrics import compute_detection_metrics
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging

# Registrar todos los detectores
import src.detectors.mmd_detector  # noqa: F401
import src.detectors.lsdd_detector  # noqa: F401
import src.detectors.kl_detector  # noqa: F401
import src.detectors.embedding_drift_detector  # noqa: F401
import src.detectors.evidently_detector  # noqa: F401

logger = setup_logging()


class BenchmarkProtocol:
    """Run the full distribution shift benchmark on UCI Adult."""

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
        self.scores: dict[str, dict[float, list[float]]] = {}
        self._encoder: OneHotEncoder | None = None

    def run(self) -> None:
        logger.info("Starting benchmark protocol...")
        ensure_dir(self.results_dir)
        ensure_dir(self.results_dir / "figures")

        # 1. Cargar y dividir los datos
        df = load_adult(self.data_path)
        df_ref, df_pool = train_test_split(
            df, train_size=self.reference_frac, random_state=self.random_state
        )
        logger.info(f"Reference size: {len(df_ref)}, Pool size: {len(df_pool)}")

        # 2. Preparar los datos sin la columna target
        X_ref_raw = df_ref.drop(columns=["income"])
        y_pool = df_pool["income"]
        X_pool_raw = df_pool.drop(columns=["income"])

        # 3. Entrenar el codificador one-hot una sola vez
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._encoder.fit(
            pd.concat([X_ref_raw[CATEGORICAL_COLS], X_pool_raw[CATEGORICAL_COLS]])
        )

        X_ref_num = self._encode_dataframe(X_ref_raw)
        X_pool_num = self._encode_dataframe(X_pool_raw)

        # 4. Inicializar detectores
        detectors = self._init_detectors(X_ref_num.shape)

        # 5. Entrenar todos los detectores
        for name, det in detectors.items():
            if name == "evidently":
                det.fit(X_ref_raw)
            else:
                det.fit(X_ref_num)

        # 6. Bucle principal: alphas × bootstraps
        for alpha in self.alphas:
            logger.info(f"Running alpha={alpha}")
            for detector_name in detectors:
                self.scores.setdefault(detector_name, {}).setdefault(alpha, [])

            for run_id in range(self.n_bootstrap):
                seed = self.random_state + run_id
                X_test_raw, _ = generate_adult_demographic_shift(
                    X_pool_raw,
                    y_pool,
                    subgroup_col=self.subgroup_col,
                    subgroup_value=self.subgroup_value,
                    alpha=alpha,
                    random_state=seed,
                )
                X_test_num = self._encode_dataframe(X_test_raw)

                for name, det in detectors.items():
                    if name == "evidently":
                        score = det.score(X_test_raw)
                    else:
                        score = det.score(X_test_num)
                    self.scores[name][alpha].append(score)

        # 7. Guardar scores crudos
        with open(self.results_dir / "scores.json", "w") as f:
            json.dump(self.scores, f, indent=2)

        # 8. Calcular métricas
        metrics_df = compute_detection_metrics(
            self.scores,
            self.alphas,
            self.n_bootstrap,
            threshold_percentile=(1 - self.significance_level) * 100,
        )
        metrics_df.to_csv(self.results_dir / "metrics.csv", index=False)

        logger.info("Benchmark finished.")
        logger.info(f"Results saved to {self.results_dir}")

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Convierte un DataFrame en array numérico one-hot."""
        num = df.select_dtypes(include=[np.number])
        cat_encoded = self._encoder.transform(df[CATEGORICAL_COLS])
        return np.hstack([num.values, cat_encoded])

    def _init_detectors(self, X_ref_num_shape: tuple) -> dict:
        names = ["mmd", "lsdd", "kl", "embedding", "evidently"]
        detectors = {}
        for name in names:
            kwargs = {}
            # For detectors that may use GPU, force CPU if reference has >5000 samples
            if name in ("mmd", "lsdd") and X_ref_num_shape[0] > 5000:
                kwargs["device"] = "cpu"
            detectors[name] = DetectorFactory.create(name, **kwargs)
        return detectors