"""
evidently_detector.py

Drift detection using Evidently AI.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount

from .base import BaseDetector
from .factory import DetectorFactory


@DetectorFactory.register("evidently")
class EvidentlyDetector(BaseDetector):
    """Drift detector wrapping Evidently AI's DriftedColumnsCount metric."""

    def __init__(self):
        self._ref_dataset: Dataset | None = None

    @staticmethod
    def _to_dataset(X: np.ndarray | pd.DataFrame) -> Dataset:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include="number").columns.tolist()
        definition = DataDefinition(
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
        )
        return Dataset.from_pandas(X, data_definition=definition)

    def fit(self, X_ref: np.ndarray | pd.DataFrame) -> None:
        self._ref_dataset = self._to_dataset(X_ref)

    def score(self, X_test: np.ndarray | pd.DataFrame) -> float:
        cur_dataset = self._to_dataset(X_test)

        report = Report(metrics=[DriftedColumnsCount()])
        my_eval = report.run(current_data=cur_dataset, reference_data=self._ref_dataset)

        result = my_eval.dict()
        value = result["metrics"][0]["value"]
        drift_share = value["share"] if isinstance(value, dict) else value
        return float(drift_share)