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
        self._X_ref = None

    def fit(self, X_ref: np.ndarray) -> None:
        self._X_ref = X_ref

    @staticmethod
    def _to_dataset(X: np.ndarray) -> Dataset:
        columns = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=columns)
        definition = DataDefinition(numerical_columns=columns)
        return Dataset.from_pandas(df, data_definition=definition)

    def score(self, X_test: np.ndarray) -> float:
        ref_dataset = self._to_dataset(self._X_ref)
        cur_dataset = self._to_dataset(X_test)

        report = Report(metrics=[DriftedColumnsCount()])
        my_eval = report.run(current_data=cur_dataset, reference_data=ref_dataset)

        result = my_eval.dict()
        value = result["metrics"][0]["value"]
        drift_share = value["share"] if isinstance(value, dict) else value
        return float(drift_share)