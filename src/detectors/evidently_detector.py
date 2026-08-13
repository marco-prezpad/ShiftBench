"""
evidently_detector.py

Drift detection using Evidently AI.

Author: Marco Pérez Padilla
Date:   12-08-2026
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

    def fit(self, X_ref: np.ndarray | pd.DataFrame) -> None:
        """Fit the detector on reference data. Accepts both numpy arrays and DataFrames."""
        if isinstance(X_ref, np.ndarray):
            cols = [f"f{i}" for i in range(X_ref.shape[1])]
            df = pd.DataFrame(X_ref, columns=cols)
        else:
            df = X_ref
        self._ref_dataset = self._to_dataset(df)

    @staticmethod
    def _to_dataset(df: pd.DataFrame) -> Dataset:
        """Create an Evidently Dataset with explicit column types."""
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        definition = DataDefinition(
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
        )
        return Dataset.from_pandas(df, data_definition=definition)

    def score(self, X_test: np.ndarray | pd.DataFrame) -> float:
        """Compute drift score. Accepts both numpy arrays and DataFrames."""
        if isinstance(X_test, np.ndarray):
            cols = [f"f{i}" for i in range(X_test.shape[1])]
            df = pd.DataFrame(X_test, columns=cols)
        else:
            df = X_test

        cur_dataset = self._to_dataset(df)
        report = Report(metrics=[DriftedColumnsCount()])
        my_eval = report.run(current_data=cur_dataset, reference_data=self._ref_dataset)

        result = my_eval.dict()
        value = result["metrics"][0]["value"]
        drift_share = value["share"] if isinstance(value, dict) else value
        return float(drift_share)