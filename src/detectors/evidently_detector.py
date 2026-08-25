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
        reference_df = self._to_dataframe(X_ref)
        self._ref_dataset = self._to_dataset(reference_df)

    @staticmethod
    def _to_dataframe(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Wrap a numpy array in a DataFrame with generic column names."""
        if isinstance(X, np.ndarray):
            column_names = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=column_names)
        return X

    @staticmethod
    def _to_dataset(dataframe: pd.DataFrame) -> Dataset:
        """Create an Evidently Dataset with explicit column types."""
        numerical_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()
        definition = DataDefinition(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
        return Dataset.from_pandas(dataframe, data_definition=definition)

    def score(self, X_test: np.ndarray | pd.DataFrame) -> float:
        """Compute drift score. Accepts both numpy arrays and DataFrames."""
        test_df = self._to_dataframe(X_test)
        current_dataset = self._to_dataset(test_df)

        report = Report(metrics=[DriftedColumnsCount()])
        evaluation = report.run(current_data=current_dataset, reference_data=self._ref_dataset)

        result = evaluation.dict()
        value = result["metrics"][0]["value"]
        drift_share = value["share"] if isinstance(value, dict) else value
        return float(drift_share)
