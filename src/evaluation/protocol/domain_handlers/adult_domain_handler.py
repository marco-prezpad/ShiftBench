"""
adult_domain_handler.py

Domain handler for the tabular UCI Adult dataset, with a synthetic
demographic-subgroup shift.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data.loaders.adult_loader import CATEGORICAL_COLS, load_adult
from src.data.shifts.demographic_shift import generate_adult_demographic_shift

from .base_domain_handler import BaseDomainHandler
from .factory import DomainHandlerFactory


@DomainHandlerFactory.register("adult")
class AdultDomainHandler(BaseDomainHandler):
    """Prepares reference/pool data and shifts for the tabular Adult domain."""

    def __init__(
        self,
        data_path: str | Path = "data/adult.data",
        reference_frac: float = 0.5,
        subgroup_col: str = "race",
        subgroup_value: str = "Black",
        random_state: int = 42,
        **_unused_kwargs,
    ):
        self.data_path = Path(data_path)
        self.reference_frac = reference_frac
        self.subgroup_col = subgroup_col
        self.subgroup_value = subgroup_value
        self.random_state = random_state

        self._encoder: OneHotEncoder | None = None
        self._X_ref_raw: pd.DataFrame | None = None
        self._X_pool_raw: pd.DataFrame | None = None
        self._y_pool: pd.Series | None = None

    def prepare_reference_and_pool(self) -> tuple[np.ndarray, np.ndarray]:
        dataframe = load_adult(self.data_path)
        df_ref, df_pool = train_test_split(
            dataframe, train_size=self.reference_frac, random_state=self.random_state
        )
        X_ref_raw = df_ref.drop(columns=["income"])
        y_pool = df_pool["income"]
        X_pool_raw = df_pool.drop(columns=["income"])

        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._encoder.fit(
            pd.concat([X_ref_raw[CATEGORICAL_COLS], X_pool_raw[CATEGORICAL_COLS]])
        )

        self._X_ref_raw = X_ref_raw
        self._X_pool_raw = X_pool_raw
        self._y_pool = y_pool

        self.X_ref_numeric = self._encode_dataframe(X_ref_raw)
        self.X_pool_numeric = self._encode_dataframe(X_pool_raw)
        return self.X_ref_numeric, self.X_pool_numeric

    def _encode_dataframe(self, dataframe: pd.DataFrame) -> np.ndarray:
        """One-hot encode categorical columns and return a float32 array."""
        numerical_values = dataframe.select_dtypes(include=[np.number])
        categorical_encoded = self._encoder.transform(dataframe[CATEGORICAL_COLS])
        return np.hstack([numerical_values.values, categorical_encoded]).astype(np.float32)

    def generate_shifted_test_set(self, alpha: float, seed: int):
        return generate_adult_demographic_shift(
            self._X_pool_raw,
            self._y_pool,
            subgroup_col=self.subgroup_col,
            subgroup_value=self.subgroup_value,
            alpha=alpha,
            random_state=seed,
        )

    def build_numeric_test_view(self, X_test_shifted: pd.DataFrame) -> np.ndarray:
        return self._encode_dataframe(X_test_shifted)

    def build_evidently_test_view(self, X_test_shifted: pd.DataFrame) -> pd.DataFrame:
        return X_test_shifted

    @property
    def reference_evidently(self) -> pd.DataFrame:
        return self._X_ref_raw
