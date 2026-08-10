"""
test_load.py

Test suite for the load module.
It tests the load_adult function to ensure it correctly loads the 
Adult dataset and performs basic data integrity checks.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.load import CATEGORICAL_COLS, COLUMN_NAMES, NUMERICAL_COLS, load_adult


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def adult_path():
    """Path to adult.data, skipping test if not downloaded."""
    path = DATA_DIR / "adult.data"
    if not path.exists():
        pytest.skip("adult.data not found. Run: python scripts/download_datasets.py")
    return path


class TestLoadAdult:
    def test_returns_dataframe(self, adult_path):
        df = load_adult(adult_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_columns(self, adult_path):
        df = load_adult(adult_path)
        assert list(df.columns) == COLUMN_NAMES

    def test_no_missing_values(self, adult_path):
        df = load_adult(adult_path)
        assert df.isnull().sum().sum() == 0

    def test_no_spaces_in_categorical(self, adult_path):
        df = load_adult(adult_path)
        for col in CATEGORICAL_COLS:
            for val in df[col].unique():
                assert val == str(val).strip()

    def test_minimum_rows(self, adult_path):
        df = load_adult(adult_path)
        assert len(df) > 30000

    def test_income_binary(self, adult_path):
        df = load_adult(adult_path)
        assert set(df["income"].unique()) == {"<=50K", ">50K"}

    def test_numerical_cols_are_numeric(self, adult_path):
        df = load_adult(adult_path)
        for col in NUMERICAL_COLS:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_categorical_cols_are_strings(self, adult_path):
        df = load_adult(adult_path)
        for col in CATEGORICAL_COLS:
            assert pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_adult("/nonexistent/path.csv")