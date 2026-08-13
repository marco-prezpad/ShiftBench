"""
test_synthetic_shifts.py

Test suite for the synthetic shift generation functions.
It tests the `generate_adult_demographic_shift` function to ensure it 
behaves as expected under various scenarios.

Author: Marco Pérez Padilla
Date:   12-08-2026
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_adult
from src.data.synthetic_shifts import generate_adult_demographic_shift


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def adult_df():
    """Load cleaned Adult dataset for shift tests."""
    path = DATA_DIR / "adult.data"
    if not path.exists():
        pytest.skip("adult.data not found. Run: python scripts/download_datasets.py")
    return load_adult(path)


class TestGenerateAdultDemographicShift:
    def test_returns_tuple_of_df_and_series(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, y_test = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5
        )
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

    def test_preserves_all_columns_except_target(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5
        )
        assert list(X_test.columns) == list(X.columns)

    def test_same_length_as_input(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, y_test = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5
        )
        assert len(X_test) == len(X)
        assert len(y_test) == len(y)

    def test_alpha_zero_original_proportion(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        original_pct = (X["race"] == "Black").mean()
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.0
        )
        test_pct = (X_test["race"] == "Black").mean()
        assert abs(test_pct - original_pct) < 0.02

    def test_alpha_one_all_subgroup(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=1.0
        )
        assert (X_test["race"] == "Black").all()

    def test_alpha_half_matches_interpolation(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        baseline = (X["race"] == "Black").mean()
        expected = baseline + 0.5 * (1.0 - baseline)
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5
        )
        pct = (X_test["race"] == "Black").mean()
        assert abs(pct - expected) < 0.02

    def test_indices_are_valid(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.7
        )
        assert X_test.index.min() >= 0
        assert X_test.index.max() < len(adult_df)

    def test_reproducible_with_seed(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X1, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5, random_state=42
        )
        X2, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Black", alpha=0.5, random_state=42
        )
        pd.testing.assert_frame_equal(X1, X2)

    def test_works_with_sex_column(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="sex", subgroup_value="Female", alpha=1.0
        )
        assert (X_test["sex"] == "Female").all()

    def test_invalid_subgroup_value_does_not_break(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        X_test, _ = generate_adult_demographic_shift(
            X, y, subgroup_col="race", subgroup_value="Martian", alpha=1.0
        )
        assert len(X_test) == len(X)

    def test_alpha_out_of_range_raises(self, adult_df):
        X = adult_df.drop(columns=["income"])
        y = adult_df["income"]
        with pytest.raises(ValueError, match="alpha must be in"):
            generate_adult_demographic_shift(
                X, y, subgroup_col="race", subgroup_value="Black", alpha=1.5
            )