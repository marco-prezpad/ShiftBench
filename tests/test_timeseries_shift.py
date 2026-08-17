"""
test_timeseries_shift.py

Tests suite for the time series shift generation functions.

Author: Marco Pérez Padilla
Date:   17-08-2026
"""

import numpy as np

from src.data.synthetic_shifts import (
    generate_class_mixture_shift,
    generate_synthetic_series,
)


class TestTimeSeriesShift:
    def test_generator_returns_two_classes(self):
        X, y = generate_synthetic_series(n_samples=100, random_state=42)
        assert X.shape == (100, 50)
        assert set(y) == {0, 1}

    def test_shift_alpha_zero_only_class_a(self):
        X, y = generate_synthetic_series(n_samples=100, random_state=42)
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=0.0, class_a=0, class_b=1, random_state=42
        )
        assert set(y_test) == {0}

    def test_shift_alpha_one_only_class_b(self):
        X, y = generate_synthetic_series(n_samples=100, random_state=42)
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=1.0, class_a=0, class_b=1, random_state=42
        )
        assert set(y_test) == {1}

    def test_shift_alpha_half_mixture(self):
        X, y = generate_synthetic_series(n_samples=100, random_state=42)
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=0.5, class_a=0, class_b=1, random_state=42
        )
        unique, counts = np.unique(y_test, return_counts=True)
        assert set(unique) == {0, 1}
        assert 40 < counts[0] < 60