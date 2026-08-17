"""
test_image_shift.py

Tests for the image domain shift generator (class mixture).

Author: Marco Pérez Padilla
Date:   16-08-2026
"""
import numpy as np
import pytest

from src.data.synthetic_shifts import generate_class_mixture_shift


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    y = np.array([0] * 50 + [1] * 50)
    return X, y


class TestGenerateClassMixtureShift:
    def test_alpha_zero_only_class_a(self, sample_data):
        X, y = sample_data
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=0.0, class_a=0, class_b=1, random_state=42
        )
        assert set(y_test) == {0}

    def test_alpha_one_only_class_b(self, sample_data):
        X, y = sample_data
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=1.0, class_a=0, class_b=1, random_state=42
        )
        assert set(y_test) == {1}

    def test_alpha_half_mixture(self, sample_data):
        X, y = sample_data
        X_test, y_test = generate_class_mixture_shift(
            X, y, alpha=0.5, class_a=0, class_b=1, random_state=42
        )
        unique, counts = np.unique(y_test, return_counts=True)
        assert set(unique) == {0, 1}
        assert 40 < counts[0] < 60  # ~50% cada clase

    def test_reproducible_with_seed(self, sample_data):
        X, y = sample_data
        X1, _ = generate_class_mixture_shift(
            X, y, alpha=0.5, class_a=0, class_b=1, random_state=7
        )
        X2, _ = generate_class_mixture_shift(
            X, y, alpha=0.5, class_a=0, class_b=1, random_state=7
        )
        assert np.array_equal(X1, X2)

    def test_alpha_out_of_range_raises(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError, match="alpha must be in"):
            generate_class_mixture_shift(
                X, y, alpha=1.5, class_a=0, class_b=1
            )