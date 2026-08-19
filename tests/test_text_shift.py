"""
test_text_shift.py

Tests for the text domain shift generator.

Author: Marco Pérez Padilla
Date:   19-08-2026
"""
import numpy as np

from src.data.synthetic_shifts import generate_class_mixture_shift


def test_alpha_zero_only_class_a():
    X = np.random.rand(100, 50).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    X_test, y_test = generate_class_mixture_shift(
        X, y, alpha=0.0, class_a=0, class_b=1, random_state=42
    )
    assert set(y_test) == {0}

def test_alpha_one_only_class_b():
    X = np.random.rand(100, 50).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    X_test, y_test = generate_class_mixture_shift(
        X, y, alpha=1.0, class_a=0, class_b=1, random_state=42
    )
    assert set(y_test) == {1}

def test_alpha_half_mixture():
    X = np.random.rand(100, 50).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    X_test, y_test = generate_class_mixture_shift(
        X, y, alpha=0.5, class_a=0, class_b=1, random_state=42
    )
    unique, counts = np.unique(y_test, return_counts=True)
    assert set(unique) == {0, 1}
    assert 40 < counts[0] < 60