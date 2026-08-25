"""
class_mixture_shift.py

Synthetic class-mixture shift generator, shared by the image, time series,
and text domains.

Author: Marco Pérez Padilla
Date:   12-08-2026
"""

import numpy as np


def generate_class_mixture_shift(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    class_a: int,
    class_b: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a test set with controlled mixture of two classes.

    alpha=0.0 -> all samples from class_a (no shift).
    alpha=1.0 -> all samples from class_b (maximum shift).
    Intermediate alphas linearly interpolate the proportion of class_b.

    Bootstrap resamples within each class to ensure variability.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    rng = np.random.default_rng(random_state)
    total_size = len(X)

    idx_class_a = np.where(y == class_a)[0]
    idx_class_b = np.where(y == class_b)[0]

    if len(idx_class_a) == 0 or len(idx_class_b) == 0:
        bootstrap_idx = rng.choice(total_size, size=total_size, replace=True)
        return X[bootstrap_idx].copy(), y[bootstrap_idx].copy()

    n_class_b = int(round(alpha * total_size))
    n_class_a = total_size - n_class_b

    chosen_class_a = rng.choice(idx_class_a, size=n_class_a, replace=True)
    chosen_class_b = rng.choice(idx_class_b, size=n_class_b, replace=True)

    test_idx = np.concatenate([chosen_class_a, chosen_class_b])
    rng.shuffle(test_idx)

    return X[test_idx].copy(), y[test_idx].copy()
