"""
synthetic_shifts.py

Generate synthetic distribution shifts of controlled intensity.

Author: Marco Pérez Padilla
Date:   12-08-2026
"""

import numpy as np
import pandas as pd


def generate_adult_demographic_shift(
    X: pd.DataFrame,
    y: pd.Series,
    subgroup_col: str,
    subgroup_value: str,
    alpha: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a test set with controlled shift intensity for a demographic subgroup.

    alpha = 0.0 -> baseline (original) subgroup proportion (no shift).
    alpha = 1.0 -> 100% of test samples belong to the subgroup (maximum shift).
    Intermediate alphas linearly interpolate the target proportion between
    baseline and 1.0.

    If subgroup_value is not found, returns a bootstrap sample of the full pool.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    total_size = len(X)
    mask = X[subgroup_col] == subgroup_value
    idx_subgroup = X.index[mask].values
    idx_others = X.index[~mask].values

    if len(idx_subgroup) == 0:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.index, size=total_size, replace=True)
        return X.loc[idx].copy(), y.loc[idx].copy()

    baseline_prop = mask.mean()
    target_prop = baseline_prop + alpha * (1.0 - baseline_prop)
    n_subgroup = int(round(target_prop * total_size))
    n_others = total_size - n_subgroup

    rng = np.random.default_rng(random_state)

    chosen_sub = rng.choice(idx_subgroup, size=n_subgroup, replace=True)
    chosen_others = rng.choice(idx_others, size=n_others, replace=True)

    test_idx = np.concatenate([chosen_sub, chosen_others])
    rng.shuffle(test_idx)

    return X.loc[test_idx].copy(), y.loc[test_idx].copy()


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

    # Índices por clase
    idx_a = np.where(y == class_a)[0]
    idx_b = np.where(y == class_b)[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        boot_idx = rng.choice(total_size, size=total_size, replace=True)
        return X[boot_idx].copy(), y[boot_idx].copy()

    n_b = int(round(alpha * total_size))
    n_a = total_size - n_b

    chosen_a = rng.choice(idx_a, size=n_a, replace=True)
    chosen_b = rng.choice(idx_b, size=n_b, replace=True)

    test_idx = np.concatenate([chosen_a, chosen_b])
    rng.shuffle(test_idx)

    return X[test_idx].copy(), y[test_idx].copy()


def generate_synthetic_series(
    n_samples: int = 1000,
    length: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series with two classes.

    Class 0: sine wave with fixed frequency.
    Class 1: square wave with same frequency.

    Each sample is a vector of length `length`.

    Returns:
        X: array of shape (n_samples, length)
        y: labels (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 1, length)

    X = []
    y = []
    half = n_samples // 2

    for _ in range(half):
        phase = rng.uniform(0, 2 * np.pi)
        X.append(np.sin(2 * np.pi * 3 * t + phase))
        y.append(0)

        phase = rng.uniform(0, 2 * np.pi)
        X.append(np.sign(np.sin(2 * np.pi * 3 * t + phase)))
        y.append(1)

    if n_samples % 2 == 1:
        if rng.random() < 0.5:
            X.append(np.sin(2 * np.pi * 3 * t + rng.uniform(0, 2 * np.pi)))
            y.append(0)
        else:
            X.append(np.sign(np.sin(2 * np.pi * 3 * t + rng.uniform(0, 2 * np.pi))))
            y.append(1)

    return np.array(X, dtype=np.float32), np.array(y)