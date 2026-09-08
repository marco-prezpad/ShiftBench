"""
series_generator.py

Synthetic time series generator (sine vs. square wave classes) for the
time series domain.

Author: Marco Pérez Padilla
Date:   12-08-2026
"""

import numpy as np


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
    time_axis = np.linspace(0, 1, length)

    series_list = []
    label_list = []
    half = n_samples // 2

    for _ in range(half):
        phase = rng.uniform(0, 2 * np.pi)
        series_list.append(np.sin(2 * np.pi * 3 * time_axis + phase))
        label_list.append(0)

        phase = rng.uniform(0, 2 * np.pi)
        series_list.append(np.sign(np.sin(2 * np.pi * 3 * time_axis + phase)))
        label_list.append(1)

    if n_samples % 2 == 1:
        if rng.random() < 0.5:
            series_list.append(np.sin(2 * np.pi * 3 * time_axis + rng.uniform(0, 2 * np.pi)))
            label_list.append(0)
        else:
            series_list.append(np.sign(np.sin(2 * np.pi * 3 * time_axis + rng.uniform(0, 2 * np.pi))))
            label_list.append(1)

    return np.array(series_list, dtype=np.float32), np.array(label_list)
