"""
demographic_shift.py

Synthetic demographic shift generator for the tabular (Adult) domain.

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
    subgroup_mask = X[subgroup_col] == subgroup_value
    idx_subgroup = X.index[subgroup_mask].values
    idx_others = X.index[~subgroup_mask].values

    if len(idx_subgroup) == 0:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.index, size=total_size, replace=True)
        return X.loc[idx].copy(), y.loc[idx].copy()

    baseline_prop = subgroup_mask.mean()
    target_prop = baseline_prop + alpha * (1.0 - baseline_prop)
    n_subgroup = int(round(target_prop * total_size))
    n_others = total_size - n_subgroup

    rng = np.random.default_rng(random_state)

    chosen_subgroup = rng.choice(idx_subgroup, size=n_subgroup, replace=True)
    chosen_others = rng.choice(idx_others, size=n_others, replace=True)

    test_idx = np.concatenate([chosen_subgroup, chosen_others])
    rng.shuffle(test_idx)

    return X.loc[test_idx].copy(), y.loc[test_idx].copy()
