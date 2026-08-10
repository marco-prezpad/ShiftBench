"""
synthetic_shifts.py

Generate synthetic distribution shifts of controlled intensity.

Author: Marco Pérez Padilla
Date:   10-08-2026
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
    """Generate a test set with controlled proportion of a demographic subgroup.

    The output has the same number of rows as the input. When the subgroup
    is smaller than alpha * len(X), sampling with replacement is used.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        subgroup_col: Column name for the demographic feature.
        subgroup_value: Value of the subgroup to oversample.
        alpha: Proportion of the subgroup in the output (0.0 to 1.0).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_test, y_test) with the same shape as input.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    total_size = len(X)
    n_subgroup = int(round(alpha * total_size))
    n_others = total_size - n_subgroup

    mask = X[subgroup_col] == subgroup_value
    idx_subgroup = X.index[mask].values
    idx_others = X.index[~mask].values

    rng = np.random.default_rng(random_state)

    if n_subgroup > 0 and len(idx_subgroup) > 0:
        chosen_sub = rng.choice(idx_subgroup, size=n_subgroup, replace=True)
    elif n_subgroup > 0 and len(idx_subgroup) == 0:
        chosen_sub = np.array([], dtype=int)
        n_others = total_size
    else:
        chosen_sub = np.array([], dtype=int)

    if n_others > 0 and len(idx_others) > 0:
        chosen_others = rng.choice(idx_others, size=n_others, replace=True)
    else:
        chosen_others = np.array([], dtype=int)

    test_idx = np.concatenate([chosen_sub, chosen_others])
    rng.shuffle(test_idx)

    return X.loc[test_idx].copy(), y.loc[test_idx].copy()