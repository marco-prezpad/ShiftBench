"""
load.py

Dataset loading functions.
It provides a function to load the UCI Adult dataset, ensuring that 
the data is cleaned and formatted correctly for analysis.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
from pathlib import Path

import pandas as pd


COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

NUMERICAL_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def load_adult(path: str | Path) -> pd.DataFrame:
    """Load the UCI Adult dataset and return a cleaned DataFrame.

    Args:
        path: Path to the adult.data file.

    Returns:
        DataFrame with named columns, stripped strings, and no missing values.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
    )

    for col in CATEGORICAL_COLS:
        df[col] = df[col].str.strip()

    df = df.dropna().reset_index(drop=True)

    return df