"""
adult_loader.py

Loader for the UCI Adult dataset.
Ensures the data is cleaned and formatted correctly for analysis.

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
    dataframe = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
    )

    for column in CATEGORICAL_COLS:
        dataframe[column] = dataframe[column].str.strip()

    dataframe = dataframe.dropna().reset_index(drop=True)

    return dataframe
