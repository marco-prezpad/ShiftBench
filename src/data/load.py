"""
load.py

Dataset loading functions.
It provides a function to load the UCI Adult dataset, ensuring that 
the data is cleaned and formatted correctly for analysis.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
from pathlib import Path

import numpy as np
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


def load_cifar10_embeddings(embeddings_dir: str | Path = "embeddings/cifar10") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed CIFAR-10 embeddings and labels.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays.
    """
    embeddings_dir = Path(embeddings_dir)
    train_path = embeddings_dir / "train_embeddings.npy"
    train_labels_path = embeddings_dir / "train_labels.npy"
    test_path = embeddings_dir / "test_embeddings.npy"
    test_labels_path = embeddings_dir / "test_labels.npy"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "CIFAR-10 embeddings not found. Run scripts/extract_cifar10_embeddings.py first."
        )

    X_train = np.load(train_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_path)
    y_test = np.load(test_labels_path)

    return X_train, y_train, X_test, y_test