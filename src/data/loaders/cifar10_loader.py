"""
cifar10_loader.py

Loader for precomputed CIFAR-10 ResNet18 embeddings.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
from pathlib import Path

import numpy as np


def load_cifar10_embeddings(
    embeddings_dir: str | Path = "embeddings/cifar10",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed CIFAR-10 embeddings and labels.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays.
    """
    embeddings_dir = Path(embeddings_dir)
    train_embeddings_path = embeddings_dir / "train_embeddings.npy"
    train_labels_path = embeddings_dir / "train_labels.npy"
    test_embeddings_path = embeddings_dir / "test_embeddings.npy"
    test_labels_path = embeddings_dir / "test_labels.npy"

    if not train_embeddings_path.exists() or not test_embeddings_path.exists():
        raise FileNotFoundError(
            "CIFAR-10 embeddings not found. Run scripts/extract_cifar10_embeddings.py first."
        )

    X_train = np.load(train_embeddings_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_embeddings_path)
    y_test = np.load(test_labels_path)

    return X_train, y_train, X_test, y_test
