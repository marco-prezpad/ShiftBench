"""
src/data/loaders/__init__.py

Dataset loading functions, one module per data source.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .adult_loader import CATEGORICAL_COLS, COLUMN_NAMES, NUMERICAL_COLS, load_adult
from .cifar10_loader import load_cifar10_embeddings

__all__ = [
    "CATEGORICAL_COLS",
    "COLUMN_NAMES",
    "NUMERICAL_COLS",
    "load_adult",
    "load_cifar10_embeddings",
]
