"""
src/data/__init__.py

Data loading and synthetic shift generation modules.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .loaders import CATEGORICAL_COLS, COLUMN_NAMES, NUMERICAL_COLS, load_adult, load_cifar10_embeddings
from .shifts import (
    generate_adult_demographic_shift,
    generate_class_mixture_shift,
    generate_synthetic_series,
)

__all__ = [
    "CATEGORICAL_COLS",
    "COLUMN_NAMES",
    "NUMERICAL_COLS",
    "generate_adult_demographic_shift",
    "generate_class_mixture_shift",
    "generate_synthetic_series",
    "load_adult",
    "load_cifar10_embeddings",
]
