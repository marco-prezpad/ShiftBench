"""
src/data/__init__.py

Data loading and preprocessing modules.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .load import CATEGORICAL_COLS, COLUMN_NAMES, NUMERICAL_COLS, load_adult

__all__ = ["CATEGORICAL_COLS", "COLUMN_NAMES", "NUMERICAL_COLS", "load_adult"]