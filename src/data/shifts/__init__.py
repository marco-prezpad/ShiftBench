"""
src/data/shifts/__init__.py

Synthetic distribution shift generators, one module per shift strategy.

Author: Marco Pérez Padilla
Date:   12-08-2026
"""

from .class_mixture_shift import generate_class_mixture_shift
from .demographic_shift import generate_adult_demographic_shift
from .series_generator import generate_synthetic_series

__all__ = [
    "generate_adult_demographic_shift",
    "generate_class_mixture_shift",
    "generate_synthetic_series",
]
