"""
src/evaluation/__init__.py

Evaluation metrics and visualization for ShiftBench.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .metrics import compute_detection_metrics, compute_threshold

__all__ = ["compute_detection_metrics", "compute_threshold"]
