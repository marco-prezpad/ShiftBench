"""
src/evaluation/__init__.py

Evaluation protocol, metrics, and visualization for ShiftBench.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .protocol import BenchmarkProtocol
from .metrics import compute_detection_metrics, compute_threshold

__all__ = ["BenchmarkProtocol", "compute_detection_metrics", "compute_threshold"]