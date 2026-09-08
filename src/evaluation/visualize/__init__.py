"""
src/evaluation/visualize/__init__.py

Figure generation for ShiftBench results.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""
from .detector_styles import DETECTOR_STYLES
from .plots import (
    plot_all,
    plot_auc_tpr_bars,
    plot_fpr_calibration,
    plot_score_distribution,
    plot_tpr_vs_alpha,
)

__all__ = [
    "DETECTOR_STYLES",
    "plot_all",
    "plot_auc_tpr_bars",
    "plot_fpr_calibration",
    "plot_score_distribution",
    "plot_tpr_vs_alpha",
]
