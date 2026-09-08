"""
test_visualize.py

Tests for visualization functions.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.evaluation.visualize import plot_all


def test_plot_all_creates_figures():
    """Verify that plot_all generates the expected PNG files."""
    metrics = pd.DataFrame({
        "detector": ["mmd", "mmd", "lsdd", "lsdd"],
        "alpha": [0.0, 0.5, 0.0, 0.5],
        "TPR": [0.05, 0.8, 0.05, 0.7],
        "FPR": [0.0, 0.2, 0.0, 0.3],
        "Precision": [0.0, 0.8, 0.0, 0.7],
        "F1": [0.0, 0.8, 0.0, 0.7],
        "mean_score": [0.1, 0.6, 0.1, 0.5],
        "std_score": [0.01, 0.1, 0.01, 0.1],
        "TNR": [0.95, 0.95, 0.95, 0.95],
        "AUC_TPR": [0.8, 0.8, 0.7, 0.7],
        "alpha@80%TPR": [0.5, 0.5, 0.6, 0.6],
        "TPR@alpha=0.5": [0.8, 0.8, 0.7, 0.7],
    })
    scores = {
        "mmd": {0.0: [0.1, 0.12, 0.11], 0.5: [0.6, 0.7, 0.65]},
        "lsdd": {0.0: [0.1, 0.12, 0.11], 0.5: [0.5, 0.55, 0.6]},
    }

    with tempfile.TemporaryDirectory() as tmp:
        figures_dir = Path(tmp) / "figures"
        figures_dir.mkdir()
        plot_all(metrics, scores, figures_dir)

        assert (figures_dir / "tpr_vs_alpha.png").exists()
        assert (figures_dir / "auc_tpr_bars.png").exists()
        assert (figures_dir / "h0_scores.png").exists()
