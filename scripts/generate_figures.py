#!/usr/bin/env python3
"""
generate_figures.py

Generate all benchmark figures from results/metrics.csv and results/scores.json.

Usage:
    python scripts/generate_figures.py

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.visualize import plot_all


def main() -> None:
    results_dir = Path("results")
    metrics_path = results_dir / "metrics.csv"
    scores_path = results_dir / "scores.json"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists() or not scores_path.exists():
        print("No results found. Run the benchmark first (python scripts/run_experiments.py).")
        return

    metrics = pd.read_csv(metrics_path)
    with open(scores_path) as f:
        raw_scores = json.load(f)
    scores = {
        det: {float(a): v for a, v in alpha_dict.items()}
        for det, alpha_dict in raw_scores.items()
    }

    from src.utils.io import load_yaml
    config = load_yaml("configs/config.yaml")
    sig_level = config["experiment"]["significance_level"]

    plot_all(metrics, scores, figures_dir, significance_level=sig_level)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()