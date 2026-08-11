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
        scores = json.load(f)

    plot_all(metrics, scores, figures_dir)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()