#!/usr/bin/env python3
"""
generate_figures.py

Generate benchmark figures for a given domain.

Usage:
    python scripts/generate_figures.py --domain adult|cifar10c|timeseries|text

Author: Marco Pérez Padilla
Date:   13-08-2026
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.visualize import plot_all
from src.utils.io import load_yaml

CONFIG_PATH = "configs/config.yaml"

RESULTS_DIR_BY_DOMAIN = {
    "adult": lambda config: config["datasets"]["adult"]["results_dir"],
    "cifar10c": lambda config: config["images"]["results_dir"],
    "timeseries": lambda config: config["timeseries"]["results_dir"],
    "text": lambda config: config["text"]["results_dir"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShiftBench figures.")
    parser.add_argument(
        "--domain", default="adult", choices=["adult", "cifar10c", "timeseries", "text"]
    )
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)
    experiment_config = config["experiment"]

    results_dir = Path(RESULTS_DIR_BY_DOMAIN[args.domain](config))

    metrics_path = results_dir / "metrics.csv"
    scores_path = results_dir / "scores.json"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists() or not scores_path.exists():
        print(f"No results found in {results_dir}. Run the benchmark first.")
        return

    metrics = pd.read_csv(metrics_path)
    with open(scores_path) as f:
        raw_scores = json.load(f)
    scores = {
        detector_name: {float(a): v for a, v in alpha_scores.items()}
        for detector_name, alpha_scores in raw_scores.items()
    }

    significance_level = experiment_config["significance_level"]

    plot_all(metrics, scores, figures_dir, significance_level=significance_level)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()