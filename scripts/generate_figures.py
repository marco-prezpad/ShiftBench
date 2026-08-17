#!/usr/bin/env python3
"""
generate_figures.py

Generate benchmark figures for a given domain.

Usage:
    python scripts/generate_figures.py --domain adult|cifar10c

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShiftBench figures.")
    parser.add_argument("--domain", default="adult", choices=["adult", "cifar10c"])
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]

    if args.domain == "adult":
        results_dir = Path(config["datasets"]["adult"]["results_dir"])
    elif args.domain == "cifar10c":
        results_dir = Path(config["images"]["results_dir"])
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

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
        det: {float(a): v for a, v in alpha_dict.items()}
        for det, alpha_dict in raw_scores.items()
    }

    sig_level = exp["significance_level"]

    plot_all(metrics, scores, figures_dir, significance_level=sig_level)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()