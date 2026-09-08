#!/usr/bin/env python3
"""
paper_figures.py

Generate the two main figures for the ShiftBench paper:
  1. Combined TPR vs alpha (2x2 grid) for all four domains.
  2. Non-monotonic score behaviour for KL and Embedding in problematic domains.

Usage:
    python scripts/paper_figures.py [--results-root results]

Author: Marco Pérez Padilla
Date:   08-09-2026
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DOMAINS = ["adult", "cifar10", "timeseries", "text"]
DOMAIN_TITLES = {
    "adult": "UCI Adult",
    "cifar10": "CIFAR-10",
    "timeseries": "Time series",
    "text": "Text",
}

NONMONO_CASES = [
    ("cifar10", "kl", "KL CIFAR-10"),
    ("text", "kl", "KL Text"),
    ("cifar10", "embedding", "Embedding CIFAR-10"),
]

DETECTOR_COLORS = {
    "mmd": "tab:blue",
    "lsdd": "tab:orange",
    "kl": "tab:green",
    "embedding": "tab:red",
    "evidently": "tab:purple",
}

DETECTOR_LABELS = {
    "mmd": "MMD",
    "lsdd": "LSDD",
    "kl": "KL",
    "embedding": "Embedding",
    "evidently": "Evidently",
}


def load_metrics(results_root: Path, domain: str) -> pd.DataFrame:
    """Load metrics.csv for a single domain."""
    path = results_root / domain / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def plot_tpr_combined(results_root: Path, output_path: Path) -> None:
    """Create 2x2 TPR vs alpha figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, domain in zip(axes, DOMAINS):
        df = load_metrics(results_root, domain)
        for det in df["detector"].unique():
            sub = df[df["detector"] == det].sort_values("alpha")
            ax.plot(
                sub["alpha"],
                sub["TPR"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=DETECTOR_COLORS.get(det, "black"),
                label=DETECTOR_LABELS.get(det, det),
            )
        ax.set_title(DOMAIN_TITLES[domain])
        ax.set_xlabel(r"Shift intensity $\alpha$")
        ax.set_ylabel("TPR")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"OK TPR combined figure saved to {output_path}")


def plot_nonmonotonic(results_root: Path, output_path: Path) -> None:
    """Create 1x3 figure showing non-monotonic score behaviour."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (domain, det, title) in zip(axes, NONMONO_CASES):
        df = load_metrics(results_root, domain)
        sub = df[df["detector"] == det].sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["mean_score"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="tab:red",
        )
        ax.set_title(title)
        ax.set_xlabel(r"Shift intensity $\alpha$")
        ax.set_ylabel("Mean score")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"OK non-monotonic figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShiftBench paper figures.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing per-domain results (default: results)",
    )
    args = parser.parse_args()

    figures_dir = args.results_root / "figures"
    plot_tpr_combined(args.results_root, figures_dir / "tpr_vs_alpha_all_domains.png")
    plot_nonmonotonic(args.results_root, figures_dir / "nonmonotonic_scores.pdf")


if __name__ == "__main__":
    main()