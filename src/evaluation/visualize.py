"""
visualize.py

Visualization functions for ShiftBench results.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_all(
    metrics: pd.DataFrame,
    scores: dict,
    figures_dir: Path,
) -> None:
    """Generate all standard figures for the benchmark."""
    plot_tpr_vs_alpha(metrics, figures_dir)
    plot_auc_tpr_bars(metrics, figures_dir)
    plot_score_distribution(scores, figures_dir)


def plot_tpr_vs_alpha(metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Plot TPR vs alpha for each detector."""
    plt.figure(figsize=(8, 5))
    for detector in metrics["detector"].unique():
        subset = metrics[metrics["detector"] == detector]
        plt.plot(subset["alpha"], subset["TPR"], marker="o", label=detector)
    plt.xlabel("Alpha (shift intensity)")
    plt.ylabel("TPR (sensitivity)")
    plt.title("Detection sensitivity vs shift intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "tpr_vs_alpha.png", dpi=150)
    plt.close()


def plot_auc_tpr_bars(metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart of AUC_TPR per detector."""
    aucs = metrics.groupby("detector")["AUC_TPR"].first().sort_values()
    plt.figure(figsize=(8, 4))
    aucs.plot(kind="barh", color="steelblue")
    plt.xlabel("AUC_TPR")
    plt.title("Area under TPR vs alpha curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "auc_tpr_bars.png", dpi=150)
    plt.close()


def plot_score_distribution(scores: dict, figures_dir: Path) -> None:
    """Boxplot of H0 scores for all detectors."""
    h0_data = []
    for name, alpha_scores in scores.items():
        if 0.0 in alpha_scores:
            h0_data.append(pd.DataFrame({"detector": name, "score": alpha_scores[0.0]}))
    if not h0_data:
        return
    df = pd.concat(h0_data, ignore_index=True)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="detector", y="score", palette="Set2")
    plt.title("Score distribution under H0 (alpha=0)")
    plt.tight_layout()
    plt.savefig(figures_dir / "h0_scores.png", dpi=150)
    plt.close()