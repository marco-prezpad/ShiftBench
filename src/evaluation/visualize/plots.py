"""
plots.py

Figure-generation functions for ShiftBench results.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .detector_styles import DETECTOR_STYLES

sns.set_theme(style="whitegrid")


def plot_all(
    metrics: pd.DataFrame,
    scores: dict,
    figures_dir: Path,
    significance_level: float = 0.05,
) -> None:
    """Generate all standard figures for the benchmark."""
    plot_tpr_vs_alpha(metrics, figures_dir)
    plot_auc_tpr_bars(metrics, figures_dir)
    plot_score_distribution(scores, figures_dir)
    plot_score_distribution_no_kl(scores, figures_dir)   
    plot_fpr_calibration(metrics, significance_level, figures_dir)


def plot_tpr_vs_alpha(metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Plot TPR vs alpha for each detector."""
    plt.figure(figsize=(8, 5))
    for detector_name in metrics["detector"].unique():
        detector_metrics = metrics[metrics["detector"] == detector_name].sort_values("alpha")
        style = DETECTOR_STYLES.get(detector_name, {})
        plt.plot(
            detector_metrics["alpha"],
            detector_metrics["TPR"],
            label=detector_name,
            **style,
            linewidth=2,
            markersize=6,
        )
    plt.xlabel("Alpha (shift intensity)")
    plt.ylabel("TPR (sensitivity)")
    plt.title("Detection sensitivity vs shift intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "tpr_vs_alpha.png", dpi=150)
    plt.close()


def plot_auc_tpr_bars(metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart of AUC_TPR per detector."""
    auc_by_detector = metrics.groupby("detector")["AUC_TPR"].first().sort_values()
    plt.figure(figsize=(8, 4))
    auc_by_detector.plot(kind="barh", color="steelblue")
    plt.xlabel("AUC_TPR")
    plt.title("Area under TPR vs alpha curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "auc_tpr_bars.png", dpi=150)
    plt.close()


def plot_score_distribution(scores: dict, figures_dir: Path) -> None:
    """Boxplot of H0 scores for all detectors."""
    h0_score_frames = []
    for detector_name, alpha_scores in scores.items():
        if 0.0 in alpha_scores:
            h0_score_frames.append(
                pd.DataFrame({"detector": detector_name, "score": alpha_scores[0.0]})
            )
    if not h0_score_frames:
        return
    h0_scores_df = pd.concat(h0_score_frames, ignore_index=True)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=h0_scores_df, x="detector", y="score", palette="Set2")
    plt.title("Score distribution under H0 (alpha=0)")
    plt.tight_layout()
    plt.savefig(figures_dir / "h0_scores.png", dpi=150)
    plt.close()


def plot_score_distribution_no_kl(scores: dict, figures_dir: Path) -> None:
    """Boxplot of H0 scores for all detectors except KL.

    KL often produces scores on a much larger scale than the other
    detectors, which can hide their variability. This additional figure
    zooms in on the remaining detectors.
    """
    h0_score_frames = []
    for detector_name, alpha_scores in scores.items():
        # Skip KL to avoid scaling issues
        if detector_name == "kl":
            continue
        if 0.0 in alpha_scores:
            h0_score_frames.append(
                pd.DataFrame({"detector": detector_name, "score": alpha_scores[0.0]})
            )
    if not h0_score_frames:
        return

    h0_scores_df = pd.concat(h0_score_frames, ignore_index=True)

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=h0_scores_df, x="detector", y="score", palette="Set2")
    plt.title("Score distribution under H0 (alpha=0) - excluding KL")
    plt.tight_layout()
    plt.savefig(figures_dir / "h0_scores_no_kl.png", dpi=150)
    plt.close()


def plot_fpr_calibration(
    metrics: pd.DataFrame,
    significance_level: float,
    figures_dir: Path,
) -> None:
    """Empirical FPR vs nominal significance level."""
    fpr_at_h0 = metrics[metrics["alpha"] == 0.0][["detector", "FPR"]].copy()
    if fpr_at_h0.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.barplot(data=fpr_at_h0, x="detector", y="FPR", hue="detector", palette="Set2", legend=False)
    plt.axhline(
        significance_level,
        color="red",
        linestyle="--",
        label=f"Nominal α={significance_level}",
    )
    plt.title("Empirical FPR under H0 (alpha=0)")
    plt.ylabel("False Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fpr_calibration.png", dpi=150)
    plt.close()
