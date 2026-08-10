"""
metrics.py

Metrics computation for ShiftBench evaluation protocol.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
import pandas as pd


def compute_threshold(scores_h0: np.ndarray, percentile: float = 95.0) -> float:
    """Compute detection threshold as a percentile of the H0 scores.

    Args:
        scores_h0: Array of scores under no-shift (alpha=0).
        percentile: Percentile to use (default 95).

    Returns:
        Threshold value.
    """
    return float(np.percentile(scores_h0, percentile))


def compute_metrics_for_alpha(
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Compute binary classification metrics for a single alpha.

    Args:
        scores: Array of scores for this alpha.
        threshold: Decision threshold.

    Returns:
        Dictionary with TPR, FPR, Precision, F1, mean_score, std_score.
    """
    preds = (scores >= threshold).astype(int)
    # All samples are "positive" if alpha > 0
    tp = preds.sum()
    fp = len(preds) - tp  

    tpr = tp / len(preds) if len(preds) > 0 else 0.0
    fpr = fp / len(preds) if len(preds) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def compute_detection_metrics(
    scores_dict: dict[str, dict[float, list[float]]],
    alphas: list[float],
    n_bootstrap: int,
    threshold_percentile: float = 95.0,
) -> pd.DataFrame:
    """Compute all metrics for every detector and alpha.

    Args:
        scores_dict: Nested dict detector_name -> alpha -> list of scores.
        alphas: List of alpha values (first must be 0.0).
        n_bootstrap: Number of bootstrap runs (used to compute TNR).
        threshold_percentile: Percentile for threshold on H0 scores.

    Returns:
        DataFrame with columns: detector, alpha, TPR, FPR, Precision, F1,
        mean_score, std_score, and global metrics AUC_TPR, TNR, etc.
    """
    rows = []
    global_metrics = {}

    for detector_name, alpha_scores in scores_dict.items():
        # Threshold from H0 (alpha=0)
        h0_scores = np.array(alpha_scores.get(0.0, []))
        if len(h0_scores) == 0:
            continue
        threshold = compute_threshold(h0_scores, threshold_percentile)

        tpr_list = []
        for alpha in alphas:
            scores = np.array(alpha_scores.get(alpha, []))
            if len(scores) == 0:
                continue
            metrics = compute_metrics_for_alpha(scores, threshold)
            rows.append(
                {
                    "detector": detector_name,
                    "alpha": alpha,
                    **metrics,
                }
            )
            if alpha > 0.0:
                tpr_list.append(metrics["TPR"])

        # Global metrics per detector
        tnr = 1.0 - float(np.mean(np.array(h0_scores) >= threshold))
        auc_tpr = (
            np.trapz(tpr_list, [a for a in alphas if a > 0.0])
            / (max(alphas) - min(alphas))
            if len(tpr_list) > 1
            else 0.0
        )
        alpha_at_80 = None
        for alpha, tpr in zip([a for a in alphas if a > 0.0], tpr_list):
            if tpr >= 0.8:
                alpha_at_80 = alpha
                break
        tpr_at_05 = tpr_list[len(tpr_list) // 2] if tpr_list else 0.0

        global_metrics[detector_name] = {
            "TNR": tnr,
            "AUC_TPR": auc_tpr,
            "alpha@80%TPR": alpha_at_80,
            "TPR@alpha=0.5": tpr_at_05,
        }

    df = pd.DataFrame(rows)

    for detector_name, gm in global_metrics.items():
        df.loc[df["detector"] == detector_name, "TNR"] = gm["TNR"]
        df.loc[df["detector"] == detector_name, "AUC_TPR"] = gm["AUC_TPR"]
        df.loc[df["detector"] == detector_name, "alpha@80%TPR"] = gm["alpha@80%TPR"]
        df.loc[df["detector"] == detector_name, "TPR@alpha=0.5"] = gm["TPR@alpha=0.5"]

    return df


def compute_auc_tpr(tpr_array: list[float], alphas: list[float]) -> float:
    """Compute area under the TPR vs alpha curve (normalized)."""
    if len(tpr_array) < 2:
        return 0.0
    return float(np.trapz(tpr_array, alphas) / (max(alphas) - min(alphas)))


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC-ROC treating scores as confidence of positive class.

    Args:
        scores: 1D array of scores.
        labels: Binary array (1 for alpha>0, 0 for alpha=0).

    Returns:
        AUC value.
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))