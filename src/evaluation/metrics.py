"""
metrics.py

Metrics computation for ShiftBench evaluation protocol.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""
import numpy as np
import pandas as pd


def compute_threshold(scores_h0: np.ndarray, percentile: float = 95.0) -> float:
    """Compute detection threshold as a percentile of the H0 scores."""
    return float(np.percentile(scores_h0, percentile))


def compute_metrics_for_alpha(
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Binary classification metrics for a single alpha.

    Note:
        For alpha > 0, TPR = proportion of scores >= threshold (true positives).
        For alpha = 0, this same proportion actually represents the FPR.
        The caller must handle the special case alpha=0 separately.
    """
    preds = (scores > threshold).astype(int)
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

    The AUC‑TPR includes α=0, and TPR@α=0.5 uses linear interpolation.
    """
    rows = []
    global_metrics = {}

    for detector_name, alpha_scores in scores_dict.items():
        h0_scores = np.array(alpha_scores.get(0.0, []))
        if len(h0_scores) == 0:
            continue
        threshold = compute_threshold(h0_scores, threshold_percentile)

        alpha_vals, tpr_vals = [], []
        for alpha in alphas:
            scores = np.array(alpha_scores.get(alpha, []))
            if len(scores) == 0:
                continue
            m = compute_metrics_for_alpha(scores, threshold)

            # Correct FPR for alpha=0 (the "flagged" samples are false positives)
            if alpha == 0.0:
                m["FPR"] = m["TPR"]          # real FPR under H0
                m["TPR"] = 0.0                # set TPR=0 at alpha=0 for the curve
            rows.append({"detector": detector_name, "alpha": alpha, **m})
            alpha_vals.append(alpha)
            tpr_vals.append(m["TPR"])

        # Global metrics per detector
        tnr = 1.0 - float(np.mean(h0_scores > threshold))
        # AUC‑TPR including alpha=0 (TPR=0 at alpha=0)
        auc_tpr = (
            np.trapz(tpr_vals, alpha_vals) / (max(alpha_vals) - min(alpha_vals))
            if len(tpr_vals) > 1 else 0.0
        )
        # alpha at 80% TPR (first alpha>0 where TPR>=0.8)
        alpha_at_80 = next(
            (a for a, t in zip(alpha_vals, tpr_vals) if a > 0.0 and t >= 0.8), None
        )
        # TPR@alpha=0.5 via linear interpolation
        tpr_at_05 = (
            float(np.interp(0.5, alpha_vals, tpr_vals))
            if len(tpr_vals) > 1 else 0.0
        )
        # AUC‑ROC (continuous sweep, using all scores vs binary label: α=0 vs α>0)
        try:
            from sklearn.metrics import roc_auc_score
            all_scores = []
            all_labels = []
            for alpha in alphas:
                sc = alpha_scores.get(alpha, [])
                all_scores.extend(sc)
                all_labels.extend([1 if alpha > 0 else 0] * len(sc))
            auc_roc = (
                float(roc_auc_score(all_labels, all_scores))
                if len(set(all_labels)) > 1 else 0.5
            )
        except Exception:
            auc_roc = 0.5

        global_metrics[detector_name] = {
            "TNR": tnr,
            "AUC_TPR": auc_tpr,
            "alpha@80%TPR": alpha_at_80,
            "TPR@alpha=0.5": tpr_at_05,
            "AUC_ROC": auc_roc,
        }

    df = pd.DataFrame(rows)
    for det, gm in global_metrics.items():
        mask = df["detector"] == det
        df.loc[mask, "TNR"] = gm["TNR"]
        df.loc[mask, "AUC_TPR"] = gm["AUC_TPR"]
        df.loc[mask, "alpha@80%TPR"] = gm["alpha@80%TPR"]
        df.loc[mask, "TPR@alpha=0.5"] = gm["TPR@alpha=0.5"]
        df.loc[mask, "AUC_ROC"] = gm["AUC_ROC"]

    return df


def compute_auc_tpr(tpr_array: list[float], alphas: list[float]) -> float:
    if len(tpr_array) < 2:
        return 0.0
    return float(np.trapz(tpr_array, alphas) / (max(alphas) - min(alphas)))


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))