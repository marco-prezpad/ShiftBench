#!/usr/bin/env python3
"""
calibration_report.py

Calculates the calibration report for each domain and detector at alpha=0. It computes:
- FPR and its Wilson confidence interval.
- Binomial test p-value against the nominal significance level (0.05).
It also includes the AUC-ROC score if available in the metrics.csv.

Usage:
    python scripts/calibration_report.py [--results-root results]

Author: Marco Pérez Padilla
Date:   08-09-2026
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

N_BOOTSTRAP = 100          
P0 = 0.05                 
DOMAINS = {
    "adult": "results/adult/metrics.csv",
    "cifar10c": "results/cifar10c/metrics.csv",
    "timeseries": "results/timeseries/metrics.csv",
    "text": "results/text/metrics.csv",
}
OUTPUT = "results/calibration_report.csv"

def fpr_ci(n_success, n_total, p0=0.05):
    """IC Wilson and exact binomial p-value."""
    ci_low, ci_high = proportion_confint(n_success, n_total, alpha=0.05, method="wilson")
    p_value = binomtest(n_success, n_total, p0, alternative="two-sided").pvalue
    return ci_low, ci_high, p_value


def bootstrap_auc_ci(scores_h0, scores_h1, n_boot=1000, seed=42):
    """IC bootstrap for AUC-ROC using paired samples."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(len(scores_h0)), np.ones(len(scores_h1))])
    X = np.concatenate([scores_h0, scores_h1])
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        X_b = X[idx]
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        order = np.argsort(X_b)
        ranked = np.argsort(order)
        n0 = np.sum(y_b == 0)
        n1 = np.sum(y_b == 1)
        r0 = np.sum(ranked[y_b == 0])
        auc = (r0 - n0 * (n0 + 1) / 2) / (n0 * n1)
        aucs.append(auc)
    return np.percentile(aucs, [2.5, 97.5])


filas = []
for domain, raw_path in DOMAINS.items():
    path = Path(raw_path)
    if not path.exists():
        print(f"WARNING: {path} does not exist, skipping {domain}")
        continue

    df = pd.read_csv(path)
    h0 = df[df["alpha"] == 0.0].copy()
    if h0.empty:
        print(f"WARNING: No alpha=0 row in {domain}")
        continue

    for _, row in h0.iterrows():
        det = row["detector"]
        fpr = row["FPR"]
        n_fp = int(round(fpr * N_BOOTSTRAP))
        ci_low, ci_high, p_val = fpr_ci(n_fp, N_BOOTSTRAP, P0)

        auc_roc = row.get("AUC_ROC", np.nan)

        filas.append({
            "domain": domain,
            "detector": det,
            "FPR": fpr,
            "n_fp": n_fp,
            "CI_low_wilson": ci_low,
            "CI_high_wilson": ci_high,
            "p_binomial_vs_0.05": p_val,
            "AUC_ROC": auc_roc,
        })

report = pd.DataFrame(filas)
report.to_csv(OUTPUT, index=False)
print(f"OK report saved in {OUTPUT}")
print(report.round(4).to_string(index=False))