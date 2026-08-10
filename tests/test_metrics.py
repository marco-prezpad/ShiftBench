"""
test_metrics.py

Tests for evaluation metrics.
It checks the correctness of the metrics computation functions in the evaluation module.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_threshold,
    compute_metrics_for_alpha,
    compute_detection_metrics,
    compute_auc_tpr,
    compute_auc_roc,
)


class TestComputeThreshold:
    def test_percentile_95(self):
        scores = np.random.randn(1000)
        t = compute_threshold(scores, 95)
        assert np.isclose(np.percentile(scores, 95), t)

    def test_all_same(self):
        scores = np.ones(100) * 0.5
        t = compute_threshold(scores, 95)
        assert t == 0.5


class TestComputeMetricsForAlpha:
    def test_all_above_threshold(self):
        scores = np.array([1.0, 2.0, 3.0])
        m = compute_metrics_for_alpha(scores, 0.5)
        assert m["TPR"] == 1.0
        assert m["FPR"] == 0.0

    def test_none_above_threshold(self):
        scores = np.array([0.0, 0.1, 0.2])
        m = compute_metrics_for_alpha(scores, 0.5)
        assert m["TPR"] == 0.0
        assert m["FPR"] == 1.0


class TestComputeDetectionMetrics:
    def test_returns_dataframe(self):
        scores = {
            "dummy": {
                0.0: [0.1, 0.2, 0.15],
                0.5: [0.9, 0.8, 0.85],
                1.0: [0.95, 0.99, 0.98],
            }
        }
        df = compute_detection_metrics(scores, alphas=[0.0, 0.5, 1.0], n_bootstrap=3)
        assert isinstance(df, pd.DataFrame)
        assert "TPR" in df.columns
        assert "TNR" in df.columns

        def test_tnr_near_one_for_no_shift(self):
            rng = np.random.default_rng(42)
            h0_scores = rng.normal(0, 1, 100).tolist()
            drift_scores = rng.normal(3, 0.5, 100).tolist()
            scores = {
                "good": {
                    0.0: h0_scores,
                    0.5: drift_scores,
                }
            }
            df = compute_detection_metrics(scores, alphas=[0.0, 0.5], n_bootstrap=100)
            tnr = df.loc[df["detector"] == "good", "TNR"].values[0]
            assert tnr >= 0.85  


class TestComputeAucTpr:
    def test_perfect_detector(self):
        tpr = [1.0, 1.0, 1.0]
        alphas = [0.1, 0.5, 1.0]
        auc = compute_auc_tpr(tpr, alphas)
        assert auc == 1.0

    def test_worst_detector(self):
        tpr = [0.0, 0.0, 0.0]
        alphas = [0.1, 0.5, 1.0]
        auc = compute_auc_tpr(tpr, alphas)
        assert auc == 0.0


class TestComputeDetectionMetricsEdgeCases:
    def test_missing_h0_scores(self):
        scores = {"det": {0.5: [0.8, 0.9]}}  
        df = compute_detection_metrics(scores, alphas=[0.0, 0.5], n_bootstrap=2)
        assert df.empty

    def test_empty_scores_for_alpha(self):
        scores = {"det": {0.0: [0.1, 0.2], 0.5: []}}  
        df = compute_detection_metrics(scores, alphas=[0.0, 0.5], n_bootstrap=2)
        assert len(df) == 1

    def test_single_alpha_auc_tpr_zero(self):
        tpr = [0.8]
        alphas = [0.5]
        auc = compute_auc_tpr(tpr, alphas)
        assert auc == 0.0


class TestComputeAucRoc:
    def test_auc_roc_perfect(self):
        scores = np.array([0.1, 0.2, 0.9, 0.95])
        labels = np.array([0, 0, 1, 1])
        auc = compute_auc_roc(scores, labels)
        assert auc == 1.0

    def test_auc_roc_random(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 0, 1, 1])
        auc = compute_auc_roc(scores, labels)
        assert auc == 0.5