"""
test_mmd_detector.py

Tests for MMDDetector.

Author: Marco Pérez Padilla
Date:   13-08-2026
"""
import numpy as np
import pytest

from src.detectors.factory import DetectorFactory


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


class TestMMDDetector:
    def test_no_drift_low_score(self):
        detector = DetectorFactory.create("mmd")
        X_ref = np.random.randn(500, 5)
        X_test = np.random.randn(500, 5)

        detector.fit(X_ref)
        score = detector.score(X_test)
        assert isinstance(score, float)
        assert 0.0 <= score < 0.8   

    def test_drift_high_score(self):
        detector = DetectorFactory.create("mmd")
        X_ref = np.random.randn(500, 5)
        X_test = np.random.randn(500, 5) + 2.0

        detector.fit(X_ref)
        score_drift = detector.score(X_test)

        X_no_drift = np.random.randn(500, 5)
        score_no_drift = detector.score(X_no_drift)

        assert score_drift > score_no_drift