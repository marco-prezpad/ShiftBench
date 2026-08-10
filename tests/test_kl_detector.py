"""
test_kl_detector.py

Tests for KLDetector.
It checks that the detector can be created, fitted, and scored correctly.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
import pytest

from src.detectors.factory import DetectorFactory


@pytest.fixture(autouse=True)
def _register_kl():
    import src.detectors.kl_detector  # noqa: F401


class TestKLDetector:
    def test_no_drift_low_score(self):
        detector = DetectorFactory.create("kl")
        X_ref = np.random.randn(200, 5)
        X_test = np.random.randn(200, 5)

        detector.fit(X_ref)
        score = detector.score(X_test)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_drift_high_score(self):
        detector = DetectorFactory.create("kl")
        X_ref = np.random.randn(200, 5)
        X_test = np.random.randn(200, 5) + 2.0

        detector.fit(X_ref)
        score_drift = detector.score(X_test)

        X_no_drift = np.random.randn(200, 5)
        score_no_drift = detector.score(X_no_drift)

        assert score_drift > score_no_drift