"""
test_embedding_detector.py

Tests for EmbeddingDriftDetector.
It checks that the detector can be created, fitted, and scored correctly.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import numpy as np
import pytest

from src.detectors.factory import DetectorFactory


@pytest.fixture(autouse=True)
def _register_embedding():
    import src.detectors.embedding_drift_detector  # noqa: F401


class TestEmbeddingDriftDetector:
    def test_no_drift_low_score(self):
        detector = DetectorFactory.create("embedding")
        X_ref = np.random.randn(200, 10)
        X_test = np.random.randn(200, 10)

        detector.fit(X_ref)
        score = detector.score(X_test)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_drift_high_score(self):
        detector = DetectorFactory.create("embedding")
        X_ref = np.random.randn(200, 10)
        X_test = np.random.randn(200, 10) + 2.0

        detector.fit(X_ref)
        score_drift = detector.score(X_test)

        X_no_drift = np.random.randn(200, 10)
        score_no_drift = detector.score(X_no_drift)

        assert score_drift > score_no_drift

    def test_n_components_clamped(self):
        detector = DetectorFactory.create("embedding", n_components=50)
        X_ref = np.random.randn(100, 5)  
        detector.fit(X_ref)
        assert detector._pca.n_components_ == 5