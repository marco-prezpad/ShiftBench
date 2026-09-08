"""
test_factory.py

Tests for src/detectors/factory.py.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import pytest

from src.detectors.base import BaseDetector
from src.detectors.factory import DetectorFactory


class TestDetectorFactory:
    def setup_method(self):
        self._saved_registry = dict(DetectorFactory._registry)

    def teardown_method(self):
        DetectorFactory._registry = self._saved_registry

    def test_register_and_create(self):
        @DetectorFactory.register("dummy")
        class DummyDetector(BaseDetector):
            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return 0.0

        detector = DetectorFactory.create("dummy")
        assert isinstance(detector, DummyDetector)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            DetectorFactory.create("nonexistent")

    def test_available_returns_names(self):
        @DetectorFactory.register("dummy2")
        class Dummy2Detector(BaseDetector):
            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return 0.0

        assert "dummy2" in DetectorFactory.available()

    def test_create_passes_kwargs(self):
        @DetectorFactory.register("dummy3")
        class Dummy3Detector(BaseDetector):
            def __init__(self, param=42):
                self.param = param

            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return float(self.param)

        detector = DetectorFactory.create("dummy3", param=99)
        assert detector.param == 99

    def test_create_from_config_filters_unknown_keys(self):
        @DetectorFactory.register("dummy4")
        class Dummy4Detector(BaseDetector):
            def __init__(self, param=1):
                self.param = param

            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return float(self.param)

        detector = DetectorFactory.create_from_config(
            "dummy4", {"param": 7, "unsupported_key": "ignored"}
        )
        assert detector.param == 7
