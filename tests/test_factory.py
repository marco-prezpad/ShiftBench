"""
test_factory.py

Tests for factory module.
It checks the registration and creation of detectors.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
import pytest

from src.detectors.base import BaseDetector
from src.detectors.factory import DetectorFactory


class TestDetectorFactory:
    def setup_method(self):
        DetectorFactory._registry.clear()

    def test_register_and_create(self):
        @DetectorFactory.register("dummy")
        class DummyDetector(BaseDetector):
            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return 0.0

        detector = DetectorFactory.create("dummy")
        assert isinstance(detector, DummyDetector)
        assert isinstance(detector, BaseDetector)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            DetectorFactory.create("nonexistent")

    def test_available_returns_names(self):
        @DetectorFactory.register("dummy")
        class DummyDetector(BaseDetector):
            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return 0.0

        assert "dummy" in DetectorFactory.available()

    def test_create_passes_kwargs(self):
        @DetectorFactory.register("dummy")
        class DummyDetector(BaseDetector):
            def __init__(self, param=42):
                self.param = param

            def fit(self, X_ref):
                pass

            def score(self, X_test):
                return float(self.param)

        detector = DetectorFactory.create("dummy", param=99)
        assert detector.param == 99