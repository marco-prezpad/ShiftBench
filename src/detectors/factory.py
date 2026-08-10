"""
factory.py

Factory to instantiate detectors from configuration.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

from .base import BaseDetector


class DetectorFactory:
    """Factory for creating detector instances by name."""

    _registry: dict[str, type[BaseDetector]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a detector class."""

        def wrapper(detector_cls: type[BaseDetector]) -> type[BaseDetector]:
            cls._registry[name] = detector_cls
            return detector_cls

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDetector:
        """Create a detector instance by name.

        Args:
            name: Detector name (e.g. "mmd", "lsdd").
            **kwargs: Arguments passed to the detector constructor.

        Returns:
            A BaseDetector instance.

        Raises:
            ValueError: If the detector name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown detector '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered detector names."""
        return list(cls._registry.keys())