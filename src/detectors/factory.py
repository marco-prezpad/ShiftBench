"""
factory.py

Factory to instantiate detectors from configuration.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""
import inspect

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
    def create_from_config(cls, name: str, params: dict) -> BaseDetector:
        """Create a detector instance, silently dropping any key in `params`
        that the detector's constructor does not accept.

        This is what lets configs/detector_params.yaml carry hyperparameters
        for detectors that don't (yet) expose every one of them -- e.g. the
        MMD/LSDD kernel or permutation settings -- without those extra keys
        raising a TypeError, and without ever silently changing a detector's
        behavior for a key it doesn't recognize.

        Args:
            name: Detector name (e.g. "mmd", "lsdd").
            params: Candidate keyword arguments; only the ones present in
                the detector's __init__ signature are actually passed.

        Returns:
            A BaseDetector instance.

        Raises:
            ValueError: If the detector name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown detector '{name}'. Registered: {list(cls._registry.keys())}"
            )
        detector_cls = cls._registry[name]
        accepted_param_names = inspect.signature(detector_cls.__init__).parameters
        accepted_params = {
            key: value for key, value in params.items() if key in accepted_param_names
        }
        return detector_cls(**accepted_params)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered detector names."""
        return list(cls._registry.keys())
