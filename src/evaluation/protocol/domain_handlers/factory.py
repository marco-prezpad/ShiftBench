"""
factory.py

Factory to instantiate domain handlers from a domain name, mirroring
src/detectors/factory.py's DetectorFactory pattern.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""

from .base_domain_handler import BaseDomainHandler


class DomainHandlerFactory:
    """Factory for creating domain handler instances by domain name."""

    _registry: dict[str, type[BaseDomainHandler]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a domain handler class."""

        def wrapper(handler_cls: type[BaseDomainHandler]) -> type[BaseDomainHandler]:
            cls._registry[name] = handler_cls
            return handler_cls

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDomainHandler:
        """Create a domain handler instance by domain name.

        Extra keyword arguments that a given handler doesn't need are
        absorbed by its constructor's `**_unused_kwargs` and ignored, so
        callers can pass one shared kwargs bundle for every domain.

        Args:
            name: Domain name (e.g. "adult", "cifar10c").
            **kwargs: Arguments passed to the handler constructor.

        Returns:
            A BaseDomainHandler instance.

        Raises:
            ValueError: If the domain name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unsupported domain: '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered domain names."""
        return list(cls._registry.keys())
