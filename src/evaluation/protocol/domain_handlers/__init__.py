"""
src/evaluation/protocol/domain_handlers/__init__.py

Per-domain strategy implementations for BenchmarkProtocol, plus the
factory used to look them up by domain name.

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
from .base_domain_handler import BaseDomainHandler
from .factory import DomainHandlerFactory

# Import handlers to trigger registration with DomainHandlerFactory.
from . import adult_domain_handler  # noqa: F401,E402
from . import cifar10c_domain_handler  # noqa: F401,E402
from . import text_domain_handler  # noqa: F401,E402
from . import timeseries_domain_handler  # noqa: F401,E402

__all__ = ["BaseDomainHandler", "DomainHandlerFactory"]
