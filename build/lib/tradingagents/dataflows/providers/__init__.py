"""Point-in-time data providers.

Use ``get_provider(name)`` to fetch a registered provider, or
``pit_safe_methods(name)`` to query its PIT-safe surface.
"""

from .base import DATA_METHODS, DataProvider, FilingLag, is_historical

# Lazy registry — providers register themselves on import. Defined before
# the side-effect imports below so decorators can find it.
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator: register a provider class under ``name``."""
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap


def pit_safe_methods(provider_name: str) -> frozenset[str]:
    """Return the set of PIT-safe method names for the named provider."""
    cls = _REGISTRY.get(provider_name)
    if cls is None:
        return frozenset()
    return getattr(cls, "PIT_SAFE", frozenset())


def warn_if_not_pit_safe(provider_name: str, method: str, as_of: str) -> None:
    """Emit a warning when a non-PIT-safe method is called in historical mode.

    Silent by default for live runs (``as_of`` is today). Made loud for
    backtests so leakage doesn't masquerade as alpha.
    """
    import logging
    if not is_historical(as_of):
        return
    if method in pit_safe_methods(provider_name):
        return
    logging.getLogger(__name__).warning(
        "PIT leakage risk: %s.%s called with as_of=%s but provider does not "
        "guarantee point-in-time correctness for this method. Result may "
        "contain information that was not public on %s.",
        provider_name, method, as_of, as_of,
    )


# Register built-in providers by import. Each module's @register decorator
# adds itself to _REGISTRY. Imports below are intentional side-effects.
from . import yfinance_provider as _yfinance_provider  # noqa: F401,E402
from . import alpha_vantage_provider as _alpha_vantage_provider  # noqa: F401,E402
from . import polygon_provider as _polygon_provider  # noqa: F401,E402
from . import edgar_provider as _edgar_provider  # noqa: F401,E402


__all__ = [
    "DATA_METHODS",
    "DataProvider",
    "FilingLag",
    "is_historical",
    "pit_safe_methods",
    "warn_if_not_pit_safe",
]
