"""Factory for creating cloud runners.

Provides a registry-based factory pattern for cloud execution backends.
External packages (like modal runner) can register their implementations
via the register() method or entry points.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import CloudRunner

logger = logging.getLogger(__name__)


class CloudRunnerFactory:
    """Factory for creating cloud runner instances.

    Uses a registry pattern allowing external packages to register their
    runner implementations. LocalRunner is registered by default.

    Examples
    --------
    Using the factory with built-in runners:

    >>> runner = CloudRunnerFactory.create("local")
    >>> result = runner.run_experiment(config)

    Registering an external runner:

    >>> from my_package import MyCloudRunner
    >>> CloudRunnerFactory.register("mycloud", MyCloudRunner)
    >>> runner = CloudRunnerFactory.create("mycloud", api_key="...")
    """

    _runners: dict[str, type[CloudRunner]] = {}

    @classmethod
    def register(cls, name: str, runner_class: type[CloudRunner]) -> None:
        """Register a runner class with the factory.

        Parameters
        ----------
        name
            Backend identifier (e.g., "local", "modal", "ray").
        runner_class
            Class implementing CloudRunner interface.
        """
        cls._runners[name] = runner_class
        logger.debug(f"Registered cloud runner: {name}")

    @classmethod
    def create(cls, backend: str, **kwargs: Any) -> CloudRunner:
        """Create a runner instance for the specified backend.

        Parameters
        ----------
        backend
            Backend identifier (e.g., "local", "modal").
        **kwargs
            Arguments passed to the runner constructor.

        Returns
        -------
        CloudRunner
            Configured runner instance.

        Raises
        ------
        ValueError
            If the backend is not registered.
        """
        if backend not in cls._runners:
            available = list(cls._runners.keys())
            raise ValueError(
                f"Unknown backend: '{backend}'. Available: {available}. "
                f"For Modal, ensure tmgg_modal is installed."
            )
        return cls._runners[backend](**kwargs)

    @classmethod
    def available_backends(cls) -> list[str]:
        """List all registered backends.

        Returns
        -------
        list[str]
            Names of registered backends.
        """
        return list(cls._runners.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a backend is registered.

        Parameters
        ----------
        name
            Backend identifier to check.

        Returns
        -------
        bool
            True if backend is registered.
        """
        return name in cls._runners


def _register_builtin_runners() -> None:
    """Register built-in runners with the factory."""
    from .base import LocalRunner

    CloudRunnerFactory.register("local", LocalRunner)


def _try_register_modal() -> None:
    """Attempt to register Modal runner if available."""
    try:
        from tmgg_modal.runner import ModalRunner

        CloudRunnerFactory.register("modal", ModalRunner)
        logger.debug("Modal runner auto-registered")
    except ImportError:
        logger.debug("Modal runner not available (tmgg_modal not installed)")


# Auto-register runners on module import
_register_builtin_runners()
_try_register_modal()
