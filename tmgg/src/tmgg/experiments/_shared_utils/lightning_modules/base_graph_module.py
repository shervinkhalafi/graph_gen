"""Shared LightningModule base class for all graph learning experiments.

``BaseGraphModule`` extracts pure infrastructure — model creation via
:class:`~tmgg.models.factory.ModelRegistry`, optimizer/scheduler setup,
batch device transfer, parameter logging — into a single base class.
It carries **no** training logic; subclasses like ``DiffusionModule``
and ``SingleStepDenoisingModule`` implement ``training_step`` and ``forward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

import pytorch_lightning as pl
import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.experiments._shared_utils.lightning_modules.optimizer_config import (
    OptimizerLRSchedulerConfig,
    SchedulerInfo,
    configure_optimizers_from_config,
)
from tmgg.experiments._shared_utils.logging import (
    log_parameter_count as _log_parameter_count,
)
from tmgg.models.base import GraphModel  # pyright: ignore[reportAttributeAccessIssue]


class BaseGraphModule(pl.LightningModule, ABC):
    """Shared infrastructure for all graph learning experiments.

    Handles model creation through :class:`~tmgg.models.factory.ModelRegistry`,
    optimizer and scheduler construction, ``GraphData`` batch transfer, and
    parameter-count logging. Subclasses only need to supply ``training_step``
    and ``forward``.

    Parameters
    ----------
    model_type : str
        Key registered in :class:`~tmgg.models.factory.ModelRegistry`.
    model_config : dict[str, Any]
        Configuration dict forwarded to the registry factory.
    learning_rate : float
        Base learning rate for the optimizer.
    weight_decay : float
        Weight decay coefficient (only effective with AdamW).
    optimizer_type : str
        One of ``"adam"`` or ``"adamw"``.
    amsgrad : bool
        Whether to enable the AMSGrad variant.
    scheduler_config : dict[str, Any] | None
        Optional scheduler configuration. Supported ``"type"`` values:
        ``"cosine"``, ``"cosine_warmup"``, ``"step"``, ``"none"``.
    """

    def __init__(
        self,
        *,
        model_type: str,
        model_config: dict[str, Any],  # pyright: ignore[reportExplicitAny]
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Store optimizer params
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.optimizer_type: str = optimizer_type.lower()
        self.amsgrad: bool = amsgrad
        self.scheduler_config: dict[str, Any] | None = scheduler_config  # pyright: ignore[reportExplicitAny]

        # Create model via factory
        self.model: GraphModel = self._make_parametrized_model(model_type, model_config)  # pyright: ignore[reportUnknownMemberType]

        # Populated by configure_optimizers when cosine_warmup scheduler is used
        self._scheduler_info: SchedulerInfo | None = None

    def _make_parametrized_model(
        self,
        model_type: str,
        model_config: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> GraphModel:
        """Create a model via :class:`~tmgg.models.factory.ModelRegistry`.

        Parameters
        ----------
        model_type
            Registered model-type key.
        model_config
            Configuration dict forwarded to the factory callable.

        Returns
        -------
        GraphModel
            Instantiated model.

        Raises
        ------
        TypeError
            If the factory returns an object that is not a ``GraphModel``.
        """
        from tmgg.models.factory import ModelRegistry

        model = ModelRegistry.create(model_type, model_config)
        if not isinstance(model, GraphModel):
            raise TypeError(
                f"ModelRegistry returned {type(model).__name__}, expected GraphModel subclass"
            )
        return model

    @override
    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        """Delegate to :func:`configure_optimizers_from_config`."""
        result, scheduler_info = configure_optimizers_from_config(
            self,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            amsgrad=self.amsgrad,
            scheduler_config=self.scheduler_config,
        )
        self._scheduler_info = scheduler_info
        return result

    @override
    def transfer_batch_to_device(
        self,
        batch: GraphData,  # type: ignore[override]
        device: torch.device,
        dataloader_idx: int,
    ) -> GraphData:
        """Move a ``GraphData`` batch to the target device."""
        return batch.to(device)

    @override
    def on_fit_start(self) -> None:
        """Log parameter count at training start."""
        _log_parameter_count(self.model, self.get_model_name(), self.logger)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    def get_model_name(self) -> str:
        """Return ``model_type`` from hyperparameters.

        Subclasses may override for a more descriptive name.
        """
        model_type: str = str(self.hparams.get("model_type", "unknown"))  # pyright: ignore[reportUnknownMemberType]
        return model_type

    def get_model_config(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Delegate to ``model.get_config()``.

        Returns
        -------
        dict[str, Any]
            Configuration dict as reported by the underlying model.
        """
        return self.model.get_config()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @abstractmethod
    @override
    def training_step(
        self,
        batch: Any,
        batch_idx: int,  # pyright: ignore[reportExplicitAny]
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        """Execute a single training step. Subclasses must implement."""
        ...

    @abstractmethod
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportExplicitAny,reportAny]
        """Forward pass. Subclasses must implement."""
        ...
