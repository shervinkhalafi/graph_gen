"""Model protocols for denoising and diffusion models.

Defines the structural contracts (typing.Protocol) that denoising and
diffusion models must satisfy. Concrete model classes may implement
these via duck typing or explicit inheritance; the protocols are the
authoritative interface specification.

The DenoisingModel ABC in base.py remains as a convenience base class,
but this protocol is what the Lightning modules type-check against.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from tmgg.models.digress.transformer_model import GraphFeatures


@runtime_checkable
class DenoisingModelProtocol(Protocol):
    """Protocol for models that denoise adjacency matrices.

    Models receive a noisy adjacency matrix (and optionally a timestep)
    and return raw logits. Post-processing (thresholding, symmetrization)
    is handled by predict() and logits_to_graph().

    Every existing DenoisingModel subclass satisfies this protocol.
    """

    def __call__(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        """Forward pass: noisy adjacency -> logits.

        Parameters
        ----------
        x : Tensor
            Noisy adjacency matrix, shape (batch, n, n).
        t : Tensor or None
            Per-sample timestep, shape (batch,). None for
            unconditional denoising.

        Returns
        -------
        Tensor
            Raw logits, shape (batch, n, n). Not thresholded.
        """
        ...

    def transform_for_loss(
        self, output: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Transform model output and target for loss computation.

        The default (identity) is correct for MSE and BCEWithLogits.
        Override when the model's output space differs from the target
        space (e.g., inv-sigmoid parameterization).

        Parameters
        ----------
        output : Tensor
            Raw model output (logits).
        target : Tensor
            Ground truth adjacency.

        Returns
        -------
        tuple[Tensor, Tensor]
            (transformed_output, transformed_target) ready for the loss
            function.
        """
        ...

    def predict(self, logits: Tensor, zero_diag: bool = True) -> Tensor:
        """Convert logits to binary graph predictions.

        Applies logits_to_graph() and optionally zeros the diagonal.

        Parameters
        ----------
        logits : Tensor
            Raw model output.
        zero_diag : bool
            If True, zero out diagonal (no self-loops).

        Returns
        -------
        Tensor
            Binary adjacency matrix.
        """
        ...

    def logits_to_graph(self, logits: Tensor) -> Tensor:
        """Convert raw logits to binary edge predictions.

        Default: threshold at 0 (equivalent to sigmoid > 0.5).

        Parameters
        ----------
        logits : Tensor
            Raw model output.

        Returns
        -------
        Tensor
            Binary predictions (0 or 1).
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for logging/serialization."""
        ...

    def parameter_count(self) -> dict[str, Any]:
        """Return hierarchical parameter counts."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Return iterator over module parameters (from nn.Module)."""
        ...


@runtime_checkable
class DiffusionModelProtocol(Protocol):
    """Protocol for models that predict clean categorical features from noisy input.

    Models receive one-hot categorical node/edge features with a node mask
    and return per-class logits. The caller is responsible for applying
    softmax and computing the loss.

    The canonical implementation is DiscreteGraphTransformer wrapping
    _GraphTransformer with assume_adjacency_input=False.
    """

    def __call__(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor,
    ) -> GraphFeatures:
        """Forward pass: categorical features -> categorical logits.

        Parameters
        ----------
        X : Tensor
            Node features, shape (batch, n, dx_in). Typically the
            concatenation of noisy one-hot features and extra features.
        E : Tensor
            Edge features, shape (batch, n, n, de_in).
        y : Tensor
            Global features, shape (batch, dy_in). Includes normalized
            timestep t/T when using timestep conditioning.
        node_mask : Tensor
            Boolean mask, shape (batch, n).

        Returns
        -------
        GraphFeatures
            Predicted logits. X: (batch, n, dx_out),
            E: (batch, n, n, de_out), y: (batch, dy_out).
            dx_out = num_node_classes, de_out = num_edge_classes.
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for logging/serialization."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Return iterator over module parameters (from nn.Module)."""
        ...
