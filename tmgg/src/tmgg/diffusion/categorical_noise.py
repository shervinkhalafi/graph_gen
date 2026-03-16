"""Categorical noise definition for discrete diffusion.

Wraps a ``TransitionModel`` to apply categorical forward noise via
transition matrix multiplication on one-hot features. This class
extracts the stateless noise math from ``CategoricalNoiseProcess`` into
a reusable definition, mirroring how ``NoiseDefinition`` (in
``tmgg.utils.noising``) captures the math for continuous noise types.

Lives in ``tmgg.diffusion`` (not ``tmgg.utils.noising``) because it
depends on ``TransitionModel`` and ``TransitionMatrices`` from the
diffusion pipeline.
"""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.diffusion_sampling import sample_discrete_features
from tmgg.diffusion.protocols import TransitionModel


class CategoricalNoiseDefinition:
    """Categorical noise via discrete transition matrices.

    Wraps a ``TransitionModel`` that defines how one-hot categorical
    features transition under noise. The transition model can be set
    after construction (deferred injection) since it may depend on data
    marginals not available at init time.

    Parameters
    ----------
    x_classes
        Number of node feature categories.
    e_classes
        Number of edge feature categories.
    transition_model
        Transition matrix provider. ``None`` for deferred injection via
        ``set_transition_model()``.
    """

    def __init__(
        self,
        x_classes: int,
        e_classes: int,
        transition_model: TransitionModel | None = None,
    ) -> None:
        self.x_classes = x_classes
        self.e_classes = e_classes
        self._transition_model = transition_model

    def set_transition_model(self, model: TransitionModel) -> None:
        """Inject the transition model (deferred initialisation).

        Parameters
        ----------
        model
            A fully-configured transition model satisfying the
            ``TransitionModel`` protocol.

        Raises
        ------
        TypeError
            If *model* does not satisfy the ``TransitionModel`` protocol.
        """
        if not isinstance(model, TransitionModel):
            raise TypeError(
                f"Expected a TransitionModel (protocol requires get_Qt, "
                f"get_Qt_bar, get_limit_dist), got {type(model).__name__}"
            )
        self._transition_model = model

    @property
    def transition_model(self) -> TransitionModel:
        """The active transition model.

        Raises
        ------
        RuntimeError
            If no transition model has been set.
        """
        if self._transition_model is None:
            raise RuntimeError(
                "TransitionModel not set. Call set_transition_model() first."
            )
        return self._transition_model

    @staticmethod
    def apply_transition(
        features: Tensor,
        transition_matrix: Tensor,
    ) -> Tensor:
        """Multiply one-hot features by a transition matrix.

        Parameters
        ----------
        features
            One-hot features, shape ``(..., num_classes)``.
        transition_matrix
            Row-stochastic transition matrix, shape
            ``(..., num_classes, num_classes)``.

        Returns
        -------
        Tensor
            Class probabilities after transition, same leading shape as
            *features*.
        """
        return torch.matmul(features.float(), transition_matrix)

    def apply_noise(self, data: GraphData, alpha_bar: Tensor) -> GraphData:
        """Apply categorical forward noise at cumulative signal level.

        Multiplies one-hot features by the cumulative transition matrix
        ``Qt_bar(alpha_bar)`` to obtain class probabilities, samples
        discrete features, and converts back to one-hot encoding.

        Parameters
        ----------
        data
            Clean graph data with one-hot X and E features.
        alpha_bar
            Cumulative signal retention, shape ``(bs,)`` or scalar.
            ``alpha_bar = 1`` means no noise; ``alpha_bar = 0`` means
            full noise (limit distribution).

        Returns
        -------
        GraphData
            Noisy graph data with sampled one-hot features.
        """
        transition = self.transition_model

        # Cumulative transition matrices Q_bar(alpha_bar)
        Qtb = transition.get_Qt_bar(alpha_bar)

        # Node features: (bs, n, dx) @ (bs, dx, dx) -> (bs, n, dx)
        prob_X = self.apply_transition(data.X, Qtb.X)

        # Edge features: flatten spatial dims for matmul
        bs, n, _, de = data.E.shape
        E_flat = data.E.float().reshape(bs, n * n, de)  # (bs, n*n, de)
        prob_E_flat = self.apply_transition(E_flat, Qtb.E)  # (bs, n*n, de)
        prob_E = prob_E_flat.reshape(bs, n, n, de)

        # Sample discrete features from the computed probabilities
        sampled = sample_discrete_features(prob_X, prob_E, data.node_mask)

        # Convert sampled indices to one-hot
        X_noisy = F.one_hot(sampled.X.long(), num_classes=self.x_classes).float()
        E_noisy = F.one_hot(sampled.E.long(), num_classes=self.e_classes).float()

        return GraphData(
            X=X_noisy,
            E=E_noisy,
            y=data.y,
            node_mask=data.node_mask,
        ).mask()
