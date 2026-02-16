"""Evaluation utilities for discrete diffusion graph generation.

Provides conversion from the categorical sample format returned by
:meth:`DiscreteDiffusionLightningModule.sample_batch` (integer class
indices) to adjacency matrices and NetworkX graphs, and an end-to-end
evaluation function that computes MMD metrics against a reference set.

For the checkpoint evaluation CLI, see ``evaluate_cli``.
"""

from __future__ import annotations

from typing import Any, Literal

import networkx as nx
import numpy as np
from torch import Tensor

from tmgg.experiment_utils.mmd_metrics import (
    MMDResults,
    adjacency_to_networkx,
    compute_mmd_metrics,
)

# ------------------------------------------------------------------
# Conversion utilities
# ------------------------------------------------------------------


def categorical_samples_to_adjacencies(
    samples: list[tuple[Tensor, Tensor]],
) -> list[np.ndarray]:
    """Convert discrete categorical samples to binary adjacency matrices.

    Each sample is ``(node_types, edge_types)`` where ``edge_types`` is an
    integer tensor of shape ``(n, n)`` with class indices.  Class 0 means
    "no edge"; any class > 0 means "edge present".

    Parameters
    ----------
    samples
        Output of :meth:`DiscreteDiffusionLightningModule.sample_batch`.

    Returns
    -------
    list[np.ndarray]
        Binary adjacency matrices, each of shape ``(n, n)`` with dtype
        ``float32``.
    """
    adjacencies: list[np.ndarray] = []
    for _node_types, edge_types in samples:
        adj = (edge_types > 0).numpy().astype(np.float32)
        # Ensure symmetric and zero diagonal
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        adjacencies.append(adj)
    return adjacencies


def categorical_samples_to_graphs(
    samples: list[tuple[Tensor, Tensor]],
) -> list[nx.Graph[Any]]:
    """Convert discrete categorical samples to NetworkX graphs.

    Parameters
    ----------
    samples
        Output of :meth:`DiscreteDiffusionLightningModule.sample_batch`.

    Returns
    -------
    list[nx.Graph]
        Undirected graphs, one per sample.
    """
    adjacencies = categorical_samples_to_adjacencies(samples)
    return [adjacency_to_networkx(adj) for adj in adjacencies]


def evaluate_discrete_samples(
    ref_graphs: list[nx.Graph[Any]],
    gen_samples: list[tuple[Tensor, Tensor]],
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 1.0,
) -> MMDResults:
    """Compute MMD metrics between reference graphs and generated samples.

    Parameters
    ----------
    ref_graphs
        Reference distribution as NetworkX graphs.
    gen_samples
        Generated samples from ``sample_batch()``.
    kernel
        MMD kernel. ``"gaussian_tv"`` is the DiGress default.
    sigma
        Kernel bandwidth.

    Returns
    -------
    MMDResults
        Degree, clustering, and spectral MMD values.
    """
    gen_graphs = categorical_samples_to_graphs(gen_samples)
    return compute_mmd_metrics(ref_graphs, gen_graphs, kernel=kernel, sigma=sigma)
