"""Composed MMD evaluator for generative graph Lightning modules.

Replaces the former ``GenerativeEvalMixin`` with an owned helper that
accumulates reference graphs during validation, then computes MMD metrics
against a set of generated graphs passed in by the caller.

The key design change: the evaluator never calls ``self.log()`` — the
owning Lightning module handles logging from the returned ``MMDResults``.
"""

from __future__ import annotations

from typing import Any, Literal

import networkx as nx

from tmgg.experiment_utils.mmd_metrics import MMDResults, compute_mmd_metrics


class MMDEvaluator:
    """Accumulates reference graphs and computes MMD metrics.

    Composed helper replacing the former GenerativeEvalMixin. Each generative
    Lightning module owns an MMDEvaluator instance and delegates accumulation
    and evaluation to it.

    Parameters
    ----------
    eval_num_samples
        Maximum number of reference/generated graphs for evaluation.
    kernel
        Kernel for MMD computation: "gaussian" or "gaussian_tv".
    sigma
        Bandwidth parameter for the kernel.
    """

    def __init__(
        self,
        eval_num_samples: int,
        kernel: Literal["gaussian", "gaussian_tv"] | str,
        sigma: float,
    ) -> None:
        self.eval_num_samples = eval_num_samples
        self.kernel = kernel
        self.sigma = sigma
        self._ref_graphs: list[nx.Graph[Any]] = []
        self._num_nodes: int | None = None

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def accumulate(self, graph: nx.Graph[Any]) -> None:
        """Add a reference graph if the accumulator is not yet full.

        Parameters
        ----------
        graph
            A NetworkX graph extracted from the validation batch.
        """
        if len(self._ref_graphs) < self.eval_num_samples:
            self._ref_graphs.append(graph)

    def set_num_nodes(self, num_nodes: int) -> None:
        """Record the node count for graph generation (set once per epoch)."""
        if self._num_nodes is None:
            self._num_nodes = num_nodes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int | None:
        """The recorded node count, or ``None`` if not yet set."""
        return self._num_nodes

    @property
    def num_ref_graphs(self) -> int:
        """Number of accumulated reference graphs."""
        return len(self._ref_graphs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, generated_graphs: list[nx.Graph[Any]]) -> MMDResults | None:
        """Compute MMD metrics between accumulated refs and generated graphs.

        Returns ``None`` when the reference set is too small (fewer than 2
        graphs) or the node count is unknown. Clears internal state after
        completion regardless of whether metrics were computed.

        Parameters
        ----------
        generated_graphs
            Generated graphs to compare against the accumulated references.

        Returns
        -------
        MMDResults or None
            The computed metrics, or ``None`` if evaluation was skipped.
        """
        if len(self._ref_graphs) < 2 or self._num_nodes is None:
            self.clear()
            return None

        num_to_evaluate = min(len(self._ref_graphs), self.eval_num_samples)

        mmd_results = compute_mmd_metrics(
            self._ref_graphs[:num_to_evaluate],
            generated_graphs,
            kernel=self.kernel,  # pyright: ignore[reportArgumentType]  # validated Literal mismatch
            sigma=self.sigma,
        )

        self.clear()
        return mmd_results

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset accumulators (reference graphs and node count)."""
        self._ref_graphs.clear()
        self._num_nodes = None
