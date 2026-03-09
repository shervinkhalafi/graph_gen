"""Single-step denoising module for graph reconstruction experiments.

``SingleStepDenoisingModule`` extends :class:`DiffusionModule` for the T=1
denoising setting: sample a noise level, corrupt the graph, predict the clean
graph in one forward pass. This matches all existing denoising experiments
(spectral, GNN, hybrid, baseline) in the codebase.

Because T=1 denoising does not require a reverse sampler, the constructor
passes ``sampler=None`` to :class:`DiffusionModule`. An optional
``GraphEvaluator`` can be provided for distributional metrics (MMD,
uniqueness, etc.) computed per noise level at epoch end.
The class overrides ``training_step``, ``forward``, ``validation_step``, and
``test_step`` to retain the per-noise-level evaluation logic that the
multi-step parent class does not provide.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast, override

import networkx as nx
import numpy as np
import torch

from tmgg.data import (
    create_noise_generator,  # pyright: ignore[reportUnknownVariableType]
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.noise import NoiseGenerator
from tmgg.diffusion.noise_process import ContinuousNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.experiments._shared_utils.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.experiments._shared_utils.lightning_modules.diffusion_module import (
    DiffusionModule,
)

# Maps the legacy denoising loss names to DiffusionModule's loss vocabulary.
_DENOISING_LOSS_MAP: dict[str, str] = {
    "MSE": "mse",
    "BCEWithLogits": "bce_logits",
}


class SingleStepDenoisingModule(DiffusionModule):
    """Single-step denoising module for graph reconstruction experiments.

    Given a noisy graph, predict the clean graph in one forward pass (T=1).
    Noise levels are sampled from a discrete set during training; at
    validation/test time every level is evaluated and per-level metrics
    are logged alongside noise-level-averaged metrics.

    Parameters
    ----------
    model_type : str
        Key registered in :class:`~tmgg.models.factory.ModelRegistry`.
    model_config : dict[str, Any]
        Configuration forwarded to the model factory.
    learning_rate : float
        Base learning rate.
    weight_decay : float
        Weight decay coefficient.
    optimizer_type : str
        ``"adam"`` or ``"adamw"``.
    amsgrad : bool
        Enable the AMSGrad variant.
    scheduler_config : dict[str, Any] | None
        Optional LR scheduler configuration.
    noise_type : str
        Noise generator type (``"digress"``, ``"gaussian"``, ``"edge_flip"``,
        ``"rotation"``, ``"logit"``).
    noise_levels : list[float] | None
        Noise levels for training. When ``None``, reads from the datamodule's
        ``noise_levels`` attribute at runtime.
    eval_noise_levels : list[float] | None
        Noise levels for evaluation. Falls back to ``noise_levels`` when ``None``.
    loss_type : str
        ``"MSE"`` or ``"BCEWithLogits"``.
    rotation_k : int
        Dimension for rotation noise skew matrix (only used when
        ``noise_type="rotation"``).
    seed : int | None
        Random seed for deterministic noise level sampling.
    spectral_k : int
        Number of top eigenvectors for spectral delta metrics.
    evaluator : GraphEvaluator | None
        When provided, distributional metrics (MMD, uniqueness, etc.) are
        computed per noise level at validation/test epoch end.
    eval_num_samples : int
        Maximum number of graphs to accumulate per noise level for
        distributional evaluation.
    """

    def __init__(
        self,
        *,
        # BaseGraphModule params
        model_type: str,
        model_config: dict[str, Any],  # pyright: ignore[reportExplicitAny]
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        # Denoising-specific params
        noise_type: str = "digress",
        noise_levels: list[float] | None = None,
        eval_noise_levels: list[float] | None = None,
        loss_type: str = "BCEWithLogits",
        rotation_k: int = 20,
        seed: int | None = None,
        # Logging options
        spectral_k: int = 4,
        # Distributional evaluation
        evaluator: GraphEvaluator | None = None,
        eval_num_samples: int = 128,
    ) -> None:
        # Map legacy loss names to DiffusionModule vocabulary
        mapped_loss = _DENOISING_LOSS_MAP.get(loss_type)
        if mapped_loss is None:
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. Use 'MSE' or 'BCEWithLogits'."
            )

        # Build the noise generator and wrap it in a ContinuousNoiseProcess
        noise_generator: NoiseGenerator = create_noise_generator(
            noise_type=noise_type, rotation_k=rotation_k, seed=seed
        )
        # T=1 schedule -- the parent's training_step is overridden anyway,
        # but NoiseSchedule is a required DiffusionModule dependency.
        noise_schedule = NoiseSchedule("linear_ddpm", timesteps=1)

        noise_process = ContinuousNoiseProcess(noise_generator, noise_schedule)

        super().__init__(
            model_type=model_type,
            model_config=model_config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
            noise_process=noise_process,
            sampler=None,
            noise_schedule=noise_schedule,
            evaluator=evaluator,
            loss_type=mapped_loss,
        )

        # Denoising-specific state
        self.noise_type: str = noise_type
        self._noise_levels_override: list[float] | None = noise_levels
        self._eval_noise_levels_override: list[float] | None = eval_noise_levels
        self.noise_generator: NoiseGenerator = noise_generator

        # Spectral delta logging
        self.spectral_k: int = spectral_k

        # Seeded RNG for noise level sampling (avoids global numpy state)
        self._noise_rng: np.random.Generator = np.random.default_rng(seed)

        # Distributional evaluation buffers (per noise level)
        self._eval_num_samples: int = eval_num_samples
        self._clean_graphs_by_eps: dict[float, list[nx.Graph[Any]]] = {}  # pyright: ignore[reportExplicitAny]
        self._denoised_graphs_by_eps: dict[float, list[nx.Graph[Any]]] = {}  # pyright: ignore[reportExplicitAny]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def noise_levels(self) -> list[float]:
        """Noise levels for training.

        Returns the explicitly provided list if set, otherwise reads from the
        datamodule's ``noise_levels`` attribute.

        Raises
        ------
        RuntimeError
            If no noise levels were provided and the datamodule is unavailable
            or lacks a ``noise_levels`` attribute.
        """
        if self._noise_levels_override is not None:
            return self._noise_levels_override
        dm = self._datamodule
        if dm is None:
            raise RuntimeError(
                "Cannot access noise_levels: none were provided at construction "  # noqa: ISC003
                + "and no trainer/datamodule is attached."
            )
        levels = getattr(dm, "noise_levels", None)  # pyright: ignore[reportAny]
        if levels is None:
            raise RuntimeError(
                f"Datamodule {type(dm).__name__} does not have a noise_levels attribute."  # pyright: ignore[reportAny]
            )
        return cast(list[float], levels)

    @property
    def eval_noise_levels(self) -> list[float]:
        """Noise levels for evaluation.

        Returns the explicitly provided eval levels if set, otherwise falls
        back to :attr:`noise_levels`.
        """
        if self._eval_noise_levels_override is not None:
            return self._eval_noise_levels_override
        return self.noise_levels

    @property
    def _datamodule(self) -> Any | None:  # pyright: ignore[reportExplicitAny]
        """Access the trainer's datamodule, if attached.

        Lightning's ``trainer`` property raises ``RuntimeError`` when no
        trainer is attached, so we catch that rather than comparing to None.
        """
        try:
            trainer = self.trainer
        except RuntimeError:
            return None
        if trainer is None:  # pyright: ignore[reportUnnecessaryComparison]
            return None
        return getattr(trainer, "datamodule", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        """Bridge raw adjacency tensors to the ``GraphData`` model interface.

        Wraps the input as ``GraphData``, calls the model, and extracts
        edge-channel logits (``E[..., 1]``) for backward compatibility with
        BCE/MSE loss computation.

        Parameters
        ----------
        x
            Input adjacency matrix, shape ``(B, N, N)``.
        t
            Optional timestep tensor (unused by most single-step models).

        Returns
        -------
        torch.Tensor
            Edge logits, shape ``(B, N, N)``.
        """
        data = GraphData.from_adjacency(x)
        result: GraphData = self.model(data, t=t)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return result.E[..., 1]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override
    def training_step(self, batch: GraphData, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Execute a single-step denoising training iteration.

        Samples a random noise level, corrupts the clean adjacency, runs
        the model forward, and computes loss against the clean target.

        Parameters
        ----------
        batch
            ``GraphData`` batch from the dataloader.
        batch_idx
            Index of the current batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss with gradient.
        """
        adj = batch.to_adjacency()

        # Sample noise level randomly from training noise levels
        eps: float = float(self._noise_rng.choice(self.noise_levels))

        # Apply noise
        batch_noisy: torch.Tensor = self.noise_generator.add_noise(adj, eps)

        # Forward pass: returns edge logits via GraphData bridge
        output: torch.Tensor = self.forward(batch_noisy)

        loss: torch.Tensor = self.criterion(output, adj)  # pyright: ignore[reportAny]

        # Training accuracy: threshold logits at 0, zero diagonal, compare
        with torch.no_grad():
            predictions: torch.Tensor = (output > 0).float()
            predictions = self._zero_diagonal(predictions)
            train_acc: torch.Tensor = (predictions == adj).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)  # pyright: ignore[reportUnknownMemberType]
        self.log(  # pyright: ignore[reportUnknownMemberType]
            "train/accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/noise_level", eps, on_step=False, on_epoch=True)  # pyright: ignore[reportUnknownMemberType]

        return loss

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_or_test(self, mode: str, batch: GraphData) -> dict[str, torch.Tensor]:
        """Evaluate across all noise levels, logging per-level and averaged metrics.

        Parameters
        ----------
        mode
            ``"val"`` or ``"test"``.
        batch
            Clean ``GraphData`` batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Noise-level-averaged loss and metrics.
        """
        adj = batch.to_adjacency()
        target = adj

        mode_loss_sum = torch.tensor(0.0, device=adj.device)
        metrics_sum: defaultdict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=adj.device)
        )
        N: int = len(self.eval_noise_levels)

        for eps in self.eval_noise_levels:
            batch_noisy: torch.Tensor = self.noise_generator.add_noise(adj, eps)
            output: torch.Tensor = self.forward(batch_noisy)
            mode_loss: torch.Tensor = self.criterion(output, target)  # pyright: ignore[reportAny]

            # Predictions: threshold logits at 0, zero diagonal
            predictions: torch.Tensor = (output > 0).float()
            predictions = self._zero_diagonal(predictions)

            # Per-graph reconstruction metrics (inlined)
            diff = target - predictions
            batch_metrics: dict[str, float] = {
                "mse": (diff**2).mean().item(),
                "frobenius_error": torch.norm(diff, p="fro", dim=(-2, -1))
                .mean()
                .item(),
                "accuracy": (predictions == target).float().mean().item(),
            }

            # Per-noise-level logging
            self.log(  # pyright: ignore[reportUnknownMemberType]
                f"{mode}_{eps}/loss",
                mode_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for metric_name, value in batch_metrics.items():
                self.log(  # pyright: ignore[reportUnknownMemberType]
                    f"{mode}_{eps}/{metric_name}", value, on_step=False, on_epoch=True
                )

            # Spectral deltas (always computed)
            self._log_spectral_deltas(mode, eps, target, batch_noisy, predictions)

            # Accumulate graphs for distributional evaluation
            if self.evaluator is not None:
                self._accumulate_graphs(eps, batch, target, predictions)

            mode_loss_sum = mode_loss_sum + mode_loss
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum[k] + torch.tensor(v, device=adj.device)

        # Noise-level-averaged metrics
        mode_loss_mean: torch.Tensor = mode_loss_sum / N
        self.log(  # pyright: ignore[reportUnknownMemberType]
            f"{mode}/loss", mode_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        metrics_mean: dict[str, torch.Tensor] = {
            k: v / N for k, v in metrics_sum.items()
        }
        for metric_name, value in metrics_mean.items():
            self.log(f"{mode}/{metric_name}", value, on_step=False, on_epoch=True)  # pyright: ignore[reportUnknownMemberType]

        return {f"{mode}_loss": mode_loss_mean, **metrics_mean}

    @torch.no_grad()
    @override
    def validation_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: GraphData, batch_idx: int
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Validation step -- delegates to :meth:`_val_or_test`."""
        return self._val_or_test(mode="val", batch=batch)

    @torch.no_grad()
    @override
    def test_step(self, batch: GraphData, batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        """Test step -- delegates to :meth:`_val_or_test`."""
        return self._val_or_test(mode="test", batch=batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_diagonal(A: torch.Tensor) -> torch.Tensor:
        """Zero out diagonal entries of a batched adjacency matrix.

        Parameters
        ----------
        A
            Adjacency matrix, shape ``(B, N, N)`` or ``(N, N)``.

        Returns
        -------
        torch.Tensor
            Copy with diagonal entries set to zero.
        """
        mask = torch.eye(A.shape[-1], device=A.device, dtype=torch.bool)
        if A.ndim == 3:
            mask = mask.unsqueeze(0)
        return A.masked_fill(mask, 0)

    def _log_spectral_deltas(
        self,
        mode: str,
        eps: float,
        A_clean: torch.Tensor,
        A_noisy: torch.Tensor,
        A_denoised: torch.Tensor,
    ) -> None:
        """Log spectral delta metrics between clean, noisy, and denoised graphs.

        Computes four spectral delta metrics for both the noisy-vs-clean and
        denoised-vs-clean comparisons.

        Parameters
        ----------
        mode
            Logging mode (``"val"`` or ``"test"``).
        eps
            Noise level.
        A_clean
            Clean adjacency matrices, shape ``(batch, n, n)``.
        A_noisy
            Noisy adjacency matrices, same shape.
        A_denoised
            Denoised (predicted) adjacency matrices, same shape.
        """
        from tmgg.experiments._shared_utils.spectral_utils.spectral_deltas import (
            compute_spectral_deltas,
        )

        with torch.no_grad():
            deltas_noisy = compute_spectral_deltas(A_clean, A_noisy, k=self.spectral_k)
            for name, values in deltas_noisy.items():
                self.log(  # pyright: ignore[reportUnknownMemberType]
                    f"{mode}_{eps}/noisy_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

            deltas_denoised = compute_spectral_deltas(
                A_clean, A_denoised, k=self.spectral_k
            )
            for name, values in deltas_denoised.items():
                self.log(  # pyright: ignore[reportUnknownMemberType]
                    f"{mode}_{eps}/denoised_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

    def _accumulate_graphs(
        self,
        eps: float,
        batch: GraphData,
        clean_adj: torch.Tensor,
        denoised_adj: torch.Tensor,
    ) -> None:
        """Convert tensor adjacencies to NetworkX and store per noise level.

        Respects ``node_mask`` for variable-size graphs. Accumulates up to
        ``_eval_num_samples`` graphs per noise level.

        Parameters
        ----------
        eps
            Noise level key.
        batch
            Original ``GraphData`` batch (used for ``node_mask``).
        clean_adj
            Clean adjacency tensors, shape ``(B, N, N)``.
        denoised_adj
            Denoised (predicted) adjacency tensors, same shape.
        """
        clean_list = self._clean_graphs_by_eps.setdefault(eps, [])
        denoised_list = self._denoised_graphs_by_eps.setdefault(eps, [])

        bs = clean_adj.shape[0]
        for i in range(bs):
            if len(clean_list) >= self._eval_num_samples:
                break
            n = int(batch.node_mask[i].sum().item())  # pyright: ignore[reportUnknownMemberType]
            clean_np = clean_adj[i, :n, :n].cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
            denoised_np = denoised_adj[i, :n, :n].cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
            clean_list.append(nx.from_numpy_array(clean_np))
            denoised_list.append(nx.from_numpy_array(denoised_np))

    @override
    def on_validation_epoch_end(self) -> None:
        """Evaluate distributional metrics per noise level at epoch end.

        For each noise level with sufficient accumulated graphs, runs the
        ``GraphEvaluator`` with clean graphs as references and denoised graphs
        as the "generated" set. Logs per-noise-level and noise-level-averaged
        distributional metrics. Clears accumulation buffers afterwards.
        """
        if self.evaluator is None:
            return

        all_results: dict[float, dict[str, float | None]] = {}

        for eps in self.eval_noise_levels:
            clean = self._clean_graphs_by_eps.get(eps, [])
            denoised = self._denoised_graphs_by_eps.get(eps, [])

            if len(clean) < 2 or len(denoised) < 2:
                continue

            # Temporarily set evaluator's ref_graphs to the clean graphs
            self.evaluator.ref_graphs = clean
            results = self.evaluator.evaluate(denoised)

            if results is None:
                continue

            metrics = results.to_dict()
            all_results[eps] = metrics

            for metric_name, value in metrics.items():
                if value is not None:
                    self.log(  # pyright: ignore[reportUnknownMemberType]
                        f"val_{eps}/{metric_name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                    )

        # Noise-level-averaged distributional metrics
        if all_results:
            metric_names = next(iter(all_results.values())).keys()
            for metric_name in metric_names:
                values: list[float] = [
                    v for r in all_results.values() if (v := r[metric_name]) is not None
                ]
                if values:
                    avg = sum(values) / len(values)
                    self.log(  # pyright: ignore[reportUnknownMemberType]
                        f"val/{metric_name}",
                        avg,
                        on_step=False,
                        on_epoch=True,
                    )

        # Clear buffers
        self._clean_graphs_by_eps.clear()
        self._denoised_graphs_by_eps.clear()
        self.evaluator.clear()
