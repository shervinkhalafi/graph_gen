"""Multi-step diffusion training loop for graph generation.

``DiffusionModule`` extends :class:`BaseGraphModule` with a complete
discrete or continuous diffusion pipeline. It composes four injected
components -- a :class:`~tmgg.diffusion.NoiseProcess` (forward diffusion),
a :class:`~tmgg.diffusion.Sampler` (reverse sampling),
a :class:`~tmgg.diffusion.NoiseSchedule` (timestep-to-noise mapping),
and a :class:`~tmgg.training.graph_evaluator.GraphEvaluator`
(generation quality metrics) -- and wires them into Lightning's
``training_step`` / ``validation_step`` / ``on_validation_epoch_end`` hooks.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportIncompatibleMethodOverride=false
# PyTorch Lightning's self.log(), self.trainer, self.model, and self.hparams
# have incomplete type stubs. Config dicts legitimately use Any.
# reportIncompatibleMethodOverride: MRO composition with LightningModule and ABC
# causes unavoidable signature mismatches across the class hierarchy.

from __future__ import annotations

from typing import Any, override

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.size_distribution import SizeDistribution
from tmgg.diffusion.collectors import DiffusionLikelihoodCollector, StepMetricCollector
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    NoiseProcess,
)
from tmgg.diffusion.sampler import Sampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel
from tmgg.training.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.training.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.training.lightning_modules.train_loss_discrete import (
    TrainLossDiscrete,
)

_VALID_LOSS_TYPES = frozenset({"cross_entropy", "mse", "bce_logits"})


class DiffusionModule(BaseGraphModule):
    """Multi-step diffusion training loop for graph generation.

    Composes ``NoiseProcess``, ``Sampler``, ``GraphEvaluator``, and
    ``NoiseSchedule`` with a ``GraphModel`` to provide ``training_step``,
    ``validation_step``, and ``test_step`` for discrete or continuous
    diffusion.

    Both this implementation and the original DiGress (Eq. 3, Vignac et al.
    2023) use masked cross-entropy with ``lambda_E``-weighted edge loss for
    training. VLB metrics (Eq. 14) are computed during validation for
    monitoring.

    Parameters
    ----------
    model : GraphModel
        Pre-constructed graph model (instantiated by Hydra from nested
        ``_target_`` in the YAML config).
    model_name : str
        Human-readable identifier for logging and experiment naming.
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
    noise_process : NoiseProcess
        Forward diffusion noise process.
    sampler : Sampler | None
        Reverse diffusion sampler for generation. ``None`` for subclasses
        (e.g. ``SingleStepDenoisingModule``) that do not use reverse sampling.
    noise_schedule : NoiseSchedule
        Timestep-to-noise mapping.
    evaluator : GraphEvaluator | None
        Optional evaluator for generation-quality metrics.
    loss_type : str
        ``"cross_entropy"`` for categorical, ``"mse"`` or ``"bce_logits"``
        for continuous diffusion.
    lambda_E : float
        Weight for the edge loss relative to node loss in
        ``TrainLossDiscrete``. Default is 5.0 per DiGress convention.
        Only used when ``loss_type="cross_entropy"``.
    num_nodes : int
        Node count used when sampling graphs during evaluation.
    eval_every_n_steps : int
        Generative evaluation runs when ``global_step`` is a multiple of
        this value. Decouples evaluation frequency from batch size and
        dataset size.
    """

    def __init__(
        self,
        *,
        # BaseGraphModule params
        model: GraphModel,
        model_name: str = "",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,
        # Diffusion-specific params
        noise_process: NoiseProcess,
        sampler: Sampler | None = None,
        noise_schedule: NoiseSchedule,
        evaluator: GraphEvaluator | None = None,
        loss_type: str = "cross_entropy",
        lambda_E: float = 5.0,
        num_nodes: int = 20,
        eval_every_n_steps: int = 5000,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
        )
        # Re-save hparams at the DiffusionModule level, excluding objects
        # that cannot be reliably pickled for checkpoint reconstruction.
        self.save_hyperparameters(
            ignore=["model", "noise_process", "sampler", "noise_schedule", "evaluator"]
        )

        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. Expected one of {sorted(_VALID_LOSS_TYPES)}."
            )

        self.noise_process: NoiseProcess = noise_process
        self.sampler: Sampler | None = sampler
        self.noise_schedule: NoiseSchedule = noise_schedule
        self.evaluator: GraphEvaluator | None = evaluator

        self.num_nodes: int = num_nodes
        self.eval_every_n_steps: int = eval_every_n_steps

        self.criterion: nn.Module
        self._train_loss_discrete: TrainLossDiscrete | None = None
        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
            self._train_loss_discrete = TrainLossDiscrete(lambda_E=lambda_E)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:  # bce_logits
            self.criterion = nn.BCEWithLogitsLoss()

        # VLB accumulators (only used when noise_process is categorical)
        self._is_categorical: bool = isinstance(noise_process, CategoricalNoiseProcess)
        self._vlb_nll: list[torch.Tensor] = []
        self._vlb_kl_prior: list[torch.Tensor] = []
        self._vlb_kl_diffusion: list[torch.Tensor] = []
        self._vlb_reconstruction: list[torch.Tensor] = []
        self._vlb_log_pn: list[torch.Tensor] = []
        self._size_distribution: SizeDistribution | None = None

    @property
    def T(self) -> int:
        """Number of diffusion steps."""
        return self.noise_schedule.timesteps

    @override
    def setup(self, stage: str | None = None) -> None:
        """Deferred initialisation for CategoricalNoiseProcess with marginal transitions.

        When the noise process is categorical and has no transition model
        yet, constructs a ``MarginalUniformTransition`` from the
        datamodule's empirical marginals and injects it. Idempotent:
        no-op if the transition model already exists or the noise process
        is not categorical.
        """
        if (
            self._is_categorical
            and isinstance(self.noise_process, CategoricalNoiseProcess)
            and self.noise_process._transition_model is None
        ):
            from tmgg.diffusion.transitions import MarginalUniformTransition

            dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]
            if dm is None:
                raise RuntimeError("DataModule required for marginal transition setup")
            x_marginals = dm.node_marginals
            e_marginals = dm.edge_marginals
            transition = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.noise_process.y_classes,
            )
            self.noise_process.set_transition_model(transition)

        # Fetch size distribution from the datamodule for VLB log_pN term
        if self._size_distribution is None:
            dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]
            if dm is not None and hasattr(dm, "get_size_distribution"):
                self._size_distribution = dm.get_size_distribution("train")

    @override
    def on_validation_epoch_start(self) -> None:
        """Clear VLB accumulators at the start of each validation epoch."""
        self._vlb_nll.clear()
        self._vlb_kl_prior.clear()
        self._vlb_kl_diffusion.clear()
        self._vlb_reconstruction.clear()
        self._vlb_log_pn.clear()

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Forward pass: delegates to the underlying ``GraphModel``."""
        return self.model(data, t=t)

    @override
    def training_step(self, batch: GraphData, batch_idx: int) -> torch.Tensor:
        """Single diffusion training step.

        Samples a uniform random timestep per batch element, applies forward
        noise via the noise process, runs the model to predict the clean
        graph, and computes the loss against the original batch.

        Parameters
        ----------
        batch
            Clean ``GraphData`` batch from the dataloader.
        batch_idx
            Index of the current batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss with gradient.
        """
        bs: int = batch.X.shape[0]
        device = batch.X.device

        # Sample random timestep per batch element: t in {1, ..., T}
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)

        # Apply forward noise at the sampled timesteps
        z_t = self.noise_process.apply(batch, t_int)

        # Normalised timestep for model conditioning: t_norm = t/T ∈ [1/T, 1].
        # This is a linear rescaling of the integer index, NOT the noise level
        # (1 - alpha_bar), which follows the schedule's non-linear curve.
        # The model learns to map this linear position to the correct
        # denoising prediction regardless of schedule shape.
        t_norm = t_int.float() / self.T

        # Model predicts clean data
        pred = self.model(z_t, t=t_norm)

        # Loss against original clean data
        loss = self._compute_loss(pred, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _compute_loss(self, pred: GraphData, target: GraphData) -> torch.Tensor:
        """Compute loss between predicted and target ``GraphData``.

        For ``cross_entropy``, categorical features are flattened and the
        target one-hot is converted to class indices. For ``mse`` /
        ``bce_logits``, edge features are compared directly.

        Parameters
        ----------
        pred
            Model output.
        target
            Ground-truth clean graph.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        discrete_loss = self._train_loss_discrete
        if discrete_loss is not None:
            # TrainLossDiscrete expects softmaxed probabilities, not logits.
            # Clone before masking: mask_distributions modifies tensors in-place.
            pred_X = F.softmax(pred.X, dim=-1).clone()
            pred_E = F.softmax(pred.E, dim=-1).clone()
            return discrete_loss(
                pred_X,
                pred_E,
                target.X,
                target.E,
                target.node_mask,
            )
        else:
            # MSE or BCE: compare E directly
            return self.criterion(pred.E, target.E.float())

    def _compute_reconstruction_at_t1(self, batch: GraphData) -> torch.Tensor:
        """Compute reconstruction log-probability at t=1 (DiGress Eq. 14).

        Applies minimal noise (t=1), runs the model, and evaluates the
        log-probability of recovering the clean graph from the prediction.

        Parameters
        ----------
        batch
            Clean ``GraphData`` batch.

        Returns
        -------
        torch.Tensor
            Per-graph reconstruction log-probability, shape ``(bs,)``.
        """
        if not isinstance(self.noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"_compute_reconstruction_at_t1 requires CategoricalNoiseProcess, "
                f"got {type(self.noise_process).__name__}"
            )
        bs = batch.X.shape[0]
        device = batch.X.device

        # Apply noise at t=1
        t_int = torch.ones(bs, dtype=torch.long, device=device)
        z_1 = self.noise_process.apply(batch, t_int)

        # Model prediction at t=1
        t_norm = t_int.float() / self.T
        pred_logits = self.model(z_1, t=t_norm)

        # Softmax to probabilities
        pred_probs = GraphData(
            X=F.softmax(pred_logits.X, dim=-1),
            E=F.softmax(pred_logits.E, dim=-1),
            y=pred_logits.y,
            node_mask=pred_logits.node_mask,
        )

        # Mask invalid positions so they contribute nothing to the log-prob.
        # reconstruction_logp computes sum(clean * log(pred)), so we need:
        #   clean[invalid] = 0  (zero contribution from numerator)
        #   pred[invalid] = 1   (log(1) = 0, avoids 0 * -inf = NaN)
        node_mask = batch.node_mask
        inv = ~node_mask  # (bs, n)
        inv_edge = inv.unsqueeze(1) | inv.unsqueeze(2)  # (bs, n, n)
        pred_probs.X[inv] = 1.0
        pred_probs.E[inv_edge] = 1.0
        batch = GraphData(
            X=batch.X.clone(), E=batch.E.clone(), y=batch.y, node_mask=node_mask
        )
        batch.X[inv] = 0.0
        batch.E[inv_edge] = 0.0

        return self.noise_process.reconstruction_logp(batch, pred_probs)

    @torch.no_grad()
    @override
    def validation_step(self, batch: GraphData, batch_idx: int) -> None:
        """Compute validation loss and VLB metrics for a single batch.

        Reference graphs for generative evaluation are pulled from the
        datamodule at epoch end, not accumulated here.

        Parameters
        ----------
        batch
            Clean ``GraphData`` batch.
        batch_idx
            Index of the current batch (unused).
        """
        bs: int = batch.X.shape[0]
        device = batch.X.device

        # Validation loss at a random timestep
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)
        z_t = self.noise_process.apply(batch, t_int)
        t_norm = t_int.float() / self.T
        pred = self.model(z_t, t=t_norm)
        loss = self._compute_loss(pred, batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # VLB estimation via single random timestep (standard DDPM approach).
        # Each batch samples one t and computes L_t; the epoch-end average
        # gives an unbiased estimate of E_t[L_t]. This matches the DiGress
        # reference implementation (cvignac/DiGress, src/diffusion_model.py,
        # compute_val_loss). For exact full-chain VLB, see the periodic
        # DiffusionLikelihoodCollector evaluation in on_validation_epoch_end.
        if self._is_categorical and isinstance(
            self.noise_process, CategoricalNoiseProcess
        ):
            # VLB methods expect probability distributions, not logits.
            # The model outputs logits (CrossEntropyLoss handles conversion);
            # apply softmax here for VLB only.
            pred_probs = GraphData(
                X=F.softmax(pred.X, dim=-1),
                E=F.softmax(pred.E, dim=-1),
                y=pred.y,
                node_mask=pred.node_mask,
            )

            kl_prior_X, kl_prior_E, _ = self.noise_process.kl_prior(
                batch.X, batch.E, batch.node_mask
            )
            # kl_prior returns batch-summed scalars; normalise to per-sample
            kl_prior = (kl_prior_X + kl_prior_E) / bs
            kl_diffusion = self.noise_process.compute_Lt(batch, pred_probs, z_t, t_int)
            reconstruction = self._compute_reconstruction_at_t1(batch)

            # log p(n_G): log-probability of graph size under training distribution
            log_pn = torch.zeros(1, device=device)
            if self._size_distribution is not None:
                node_counts = batch.node_mask.sum(dim=-1).long()  # (bs,)
                log_pn = self._size_distribution.log_prob(node_counts).mean()

            # NLL = -log_pN + kl_prior + E_t[L_t] - reconstruction_logp
            # kl_diffusion and reconstruction are (bs,); kl_prior is scalar
            nll = -log_pn + kl_prior + kl_diffusion.mean() - reconstruction.mean()

            self._vlb_nll.append(nll.detach())
            self._vlb_kl_prior.append(kl_prior.detach())
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())
            self._vlb_reconstruction.append(reconstruction.mean().detach())
            self._vlb_log_pn.append(log_pn.detach())

    @torch.no_grad()
    @override
    def test_step(self, batch: GraphData, batch_idx: int) -> None:
        """Test step -- delegates to ``validation_step``."""
        self.validation_step(batch, batch_idx)

    # TODO: add on_test_epoch_end that mirrors on_validation_epoch_end
    # but passes "test" stage to get_reference_graphs.

    @override
    def on_validation_epoch_end(self) -> None:
        """Log VLB metrics and run generative evaluation at configured intervals.

        Reference graphs are pulled from the datamodule's
        ``get_reference_graphs()`` rather than accumulated per-batch.
        Generative evaluation is skipped when ``sampler`` is ``None``
        (e.g. single-step denoising) or no evaluator is attached.
        """
        # Log VLB metrics for categorical noise
        if self._is_categorical and self._vlb_nll:
            self.log("val/epoch_NLL", torch.stack(self._vlb_nll).mean(), on_epoch=True)
            self.log(
                "val/kl_prior", torch.stack(self._vlb_kl_prior).mean(), on_epoch=True
            )
            self.log(
                "val/kl_diffusion",
                torch.stack(self._vlb_kl_diffusion).mean(),
                on_epoch=True,
            )
            self.log(
                "val/reconstruction_logp",
                torch.stack(self._vlb_reconstruction).mean(),
                on_epoch=True,
            )
            self.log("val/log_pN", torch.stack(self._vlb_log_pn).mean(), on_epoch=True)
            self._vlb_nll.clear()
            self._vlb_kl_prior.clear()
            self._vlb_kl_diffusion.clear()
            self._vlb_reconstruction.clear()
            self._vlb_log_pn.clear()

        if self.evaluator is None or self.sampler is None:
            return

        if self.global_step % self.eval_every_n_steps != 0:
            return

        refs = self.trainer.datamodule.get_reference_graphs(  # pyright: ignore[reportAttributeAccessIssue]
            "val", self.evaluator.eval_num_samples
        )
        if len(refs) < 2:
            return

        generated_graphs = self.generate_graphs(len(refs))

        results = self.evaluator.evaluate(refs=refs, generated=generated_graphs)
        if results is not None:
            for key, value in results.to_dict().items():
                if value is not None:
                    self.log(f"val/gen/{key}", value, on_epoch=True)

        # Full-chain VLB via DiffusionLikelihoodCollector.
        # Runs the complete reverse chain on a subset of reference graphs
        # and sums per-step KL for a lower-variance VLB estimate than the
        # single-timestep estimator used in validation_step.
        collector = DiffusionLikelihoodCollector()
        num_vlb_graphs = min(len(refs), 16)
        self.generate_graphs(num_vlb_graphs, collector=collector)
        vlb_results = collector.results()
        self.log("val/gen/full_chain_vlb", vlb_results["vlb"], on_epoch=True)
        if "mean_kl" in vlb_results:
            self.log("val/gen/mean_step_kl", vlb_results["mean_kl"], on_epoch=True)

    def generate_graphs(
        self,
        num_graphs: int,
        *,
        collector: StepMetricCollector | None = None,
    ) -> list[nx.Graph[Any]]:
        """Generate graphs using the sampler and convert to NetworkX.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.
        collector
            Optional per-step metric collector forwarded to the sampler.

        Returns
        -------
        list[nx.Graph]
            Generated NetworkX graphs.
        """
        if self.sampler is None:
            raise RuntimeError(
                "Cannot generate graphs: no Sampler was provided. "
                "SingleStepDenoisingModule (T=1) does not use reverse sampling."
            )
        device = next(self.parameters()).device

        # Use variable-size generation if the training distribution is non-degenerate
        num_nodes_arg: int | torch.Tensor = self.num_nodes
        if (
            self._size_distribution is not None
            and not self._size_distribution.is_degenerate
        ):
            num_nodes_arg = self._size_distribution.sample(num_graphs)

        graph_data_list = self.sampler.sample(
            model=self.model,
            num_graphs=num_graphs,
            num_nodes=num_nodes_arg,
            device=device,
            collector=collector,
        )

        nx_graphs: list[nx.Graph[Any]] = []
        for gd in graph_data_list:
            adj = gd.to_adjacency()
            if adj.ndim == 3:
                adj = adj[0]
            A_np = adj.cpu().numpy()
            G: nx.Graph[Any] = nx.from_numpy_array(A_np)
            nx_graphs.append(G)

        return nx_graphs
