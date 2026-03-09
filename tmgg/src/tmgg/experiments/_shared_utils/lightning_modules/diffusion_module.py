"""Multi-step diffusion training loop for graph generation.

``DiffusionModule`` extends :class:`BaseGraphModule` with a complete
discrete or continuous diffusion pipeline. It composes four injected
components -- a :class:`~tmgg.diffusion.NoiseProcess` (forward diffusion),
a :class:`~tmgg.diffusion.Sampler` (reverse sampling),
a :class:`~tmgg.diffusion.NoiseSchedule` (timestep-to-noise mapping),
and a :class:`~tmgg.experiments._shared_utils.graph_evaluator.GraphEvaluator`
(generation quality metrics) -- and wires them into Lightning's
``training_step`` / ``validation_step`` / ``on_validation_epoch_end`` hooks.
"""

from __future__ import annotations

from typing import Any, override

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.size_distribution import SizeDistribution
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    NoiseProcess,
)
from tmgg.diffusion.sampler import Sampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.experiments._shared_utils.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.experiments._shared_utils.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.experiments._shared_utils.lightning_modules.train_loss_discrete import (
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
    eval_every_n_epochs : int
        Generative evaluation is run every *n* validation epochs.
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
        # Diffusion-specific params
        noise_process: NoiseProcess,
        sampler: Sampler | None = None,
        noise_schedule: NoiseSchedule,
        evaluator: GraphEvaluator | None = None,
        loss_type: str = "cross_entropy",
        lambda_E: float = 5.0,
        num_nodes: int = 20,
        eval_every_n_epochs: int = 5,
    ) -> None:
        super().__init__(
            model_type=model_type,
            model_config=model_config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
        )
        # Re-save hparams at the DiffusionModule level, excluding objects
        # that cannot be reliably pickled for checkpoint reconstruction.
        self.save_hyperparameters(  # pyright: ignore[reportUnknownMemberType]
            ignore=["noise_process", "sampler", "noise_schedule", "evaluator"]
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
        self.eval_every_n_epochs: int = eval_every_n_epochs

        self.criterion: nn.Module
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

            dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
            if dm is None:
                raise RuntimeError("DataModule required for marginal transition setup")
            x_marginals = dm.node_marginals  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            e_marginals = dm.edge_marginals  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            transition = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.noise_process.y_classes,
            )  # pyright: ignore[reportUnknownArgumentType]
            self.noise_process.set_transition_model(transition)

        # Fetch size distribution from the datamodule for VLB log_pN term
        if self._size_distribution is None:
            dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
            if dm is not None and hasattr(dm, "get_size_distribution"):
                self._size_distribution = dm.get_size_distribution("train")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @override
    def on_validation_epoch_start(self) -> None:
        """Clear VLB accumulators at the start of each validation epoch."""
        self._vlb_nll.clear()
        self._vlb_kl_prior.clear()
        self._vlb_kl_diffusion.clear()
        self._vlb_reconstruction.clear()
        self._vlb_log_pn.clear()

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:  # type: ignore[override]
        """Forward pass: delegates to the underlying ``GraphModel``."""
        return self.model(data, t=t)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @override
    def training_step(self, batch: GraphData, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
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
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)  # pyright: ignore[reportUnknownMemberType]

        # Apply forward noise at the sampled timesteps
        z_t = self.noise_process.apply(batch, t_int)

        # Normalised timestep for model conditioning: t_norm = t/T ∈ [1/T, 1].
        # This is a linear rescaling of the integer index, NOT the noise level
        # (1 - alpha_bar), which follows the schedule's non-linear curve.
        # The model learns to map this linear position to the correct
        # denoising prediction regardless of schedule shape.
        t_norm = t_int.float() / self.T

        # Model predicts clean data
        pred = self.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # Loss against original clean data
        loss = self._compute_loss(pred, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)  # pyright: ignore[reportUnknownMemberType]
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
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # TrainLossDiscrete expects softmaxed probabilities, not logits.
            # Clone before masking: mask_distributions modifies tensors in-place.
            pred_X = F.softmax(pred.X, dim=-1).clone()
            pred_E = F.softmax(pred.E, dim=-1).clone()
            return self._train_loss_discrete(
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
        assert isinstance(self.noise_process, CategoricalNoiseProcess)
        bs = batch.X.shape[0]
        device = batch.X.device

        # Apply noise at t=1
        t_int = torch.ones(bs, dtype=torch.long, device=device)
        z_1 = self.noise_process.apply(batch, t_int)

        # Model prediction at t=1
        t_norm = t_int.float() / self.T
        pred_logits = self.model(z_1, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # Softmax to probabilities
        pred_probs = GraphData(
            X=F.softmax(pred_logits.X, dim=-1),  # pyright: ignore[reportUnknownMemberType]
            E=F.softmax(pred_logits.E, dim=-1),  # pyright: ignore[reportUnknownMemberType]
            y=pred_logits.y,  # pyright: ignore[reportUnknownMemberType]
            node_mask=pred_logits.node_mask,  # pyright: ignore[reportUnknownMemberType]
        )

        # Mask invalid positions so they contribute nothing to the log-prob.
        # reconstruction_logp computes sum(clean * log(pred)), so we need:
        #   clean[invalid] = 0  (zero contribution from numerator)
        #   pred[invalid] = 1   (log(1) = 0, avoids 0 * -inf = NaN)
        node_mask = batch.node_mask
        if node_mask is not None:
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
    def validation_step(self, batch: GraphData, batch_idx: int) -> None:  # type: ignore[override]
        """Validate on a batch, accumulate reference graphs for evaluation.

        Computes validation loss at a random timestep and, when an
        evaluator is attached, converts the batch to NetworkX graphs
        for later generative evaluation.

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
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)  # pyright: ignore[reportUnknownMemberType]
        z_t = self.noise_process.apply(batch, t_int)
        t_norm = t_int.float() / self.T
        pred = self.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        loss = self._compute_loss(pred, batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)  # pyright: ignore[reportUnknownMemberType]

        # VLB computation for categorical noise processes
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
            kl_prior = (kl_prior_X + kl_prior_E) / bs  # pyright: ignore[reportUnknownVariableType]
            kl_diffusion = self.noise_process.compute_Lt(batch, pred_probs, z_t, t_int)
            reconstruction = self._compute_reconstruction_at_t1(batch)

            # log p(n_G): log-probability of graph size under training distribution
            log_pn = torch.zeros(1, device=device)
            if self._size_distribution is not None:
                node_counts = batch.node_mask.sum(dim=-1).long()  # (bs,)
                log_pn = self._size_distribution.log_prob(node_counts).mean()

            # NLL = -log_pN + kl_prior + E_t[L_t] - reconstruction_logp
            # kl_diffusion and reconstruction are (bs,); kl_prior is scalar
            nll = -log_pn + kl_prior + kl_diffusion.mean() - reconstruction.mean()  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

            self._vlb_nll.append(nll.detach())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            self._vlb_kl_prior.append(kl_prior.detach())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())  # pyright: ignore[reportUnknownMemberType]
            self._vlb_reconstruction.append(reconstruction.mean().detach())  # pyright: ignore[reportUnknownMemberType]
            self._vlb_log_pn.append(log_pn.detach())

        # Accumulate reference graphs for evaluator
        if self.evaluator is not None:
            adj = batch.to_adjacency()
            for i in range(int(bs)):  # pyright: ignore[reportUnknownArgumentType]
                n = int(batch.node_mask[i].sum().item())  # pyright: ignore[reportUnknownMemberType]
                A_np = adj[i, :n, :n].cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
                G: nx.Graph[Any] = nx.from_numpy_array(A_np)  # pyright: ignore[reportExplicitAny]
                self.evaluator.accumulate(G)

    @torch.no_grad()
    @override
    def test_step(self, batch: GraphData, batch_idx: int) -> None:  # type: ignore[override]
        """Test step -- delegates to ``validation_step``."""
        self.validation_step(batch, batch_idx)

    @override
    def on_validation_epoch_end(self) -> None:
        """Run generative evaluation at configured intervals.

        Generates as many graphs as accumulated reference graphs, evaluates
        them through the ``GraphEvaluator``, logs all metrics, and clears
        the reference buffer. Skipped when ``sampler`` is ``None`` (e.g.
        single-step denoising) or no evaluator is attached.
        """
        # Log VLB metrics for categorical noise
        if self._is_categorical and self._vlb_nll:
            self.log("val/epoch_NLL", torch.stack(self._vlb_nll).mean(), on_epoch=True)  # pyright: ignore[reportUnknownMemberType]
            self.log(  # pyright: ignore[reportUnknownMemberType]
                "val/kl_prior", torch.stack(self._vlb_kl_prior).mean(), on_epoch=True
            )
            self.log(  # pyright: ignore[reportUnknownMemberType]
                "val/kl_diffusion",
                torch.stack(self._vlb_kl_diffusion).mean(),
                on_epoch=True,
            )
            self.log(  # pyright: ignore[reportUnknownMemberType]
                "val/reconstruction_logp",
                torch.stack(self._vlb_reconstruction).mean(),
                on_epoch=True,
            )
            self.log("val/log_pN", torch.stack(self._vlb_log_pn).mean(), on_epoch=True)  # pyright: ignore[reportUnknownMemberType]
            self._vlb_nll.clear()
            self._vlb_kl_prior.clear()
            self._vlb_kl_diffusion.clear()
            self._vlb_reconstruction.clear()
            self._vlb_log_pn.clear()

        if self.evaluator is None or self.sampler is None:
            return

        current_epoch: int = self.current_epoch
        if current_epoch % self.eval_every_n_epochs != 0:
            self.evaluator.clear()
            return

        num_samples = len(self.evaluator.ref_graphs)
        if num_samples < 2:
            self.evaluator.clear()
            return

        generated_graphs = self.generate_graphs(num_samples)

        results = self.evaluator.evaluate(generated_graphs)
        if results is not None:
            for key, value in results.to_dict().items():
                if value is not None:
                    self.log(f"val/gen/{key}", value, on_epoch=True)  # pyright: ignore[reportUnknownMemberType]

        self.evaluator.clear()

    def generate_graphs(self, num_graphs: int) -> list[nx.Graph[Any]]:  # pyright: ignore[reportExplicitAny]
        """Generate graphs using the sampler and convert to NetworkX.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.

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
            model=self.model,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            num_graphs=num_graphs,
            num_nodes=num_nodes_arg,
            device=device,
        )

        nx_graphs: list[nx.Graph[Any]] = []  # pyright: ignore[reportExplicitAny]
        for gd in graph_data_list:
            adj = gd.to_adjacency()
            if adj.ndim == 3:  # pyright: ignore[reportUnknownMemberType]
                adj = adj[0]
            A_np = adj.cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
            G: nx.Graph[Any] = nx.from_numpy_array(A_np)  # pyright: ignore[reportExplicitAny]
            nx_graphs.append(G)

        return nx_graphs
