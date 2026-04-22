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

from tmgg.data.datasets.graph_data_fields import (
    GRAPHDATA_LOSS_KIND,
    FieldName,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.collectors import DiffusionLikelihoodCollector, StepMetricCollector
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ExactDensityNoiseProcess,
    NoiseProcess,
)
from tmgg.diffusion.sampler import Sampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.evaluation.visualization import (
    build_validation_visualizations,
)
from tmgg.models.base import GraphModel
from tmgg.training.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.training.lightning_modules.train_loss_discrete import (
    TrainLossDiscrete,
    masked_edge_ce,
    masked_edge_mse,
    masked_node_ce,
    masked_node_mse,
    masked_y_ce,
    masked_y_mse,
)
from tmgg.training.logging import log_figures
from tmgg.utils.noising.size_distribution import SizeDistribution

_VALID_LOSS_TYPES = frozenset({"cross_entropy", "mse", "bce_logits"})
_DEFAULT_VISUALIZATION = {"enabled": True, "num_samples": 8}

#: Default per-field loss weights for the unified per-field training loop.
#: Edge-side fields carry a 5x weight to reproduce the DiGress (Vignac et al.
#: 2023, Eq. 3) ``lambda_E`` convention; node-side and continuous node fields
#: default to unit weight. A user-supplied ``lambda_per_field`` dict on the
#: constructor merges on top of this table.
#:
#: Graph-level ``y_class`` / ``y_feat`` (parity #27 / #44 / D-13) default
#: to ``0.0`` so the y-term contributes nothing on structure-only datasets
#: (SBM, etc.) where no global target exists. Molecular datasets opt in
#: with a non-zero ``lambda_y`` constructor argument.
_DEFAULT_LAMBDA_PER_FIELD: dict[FieldName, float] = {
    "X_class": 1.0,
    "E_class": 5.0,
    "X_feat": 1.0,
    "E_feat": 5.0,
    "y_class": 0.0,
    "y_feat": 0.0,
}


def _normalize_visualization_config(
    visualization: dict[str, Any] | None,
) -> dict[str, bool | int]:
    """Validate and normalize validation-visualization settings."""
    merged = dict(_DEFAULT_VISUALIZATION)
    if visualization is not None:
        merged.update(visualization)

    enabled = merged["enabled"]
    num_samples = merged["num_samples"]

    if not isinstance(enabled, bool):
        raise TypeError(
            "Visualization config field 'enabled' must be a bool, "
            f"got {type(enabled).__name__}."
        )
    if not isinstance(num_samples, int):
        raise TypeError(
            "Visualization config field 'num_samples' must be an int, "
            f"got {type(num_samples).__name__}."
        )
    if num_samples <= 0 or num_samples % 2 != 0:
        raise ValueError(
            "Visualization config field 'num_samples' must be a positive even integer."
        )

    return {"enabled": enabled, "num_samples": num_samples}


def _read_field(data: GraphData, field: FieldName) -> torch.Tensor:
    """Read a split field from ``data`` or raise if unpopulated.

    Called from the per-field training / validation loop to fetch the
    target tensor for each field declared by the noise process. For
    ``"E_feat"`` we additionally allow lifting an ``E_class`` view into
    a single-channel scalar edge so Gaussian processes on ``E_feat``
    can still operate on categorical batches.
    """
    if field == "X_class":
        if data.X_class is None:
            # Wave 9.3 (structure-only datasets): synthesise a degenerate
            # "[no-node, node]" one-hot from ``node_mask`` so categorical
            # noise processes declaring ``X_class`` can still operate on
            # structure-only batches. See
            # ``docs/specs/2026-04-15-unified-graph-features-spec.md
            # §"Removed fields"`` — this is the "architecture-internal
            # concern" synthesis applied at the training-loop boundary
            # rather than at the dataset boundary.
            #
            # Padding positions emit an **all-zero** row rather than
            # ``[1, 0]`` so the ``(true != 0).any(-1)`` predicate inside
            # :func:`masked_node_ce` correctly excludes them — matching
            # upstream's convention that padding contributes nothing to
            # the loss.
            node_ind = data.node_mask.float()
            synth = torch.stack([1.0 - node_ind, node_ind], dim=-1)
            return synth * node_ind.unsqueeze(-1)
        return data.X_class
    if field == "E_class":
        if data.E_class is None:
            raise ValueError(
                "DiffusionModule._read_field: E_class is None but the noise "
                "process declared 'E_class' as a target field."
            )
        return data.E_class
    if field == "X_feat":
        if data.X_feat is None:
            raise ValueError(
                "DiffusionModule._read_field: X_feat is None but the noise "
                "process declared 'X_feat' as a target field."
            )
        return data.X_feat
    if field == "E_feat":
        if data.E_feat is not None:
            return data.E_feat
        if data.E_class is None:
            raise ValueError(
                "DiffusionModule._read_field: neither E_feat nor E_class is "
                "populated; cannot derive a continuous edge view."
            )
        return data.to_edge_scalar(source="class").unsqueeze(-1)
    if field in ("y_class", "y_feat"):
        # Graph-level fields share the underlying ``y`` tensor on
        # ``GraphData`` (shape ``(bs, dy)``). The split between
        # ``y_class`` (categorical) and ``y_feat`` (continuous) is a
        # loss-dispatch concern carried by ``GRAPHDATA_LOSS_KIND``;
        # the data layer keeps a single tensor for both. Empty-class
        # batches (``y.shape[-1] == 0``, the SBM convention) are
        # handled by the masked CE/MSE helpers' ``numel == 0`` guard.
        return data.y
    raise ValueError(f"_read_field does not support field {field!r}.")


def _continuous_target_edge_state(data: GraphData) -> torch.Tensor:
    """Extract the dense target state for continuous losses.

    Continuous predictions are one-channel edge states. For
    categorical targets we lift ``E_class`` into a scalar adjacency so
    the loss compares semantically equivalent dense states.
    """
    if data.E_feat is not None:
        return data.to_edge_scalar(source="feat")
    return data.to_edge_scalar(source="class")


def _categorical_reconstruction_log_prob(
    clean: GraphData,
    pred_probs: GraphData,
) -> torch.Tensor:
    """Return masked categorical reconstruction log-probabilities per graph."""
    node_mask = clean.node_mask
    inv = ~node_mask
    inv_edge = inv.unsqueeze(1) | inv.unsqueeze(2)

    if clean.E_class is None or pred_probs.E_class is None:
        raise ValueError(
            "_categorical_reconstruction_log_prob requires E_class to be "
            "populated on both `clean` and `pred_probs`."
        )

    # Wave 9.3: structure-only datasets carry X_class=None. Synthesise the
    # degenerate "[no-node, node]" one-hot from node_mask on whichever side
    # (clean or pred_probs) is missing so the reconstruction loss has
    # compatible shapes.
    def _synth_x_class(data: GraphData) -> torch.Tensor:
        if data.X_class is not None:
            return data.X_class
        node_ind = data.node_mask.float()
        return torch.stack([1.0 - node_ind, node_ind], dim=-1)

    clean_x = _synth_x_class(clean).clone()
    clean_e = clean.E_class.clone()
    pred_x = _synth_x_class(pred_probs).clone()
    pred_e = pred_probs.E_class.clone()

    clean_x[inv] = 0.0
    clean_e[inv_edge] = 0.0
    pred_x[inv] = 1.0
    pred_e[inv_edge] = 1.0

    log_prob_x = (
        (clean_x * pred_x.clamp(min=1e-10).log()).flatten(start_dim=1).sum(dim=1)
    )
    log_prob_e = (
        (clean_e * pred_e.clamp(min=1e-10).log()).flatten(start_dim=1).sum(dim=1)
    )
    return log_prob_x + log_prob_e


def _categorical_kl_per_graph(p_pmf: GraphData, q_pmf: GraphData) -> torch.Tensor:
    """Sum analytic categorical KL(p || q) over all node + edge positions per graph.

    Both inputs carry per-position PMFs over the same support
    (``X_class`` for nodes, ``E_class`` for edges). Padded positions
    are replaced with delta + uniform so they contribute zero to the
    KL, matching the pattern in
    :func:`_categorical_reconstruction_log_prob`.

    Returns ``(bs,)`` per-graph KL totals.
    """
    node_mask = p_pmf.node_mask
    inv_node = ~node_mask
    inv_edge = inv_node.unsqueeze(1) | inv_node.unsqueeze(2)

    if (
        p_pmf.X_class is None
        or p_pmf.E_class is None
        or q_pmf.X_class is None
        or q_pmf.E_class is None
    ):
        raise ValueError(
            "_categorical_kl_per_graph requires X_class and E_class on both PMF inputs."
        )

    p_x = p_pmf.X_class.clone()
    q_x = q_pmf.X_class.clone()
    p_e = p_pmf.E_class.clone()
    q_e = q_pmf.E_class.clone()

    # Zero out p at masked positions (zero probability mass means zero KL
    # contribution); set q to a strictly positive constant there so the
    # downstream log() is well-defined.
    p_x[inv_node] = 0.0
    q_x[inv_node] = 1.0
    p_e[inv_edge] = 0.0
    q_e[inv_edge] = 1.0

    eps = 1e-10
    log_term = p_x.clamp(min=eps).log() - q_x.clamp(min=eps).log()
    kl_x = (p_x * log_term).flatten(start_dim=1).sum(dim=1)
    log_term_e = p_e.clamp(min=eps).log() - q_e.clamp(min=eps).log()
    kl_e = (p_e * log_term_e).flatten(start_dim=1).sum(dim=1)
    return kl_x + kl_e


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
        Weight for the edge loss relative to node loss. Default 5.0 per
        DiGress convention. Used only when ``loss_type="cross_entropy"``.
        Kept for backward compatibility with configs that pass it; when
        set, it overrides ``lambda_per_field["E_class"]`` and
        ``lambda_per_field["E_feat"]``.
    lambda_y : float
        Weight for the graph-level loss (parity #27 / #44 / D-13). Default
        ``0.0`` so structure-only datasets (SBM, etc.) match upstream
        DiGress's behaviour bit-for-bit when no global target is
        defined. Molecular and other graph-classification datasets opt
        in by passing ``lambda_y > 0``; the value overrides
        ``lambda_per_field["y_class"]`` and ``lambda_per_field["y_feat"]``.
    lambda_per_field : dict[str, float] | None
        Per-field weights for the unified per-field training loss (Wave
        5.1). Merged on top of the default table
        ``{X_class: 1.0, E_class: 5.0, X_feat: 1.0, E_feat: 5.0,
        y_class: 0.0, y_feat: 0.0}`` so a caller can override only the
        fields they care about. ``None`` keeps the default table.
    num_nodes : int
        Node count used when sampling graphs during evaluation.
    eval_every_n_steps : int
        Generative evaluation runs when ``global_step`` is a multiple of
        this value. Decouples evaluation frequency from batch size and
        dataset size.
    visualization : dict[str, Any] | None
        Validation-visualization settings. ``enabled`` toggles figure
        logging, and ``num_samples`` sets the total number of plotted
        reference/generated graphs across both distributions.
    use_marginalised_vlb_kl : bool, default False
        When False (upstream parity, default), validation_step's KL_t
        term plugs the soft x_0 prediction directly into the Bayes
        posterior formula (matches diffusion_model_discrete.py:340-360).
        When True, marginalises over discrete x_0 classes:
        ``Σ_c p(z_s | z_t, x_0=c) · p_θ(x_0=c | z_t)``. Both estimators
        are unbiased; marginalised has lower variance early in training
        when ``p_θ`` is diffuse, plug-in is the published DiGress form.
    use_upstream_reconstruction : bool, default True
        When True (upstream parity, default), the reconstruction term in
        the VLB samples ``z_0 = x_0 @ Q_0`` and scores the model's softmax
        against the clean target — the published DiGress "first step of
        denoising" reconstruction likelihood
        (diffusion_model_discrete.py:368-405). When False, samples z_1
        and scores via the reverse-chain marginalised posterior; aligns
        with the sampler's actual first reverse step. Numerical gap is
        bounded at ~1e-3 abs for T=1000.
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
        lambda_y: float = 0.0,
        lambda_per_field: dict[str, float] | None = None,
        num_nodes: int = 20,
        eval_every_n_steps: int = 5000,
        visualization: dict[str, Any] | None = None,
        use_marginalised_vlb_kl: bool = False,
        use_upstream_reconstruction: bool = True,
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
        self.visualization: dict[str, bool | int] = _normalize_visualization_config(
            visualization
        )
        self.use_marginalised_vlb_kl: bool = bool(use_marginalised_vlb_kl)
        self.use_upstream_reconstruction: bool = bool(use_upstream_reconstruction)

        # Per-field loss weights: start from the DiGress-compatible default
        # table, merge any user overrides, then have ``lambda_E`` override
        # the two edge-side fields and ``lambda_y`` the two graph-level
        # fields so existing configs that pass only the scalar knobs keep
        # working bit-for-bit. ``lambda_y`` defaults to ``0.0`` so SBM
        # behaviour (no global target) is unchanged.
        merged_lambdas: dict[FieldName, float] = dict(_DEFAULT_LAMBDA_PER_FIELD)
        if lambda_per_field is not None:
            for field_name, weight in lambda_per_field.items():
                if field_name not in _DEFAULT_LAMBDA_PER_FIELD:
                    raise ValueError(
                        f"lambda_per_field key {field_name!r} is not a known "
                        f"FieldName; expected one of "
                        f"{sorted(_DEFAULT_LAMBDA_PER_FIELD)}."
                    )
                merged_lambdas[field_name] = float(weight)  # pyright: ignore[reportArgumentType]
        merged_lambdas["E_class"] = float(lambda_E)
        merged_lambdas["E_feat"] = float(lambda_E)
        merged_lambdas["y_class"] = float(lambda_y)
        merged_lambdas["y_feat"] = float(lambda_y)
        self.lambda_per_field: dict[FieldName, float] = merged_lambdas
        self.lambda_E: float = float(lambda_E)
        self.lambda_y: float = float(lambda_y)

        self.criterion: nn.Module
        self._train_loss_discrete: TrainLossDiscrete | None = None
        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
            self._train_loss_discrete = TrainLossDiscrete(lambda_E=lambda_E)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:  # bce_logits
            self.criterion = nn.BCEWithLogitsLoss()

        # VLB accumulators (only used when the process has exact densities)
        self._vlb_nll: list[torch.Tensor] = []
        self._vlb_kl_prior: list[torch.Tensor] = []
        self._vlb_kl_diffusion: list[torch.Tensor] = []
        self._vlb_reconstruction: list[torch.Tensor] = []
        self._vlb_log_pn: list[torch.Tensor] = []
        self._size_distribution: SizeDistribution | None = None

    @property
    def T(self) -> int:
        """Number of diffusion steps."""
        return self.noise_process.timesteps

    @override
    def setup(self, stage: str | None = None) -> None:
        """Initialise process state and size metadata from the datamodule."""
        dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]

        if self.noise_process.needs_data_initialization():
            if self.noise_process.is_initialized():
                pass
            elif dm is None:
                raise RuntimeError(
                    "DataModule required for noise-process initialisation"
                )
            else:
                self.noise_process.initialize_from_data(
                    dm.train_dataloader_raw_pyg()  # pyright: ignore[reportUnknownMemberType]
                )

        if self._size_distribution is None and isinstance(
            self.noise_process, ExactDensityNoiseProcess
        ):
            if dm is None:
                raise RuntimeError(
                    "DataModule required for exact-density size distribution"
                )
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
        # Read batch size / device from ``node_mask`` rather than the
        # legacy ``batch.X`` view so the module no longer presumes a
        # categorical node tensor is present. ``node_mask`` is required
        # on every ``GraphData`` instance.
        bs: int = int(batch.node_mask.shape[0])
        device = batch.node_mask.device

        # Sample random timestep per batch element: t in {0, ..., T}.
        # Upstream parity (apply_noise, diffusion_model_discrete.py:407-442):
        # training samples t in {0..T} (the t=0 term contributes to training loss),
        # validation samples t in {1..T} (t=0 handled separately by reconstruction_logp).
        t_int = torch.randint(0, self.T + 1, (bs,), device=device)

        # Apply forward noise at the sampled timesteps. ``forward_sample``
        # now returns a ``NoisedBatch`` bundling the noised graph with the
        # schedule scalars (parity #17 / #18 / D-4); only ``z_t`` is fed
        # into the model here.
        noised = self.noise_process.forward_sample(batch, t_int)
        z_t = noised.z_t

        condition = self.noise_process.process_state_condition_vector(t_int)

        # Model predicts clean data
        pred = self.model(z_t, t=condition)

        # Loss against original clean data
        loss = self._compute_loss(pred, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _compute_loss(self, pred: GraphData, target: GraphData) -> torch.Tensor:
        """Compute loss by iterating over ``self.noise_process.fields``.

        Wave 5.1 replaces the previous hardcoded ``batch.X`` / ``batch.E``
        dual-field path with a per-field loop. For each declared field:

        * ``X_class`` / ``E_class`` (``GRAPHDATA_LOSS_KIND == "ce"``) use
          masked cross-entropy on raw logits via :func:`masked_node_ce` /
          :func:`masked_edge_ce`. Targets are zeroed at padding positions
          first (matching upstream DiGress's
          ``dense_data.mask(node_mask)`` step, see
          ``digress-upstream-readonly/src/diffusion_model_discrete.py:108``)
          so the ``(true != 0).any(-1)`` row predicate inside the helpers
          drops padding and diagonal rows automatically.
        * ``X_feat`` / ``E_feat`` (``GRAPHDATA_LOSS_KIND == "mse"``) use
          masked MSE; these helpers consume ``node_mask`` directly.

        Each term is weighted by ``self.lambda_per_field[field]``
        (defaults give the DiGress ``lambda_E=5.0`` edge weighting) and
        summed.

        When ``loss_type`` is ``"mse"`` or ``"bce_logits"`` the module
        keeps its original dense-edge-state behaviour because those
        single-channel continuous losses operate directly on the
        dense scalar edge view from :meth:`GraphData.to_edge_scalar`.
        """
        if self._train_loss_discrete is None:
            # MSE / BCE: compare dense edge states via the split edge
            # fields. This path predates the per-field iteration and
            # does not go through ``lambda_per_field``.
            if pred.E_feat is not None:
                pred_edge_state = pred.to_edge_scalar(source="feat")
            else:
                pred_edge_state = pred.to_edge_scalar(source="class")
            target_edge_state = _continuous_target_edge_state(target)
            return self.criterion(pred_edge_state, target_edge_state.float())

        # Zero padding rows in the target categorical fields so the row
        # predicate inside the CE helpers correctly excludes them. Matches
        # upstream's ``dense_data.mask(node_mask)`` call that happens before
        # ``train_loss(...)``.
        target = target.mask()

        total: torch.Tensor | None = None
        for field in sorted(self.noise_process.fields):
            pred_field = _read_field(pred, field)
            target_field = _read_field(target, field)
            kind = GRAPHDATA_LOSS_KIND[field]
            weight = self.lambda_per_field[field]
            if kind == "ce":
                # Raw logits pass straight through: the fused log_softmax
                # inside F.cross_entropy inside the helpers handles
                # normalisation, matching upstream DiGress
                # (`src/metrics/train_metrics.py:95-102`).
                if field == "X_class":
                    term = masked_node_ce(pred_field, target_field, target.node_mask)
                elif field == "E_class":
                    term = masked_edge_ce(pred_field, target_field, target.node_mask)
                elif field == "y_class":
                    # Graph-level CE — no spatial mask; mirrors upstream
                    # ``loss_y`` in train_metrics.py:62-123. Weight is
                    # ``lambda_y`` (default 0.0 ⇒ no contribution on SBM).
                    term = masked_y_ce(pred_field, target_field)
                else:  # pragma: no cover - all CE fields handled above
                    raise NotImplementedError(
                        f"No masked CE helper registered for categorical field "
                        f"{field!r}."
                    )
            else:  # kind == "mse"
                if field == "X_feat":
                    term = masked_node_mse(pred_field, target_field, target.node_mask)
                elif field == "E_feat":
                    term = masked_edge_mse(pred_field, target_field, target.node_mask)
                elif field == "y_feat":
                    # Graph-level MSE — sister of ``masked_y_ce`` for
                    # continuous global targets. Weight is ``lambda_y``
                    # (default 0.0 ⇒ no contribution on SBM).
                    term = masked_y_mse(pred_field, target_field)
                else:  # pragma: no cover - all MSE fields handled above
                    raise NotImplementedError(
                        f"No masked MSE helper registered for continuous field "
                        f"{field!r}."
                    )
            contribution = weight * term
            total = contribution if total is None else total + contribution

        if total is None:  # pragma: no cover - fields is non-empty by invariant
            raise RuntimeError(
                "DiffusionModule._compute_loss: noise_process.fields is empty; "
                "cannot compute a cross-entropy loss."
            )
        return total

    def _compute_reconstruction(self, batch: GraphData) -> torch.Tensor:
        """Reconstruction log-probability ``log p(x_0 | z_1)``.

        Aligns with upstream DiGress's ``reconstruction_logp``: noise the
        clean batch to ``z_1``, run the model, and **pull the predicted
        ``x_0`` distribution through the per-class marginalised reverse
        kernel** at ``(t=1, s=0)`` to obtain ``p(x_0 = . | z_1)`` rather
        than scoring the raw softmax. Locally pre-2026-04-15 we scored
        the softmax directly (``_compute_reconstruction_at_t1``); the
        difference is bounded by the Q_1 transition kernel ≈ identity at
        T=1000 (a ~1e-3 shift in absolute magnitude) but matters for
        numerical parity with published DiGress results.
        """
        if not isinstance(self.noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"_compute_reconstruction requires CategoricalNoiseProcess, "
                f"got {type(self.noise_process).__name__}"
            )
        bs = int(batch.node_mask.shape[0])
        device = batch.node_mask.device

        t_int = torch.ones(bs, dtype=torch.long, device=device)
        s_int = torch.zeros_like(t_int)

        z_1 = self.noise_process.forward_sample(batch, t_int).z_t
        condition = self.noise_process.process_state_condition_vector(t_int)
        pred_logits = self.model(z_1, t=condition)
        x0_param = self.noise_process.model_output_to_posterior_parameter(pred_logits)

        # ``_posterior_probabilities_marginalised`` returns the
        # per-position PMF over ``z_s = x_0`` (since s=0). We score the
        # one-hot clean batch under this PMF.
        recon_pmf = self.noise_process._posterior_probabilities_marginalised(  # pyright: ignore[reportPrivateUsage]
            z_1, x0_param, t_int, s_int
        )
        return _categorical_reconstruction_log_prob(batch, recon_pmf)

    def _compute_reconstruction_upstream_style(self, batch: GraphData) -> torch.Tensor:
        """Reconstruction log-probability in upstream DiGress's form.

        Direct port of
        ``digress-upstream-readonly/src/diffusion_model_discrete.py::
        reconstruction_logp`` (lines 368-405): sample
        ``z_0 = x_0 @ Q_0`` (one-hot through the t=0 transition, which is
        ≈ identity for cosine-IDDPM at T=1000), run the model, and score
        the raw softmax against the clean target. This is the published
        "first step of denoising" reconstruction likelihood. The
        marginalised reverse-posterior path (see
        :meth:`_compute_reconstruction`) integrates over an additional
        Q_1 step instead; the gap is ~1e-3 abs at T=1000 (parity #31).
        """
        if not isinstance(self.noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"_compute_reconstruction_upstream_style requires "
                f"CategoricalNoiseProcess, got {type(self.noise_process).__name__}"
            )
        bs = int(batch.node_mask.shape[0])
        device = batch.node_mask.device

        # Upstream samples z_0 by drawing from x_0 @ Q_0; ``forward_sample``
        # at t=0 does exactly that (forward noising at t=0 is the Q_0
        # mixing kernel) and returns a one-hot ``z_0`` ready for the model.
        t_zero = torch.zeros(bs, dtype=torch.long, device=device)
        z_0 = self.noise_process.forward_sample(batch, t_zero).z_t
        condition = self.noise_process.process_state_condition_vector(t_zero)
        pred_logits = self.model(z_0, t=condition)
        # ``model_output_to_posterior_parameter`` already applies softmax
        # along the class axis on both X_class and E_class.
        pred_probs = self.noise_process.model_output_to_posterior_parameter(pred_logits)
        return _categorical_reconstruction_log_prob(batch, pred_probs)

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
        bs: int = int(batch.node_mask.shape[0])
        device = batch.node_mask.device

        # Validation loss at a random timestep
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)
        z_t = self.noise_process.forward_sample(batch, t_int).z_t
        condition = self.noise_process.process_state_condition_vector(t_int)
        pred = self.model(z_t, t=condition)
        loss = self._compute_loss(pred, batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # VLB estimation via single random timestep (standard DDPM
        # approach). Each batch samples one ``t`` and computes ``L_t``;
        # the epoch-end average gives an estimate of ``E_t[L_t]`` that
        # is unbiased in ``t`` and exact (closed-form categorical KL)
        # for each individual ``L_t``. This matches the DiGress
        # reference implementation (cvignac/DiGress,
        # ``src/diffusion_model.py::compute_val_loss``). For full-chain
        # VLB, see the periodic ``DiffusionLikelihoodCollector``
        # evaluation in :meth:`on_validation_epoch_end`.
        if isinstance(self.noise_process, CategoricalNoiseProcess):
            cat_process = self.noise_process

            x0_param = self.noise_process.model_output_to_posterior_parameter(pred)
            s_int = t_int - 1

            # Analytic KL_t = KL(q(z_s|z_t,x_0) || p_theta(z_s|z_t)).
            # The "true" posterior always plugs the clean batch (one-hot
            # x_0) into the Bayes formula. The "predicted" posterior
            # selection follows ``use_marginalised_vlb_kl`` (D-6):
            #
            # * False (upstream parity, default): plug the soft x_0
            #   prediction directly into the same Bayes posterior formula
            #   (diffusion_model_discrete.py:340-360).
            # * True: use the marginalised form
            #   ``sum_c p(z_s|z_t,x_0=c) * pred[c]`` — the distribution
            #   the sampler actually draws from (see the Phase C sampler
            #   change). Lower variance early in training; coincides with
            #   the plug-in form once ``p_θ`` is one-hot.
            true_posterior = cat_process._posterior_probabilities(  # pyright: ignore[reportPrivateUsage]
                z_t, batch, t_int, s_int
            )
            if self.use_marginalised_vlb_kl:
                pred_posterior = cat_process._posterior_probabilities_marginalised(  # pyright: ignore[reportPrivateUsage]
                    z_t, x0_param, t_int, s_int
                )
            else:
                pred_posterior = cat_process._posterior_probabilities(  # pyright: ignore[reportPrivateUsage]
                    z_t, x0_param, t_int, s_int
                )
            kl_diffusion = self.T * _categorical_kl_per_graph(
                true_posterior, pred_posterior
            )

            # Analytic KL(q(z_T|x_0) || prior). At T=1000 with the
            # cosine-IDDPM schedule, ``q(z_T|x_0) ≈ stationary prior``
            # so this term is small but non-zero; computing it
            # analytically removes the per-sample MC variance the
            # log-prob-of-sample formulation carried before.
            t_T = torch.full((bs,), self.T, device=device, dtype=torch.long)
            forward_pmf_T = cat_process.forward_pmf(batch, t_T)
            prior_pmf = cat_process.prior_pmf(batch.node_mask)
            kl_prior = _categorical_kl_per_graph(forward_pmf_T, prior_pmf)

            # Reconstruction term selection follows ``use_upstream_reconstruction``
            # (D-7). Default True selects the upstream-style z_0 + raw
            # softmax form; False keeps the marginalised z_1 form.
            if self.use_upstream_reconstruction:
                reconstruction = self._compute_reconstruction_upstream_style(batch)
            else:
                reconstruction = self._compute_reconstruction(batch)

            if self._size_distribution is None:
                raise RuntimeError(
                    "DiffusionModule.validation_step (categorical branch): "
                    "_size_distribution is None. The datamodule must populate "
                    "a SizeDistribution in setup() — see SpectreSBMDataModule / "
                    "SyntheticCategoricalDataModule."
                )
            node_counts = batch.node_mask.sum(dim=-1).long()  # (bs,)
            log_pn = self._size_distribution.log_prob(node_counts).mean()

            # NLL = -log_pN + kl_prior + E_t[L_t] - reconstruction_logp
            nll = (
                -log_pn + kl_prior.mean() + kl_diffusion.mean() - reconstruction.mean()
            )

            self._vlb_nll.append(nll.detach())
            self._vlb_kl_prior.append(kl_prior.mean().detach())
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())
            self._vlb_reconstruction.append(reconstruction.mean().detach())
            self._vlb_log_pn.append(log_pn.detach())
        elif isinstance(self.noise_process, ExactDensityNoiseProcess):
            # Continuous (Gaussian) noise processes still use the
            # log-prob-on-sample VLB; the analytic categorical-KL path
            # above is specific to categorical PMFs.
            exact_process = self.noise_process
            x0_param = self.noise_process.model_output_to_posterior_parameter(pred)
            s_int = t_int - 1
            z_s = exact_process.posterior_sample(z_t, batch, t_int, s_int)
            log_q_true = exact_process.posterior_log_prob(z_s, z_t, batch, t_int, s_int)
            log_q_pred = exact_process.posterior_log_prob(
                z_s, z_t, x0_param, t_int, s_int
            )
            kl_diffusion = self.T * (log_q_true - log_q_pred)

            t_T = torch.full((bs,), self.T, device=device, dtype=torch.long)
            z_T = exact_process.forward_sample(batch, t_T).z_t
            kl_prior = exact_process.forward_log_prob(
                z_T, batch, t_T
            ) - exact_process.prior_log_prob(z_T)

            if self._size_distribution is None:
                raise RuntimeError(
                    "DiffusionModule.validation_step (continuous branch): "
                    "_size_distribution is None. The datamodule must populate "
                    "a SizeDistribution in setup() — see SpectreSBMDataModule / "
                    "SyntheticCategoricalDataModule."
                )
            node_counts = batch.node_mask.sum(dim=-1).long()
            log_pn = self._size_distribution.log_prob(node_counts).mean()

            nll = -log_pn + kl_prior.mean() + kl_diffusion.mean()

            self._vlb_nll.append(nll.detach())
            self._vlb_kl_prior.append(kl_prior.mean().detach())
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())
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
        # Log VLB metrics for exact-density processes.
        if self._vlb_nll:
            self.log("val/epoch_NLL", torch.stack(self._vlb_nll).mean(), on_epoch=True)
            self.log(
                "val/kl_prior", torch.stack(self._vlb_kl_prior).mean(), on_epoch=True
            )
            self.log(
                "val/kl_diffusion",
                torch.stack(self._vlb_kl_diffusion).mean(),
                on_epoch=True,
            )
            # Wave 2.5: the ExactDensity branch (Gaussian) does not yet populate
            # _vlb_reconstruction; Wave 5 will close this by wiring per-field
            # reconstruction log-probs for Gaussian. Until then, only the
            # categorical branch contributes here.
            if self._vlb_reconstruction:
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

        # Skip generative evaluation on an untrained model. ``global_step == 0``
        # covers Lightning's initial sanity-check pass (where ``global_step``
        # is 0 and the model is fresh) and any other pre-training invocation.
        # Without this guard the ``eval_every_n_steps`` modulo gate passes
        # trivially (``0 % N == 0``), causing us to sample from a random-init
        # model and log meaningless metrics.
        if self.global_step == 0:
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
            if self.visualization["enabled"]:
                figures = build_validation_visualizations(
                    refs=refs,
                    generated=generated_graphs,
                    num_samples=int(self.visualization["num_samples"]),
                )
                log_figures(
                    self.trainer.loggers,  # pyright: ignore[reportAttributeAccessIssue]
                    figures,
                    global_step=self.global_step,
                )

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
            noise_process=self.noise_process,
            num_graphs=num_graphs,
            num_nodes=num_nodes_arg,
            device=device,
            collector=collector,
        )

        # Binarisation lives on the evaluator so the threshold and the
        # ``E_class`` / ``E_feat`` disagreement warning (Wave 6.1) stay
        # configurable per-run. See
        # ``docs/specs/2026-04-15-unified-graph-features-spec.md``
        # §"Evaluator contract".
        if self.evaluator is None:
            raise RuntimeError(
                "DiffusionModule.generate_graphs requires a GraphEvaluator "
                "to derive a binary adjacency from sampled GraphData. "
                "Configure ``evaluator`` on the module (see Wave 6.1 of "
                "docs/specs/2026-04-15-unified-graph-features-spec.md)."
            )
        return self.evaluator.to_networkx_graphs(graph_data_list)
