"""Discrete denoising diffusion LightningModule for graph generation.

Wires the building blocks (noise schedule, transition matrices, transformer,
extra features) into a training/validation/sampling loop following DiGress-
style discrete diffusion: cross-entropy training loss, variational lower
bound for validation, and ancestral categorical sampling for generation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from tmgg.experiment_utils.mmd_evaluator import MMDEvaluator
from tmgg.experiment_utils.model_logging import log_parameter_count
from tmgg.experiment_utils.optimizer_config import (
    OptimizerLRSchedulerConfig,
    SchedulerInfo,
    configure_optimizers_from_config,
)
from tmgg.experiments.discrete_diffusion_generative.evaluate import (
    categorical_samples_to_graphs,
)
from tmgg.models.digress.diffusion_utils import (
    PlaceHolder,
    compute_batched_over0_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    sample_discrete_feature_noise,
    sample_discrete_features,
    sum_except_batch,
)
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
from tmgg.models.digress.extra_features import DummyExtraFeatures, ExtraFeatures
from tmgg.models.digress.noise_schedule import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
)
from tmgg.models.digress.train_loss import TrainLossDiscrete


class DiscreteDiffusionLightningModule(pl.LightningModule):
    """LightningModule for discrete graph denoising diffusion.

    Implements the DiGress training loop: forward diffusion applies categorical
    noise via transition matrices, the denoising transformer predicts the clean
    distribution, and training minimises masked cross-entropy. Validation
    computes the full variational lower bound. Sampling uses ancestral
    categorical reverse steps.

    The transition model is constructed in ``setup()`` rather than ``__init__``
    because ``MarginalUniformTransition`` requires the datamodule's empirical
    marginals, which are only available after ``DataModule.setup()``.

    Parameters
    ----------
    model
        Pre-built ``DiscreteGraphTransformer`` backbone.
    noise_schedule
        Predefined discrete noise schedule (cosine or custom).
    diffusion_steps
        Number of forward diffusion steps T.
    transition_type
        ``"uniform"`` or ``"marginal"`` transition matrices.
    extra_features
        Feature augmentation callable. Defaults to ``DummyExtraFeatures``
        (zero-width output).
    lambda_E
        Edge loss weight relative to node loss.
    learning_rate, weight_decay, optimizer_type, amsgrad, scheduler_config
        Forwarded to ``configure_optimizers_from_config``.
    visualization_interval
        Steps between visualisation logs.
    """

    # Set by setup(); pyright cannot see the deferred initialisation.
    transition_model: DiscreteUniformTransition | MarginalUniformTransition  # pyright: ignore[reportUninitializedInstanceVariable]
    limit_dist: PlaceHolder  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(
        self,
        model: DiscreteGraphTransformer,
        noise_schedule: PredefinedNoiseScheduleDiscrete,
        diffusion_steps: int,
        transition_type: str = "marginal",
        extra_features: DummyExtraFeatures | ExtraFeatures | None = None,
        lambda_E: float = 5.0,
        # Optimizer configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-12,
        optimizer_type: str = "adamw",
        amsgrad: bool = True,
        scheduler_config: dict[str, Any] | None = None,
        # Logging
        visualization_interval: int = 5000,
        # Evaluation
        eval_num_samples: int = 100,
        mmd_kernel: str = "gaussian_tv",
        mmd_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        if transition_type not in ("uniform", "marginal"):
            raise ValueError(
                f"transition_type must be 'uniform' or 'marginal', got {transition_type!r}"
            )

        self.model = model
        self.noise_schedule = noise_schedule
        self.T = diffusion_steps
        self.transition_type = transition_type
        self.extra_features = extra_features or DummyExtraFeatures()

        # Derive output dimensions from the model
        self.dx = model.output_dims["X"]
        self.de = model.output_dims["E"]
        self.dy = model.output_dims["y"]

        # Training loss
        self.train_loss_fn = TrainLossDiscrete(lambda_E=lambda_E)

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.amsgrad = amsgrad
        self.scheduler_config = scheduler_config

        self.visualization_interval = visualization_interval

        # Evaluation parameters
        self.eval_num_samples = eval_num_samples
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma

        # MMD evaluation
        self.mmd_evaluator = MMDEvaluator(
            eval_num_samples=eval_num_samples,
            kernel=mmd_kernel,
            sigma=mmd_sigma,
        )

        # Scheduler info for logging (set by configure_optimizers)
        self._scheduler_info: SchedulerInfo | None = None

        # VLB accumulators for validation epoch aggregation
        self._val_nll: list[Tensor] = []
        self._val_kl_prior: list[Tensor] = []
        self._val_kl_diffusion: list[Tensor] = []
        self._val_reconstruction: list[Tensor] = []

        self._transition_initialized: bool = False

        self.save_hyperparameters(ignore=["model", "noise_schedule", "extra_features"])

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        """Configure optimizers and learning rate schedulers."""
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

    # ------------------------------------------------------------------
    # Setup: construct transition model from datamodule marginals
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Construct the transition model using datamodule marginals.

        Idempotent: subsequent calls are no-ops once the transition model
        is built. For ``"marginal"`` transitions, the trainer's datamodule
        must be attached and set up before this is called (Lightning
        guarantees that).
        """
        if self._transition_initialized:
            return

        if self.transition_type == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.dx,
                e_classes=self.de,
                y_classes=self.dy,
            )
        else:
            if self.trainer is None:
                raise RuntimeError(
                    "Trainer must be attached for marginal transition setup"
                )
            dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]  # Lightning wiring
            if dm is None:
                raise RuntimeError(
                    "DataModule must be attached for marginal transition setup"
                )

            node_marginals: Tensor = dm.node_marginals  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # typed by DataModule
            edge_marginals: Tensor = dm.edge_marginals  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # typed by DataModule
            if len(node_marginals) != self.dx:
                raise RuntimeError(
                    f"node_marginals length {len(node_marginals)} != dx {self.dx}"
                )
            if len(edge_marginals) != self.de:
                raise RuntimeError(
                    f"edge_marginals length {len(edge_marginals)} != de {self.de}"
                )

            self.transition_model = MarginalUniformTransition(
                x_marginals=node_marginals,
                e_marginals=edge_marginals,
                y_classes=self.dy,
            )

        self.limit_dist = self.transition_model.get_limit_dist()
        self._transition_initialized = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_data: dict[str, Tensor],
        extra_data: PlaceHolder,
        node_mask: Tensor,
    ) -> PlaceHolder:
        """Predict clean graph probabilities from noisy data.

        Concatenates noisy features with extra features, appends the
        normalised timestep to global features, runs the transformer,
        and returns softmax-normalised node and edge predictions.

        Returns
        -------
        PlaceHolder
            ``.X``: ``(bs, n, dx)`` node class probabilities.
            ``.E``: ``(bs, n, n, de)`` edge class probabilities.
            ``.y``: ``(bs, dy)`` raw global features (no softmax).
        """
        X = torch.cat([noisy_data["X_t"], extra_data.X], dim=-1).float()
        E = torch.cat([noisy_data["E_t"], extra_data.E], dim=-1).float()
        y = torch.hstack([noisy_data["y_t"], extra_data.y, noisy_data["t"]]).float()

        raw = self.model(X, E, y, node_mask)

        return PlaceHolder(
            X=F.softmax(raw.X, dim=-1),
            E=F.softmax(raw.E, dim=-1),
            y=raw.y,
        )

    # ------------------------------------------------------------------
    # Noise application
    # ------------------------------------------------------------------

    def apply_noise(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> dict[str, Tensor]:
        """Sample a timestep, diffuse X and E via transition matrices.

        During training, ``t`` is sampled from ``[0, T]``; during evaluation,
        from ``[1, T]`` (the ``t=0`` reconstruction term is computed
        separately).

        Returns
        -------
        dict[str, Tensor]
            Keys: ``t_int``, ``t``, ``beta_t``, ``alpha_s_bar``,
            ``alpha_t_bar``, ``X_t``, ``E_t``, ``y_t``, ``node_mask``.
        """
        bs = X.size(0)

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(bs, 1), device=X.device
        ).float()
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        if not (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all():
            raise RuntimeError(
                f"Qtb.X rows don't sum to 1: max dev "
                f"{(Qtb.X.sum(dim=2) - 1).abs().max().item():.2e}"
            )
        if not (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all():
            raise RuntimeError(
                f"Qtb.E rows don't sum to 1: max dev "
                f"{(Qtb.E.sum(dim=2) - 1).abs().max().item():.2e}"
            )

        # Forward diffusion: multiply one-hot by cumulative transition matrix
        probX = X @ Qtb.X  # (bs, n, dx)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de)

        sampled_t = sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.dx).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime
        E_t = F.one_hot(sampled_t.E, num_classes=self.de).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime
        if X.shape != X_t.shape:
            raise RuntimeError(f"X shape {X.shape} != X_t shape {X_t.shape}")
        if E.shape != E_t.shape:
            raise RuntimeError(f"E shape {E.shape} != E_t shape {E_t.shape}")

        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

    def compute_extra_data(self, noisy_data: dict[str, Tensor]) -> PlaceHolder:
        """Compute extra features to append to the transformer input."""
        extra_X, extra_E, extra_y = self.extra_features(
            noisy_data["X_t"],
            noisy_data["E_t"],
            noisy_data["y_t"],
            noisy_data["node_mask"],
        )
        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Forward diffusion, neural-net prediction, cross-entropy loss."""
        X, E, y, node_mask = batch

        noisy_data = self.apply_noise(X, E, y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Clone predictions: mask_distributions (called inside train_loss_fn)
        # modifies tensors in-place, which would corrupt the autograd graph
        # since pred.X and pred.E are softmax outputs.
        loss = self.train_loss_fn(
            pred_X=pred.X.clone(),
            pred_E=pred.E.clone(),
            true_X=X,
            true_E=E,
            node_mask=node_mask,
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation (Variational Lower Bound)
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        """Compute the three VLB components: KL prior, diffusion KL, reconstruction."""
        return self._eval_step(batch)

    def on_validation_epoch_start(self) -> None:
        """Clear VLB accumulators at the start of each validation epoch.

        Guards against stale tensors mixing with fresh data if the previous
        epoch's ``_log_vlb_and_mmd`` raised (e.g. OOM during MMD).
        """
        self._val_nll.clear()
        self._val_kl_prior.clear()
        self._val_kl_diffusion.clear()
        self._val_reconstruction.clear()
        self.mmd_evaluator.clear()

    def on_validation_epoch_end(self) -> None:
        """Log epoch-mean VLB components and compute MMD metrics."""
        self._log_vlb_and_mmd("val")

    # ------------------------------------------------------------------
    # Test (same evaluation as validation, logged under test/ prefix)
    # ------------------------------------------------------------------

    def test_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        """Compute VLB components and accumulate reference graphs, same as validation."""
        return self._eval_step(batch)

    def on_test_epoch_start(self) -> None:
        """Clear VLB accumulators at the start of each test epoch.

        Guards against stale tensors mixing with fresh data if the previous
        epoch's ``_log_vlb_and_mmd`` raised (e.g. OOM during MMD).
        """
        self._val_nll.clear()
        self._val_kl_prior.clear()
        self._val_kl_diffusion.clear()
        self._val_reconstruction.clear()
        self.mmd_evaluator.clear()

    def on_test_epoch_end(self) -> None:
        """Log epoch-mean VLB components and compute MMD metrics for test set."""
        self._log_vlb_and_mmd("test")

    # ------------------------------------------------------------------
    # Shared eval logic for validation and test
    # ------------------------------------------------------------------

    def _eval_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> dict[str, Tensor]:
        """Shared implementation for validation_step and test_step."""
        X, E, y, node_mask = batch

        noisy_data = self.apply_noise(X, E, y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        kl_prior = self._kl_prior(X, E, node_mask)
        loss_all_t = self._compute_Lt(X, E, y, pred, noisy_data, node_mask)

        prob0 = self._reconstruction_logp(noisy_data["t"], X, E, node_mask)
        loss_term_0 = self._reconstruction_loss(X, E, prob0, node_mask)

        nll = kl_prior + loss_all_t - loss_term_0

        self._val_nll.append(nll.mean())
        self._val_kl_prior.append(kl_prior.mean())
        self._val_kl_diffusion.append(loss_all_t.mean())
        self._val_reconstruction.append(loss_term_0.mean())

        # Accumulate reference graphs for MMD evaluation
        from tmgg.experiment_utils.mmd_metrics import adjacency_to_networkx

        self.mmd_evaluator.set_num_nodes(int(node_mask[0].sum().item()))
        adj_batch = E.argmax(dim=-1)
        for i in range(adj_batch.size(0)):
            n = int(node_mask[i].sum().item())
            adj_np = (adj_batch[i, :n, :n] > 0).cpu().numpy().astype("float32")
            self.mmd_evaluator.accumulate(adjacency_to_networkx(adj_np))

        return {"val_nll": nll.mean()}

    def _log_vlb_and_mmd(self, prefix: str) -> None:
        """Log VLB components and MMD metrics under the given prefix."""
        if not self._val_nll:
            return

        self.log(f"{prefix}/epoch_NLL", torch.stack(self._val_nll).mean())
        self.log(f"{prefix}/kl_prior", torch.stack(self._val_kl_prior).mean())
        self.log(f"{prefix}/kl_diffusion", torch.stack(self._val_kl_diffusion).mean())
        self.log(
            f"{prefix}/reconstruction_logp",
            torch.stack(self._val_reconstruction).mean(),
        )

        self._val_nll.clear()
        self._val_kl_prior.clear()
        self._val_kl_diffusion.clear()
        self._val_reconstruction.clear()

        # MMD evaluation via composed MMDEvaluator
        num_nodes = self.mmd_evaluator.num_nodes
        if self.mmd_evaluator.num_ref_graphs >= 2 and num_nodes is not None:
            num_to_generate = min(
                self.mmd_evaluator.num_ref_graphs, self.eval_num_samples
            )
            gen_graphs = self._sample_graphs_for_eval(num_to_generate, num_nodes)
            results = self.mmd_evaluator.evaluate(gen_graphs)
            if results is not None:
                self.log(f"{prefix}/degree_mmd", results.degree_mmd, prog_bar=True)
                self.log(f"{prefix}/clustering_mmd", results.clustering_mmd)
                self.log(f"{prefix}/spectral_mmd", results.spectral_mmd)
        else:
            self.mmd_evaluator.clear()

    def _sample_graphs_for_eval(
        self,
        num_graphs: int,
        num_nodes: int,
    ) -> list[nx.Graph[Any]]:
        """Generate graphs via ancestral categorical sampling and convert to NetworkX."""
        samples = self.sample_batch(batch_size=num_graphs, num_nodes=num_nodes)
        return categorical_samples_to_graphs(samples)

    # ------------------------------------------------------------------
    # VLB helper methods
    # ------------------------------------------------------------------

    def _kl_prior(self, X: Tensor, E: Tensor, node_mask: Tensor) -> Tensor:
        """KL between q(z_T | x) and the limit distribution p(z_T).

        Diffuses to the final step T and measures how close the resulting
        distribution is to the prior. Near-zero for well-tuned schedules,
        but a useful sanity signal.

        Returns shape ``(bs,)``.
        """
        bs = X.size(0)
        ones = torch.ones((bs, 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        probX = X @ Qtb.X  # (bs, n, dx)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de)
        if probX.shape != X.shape:
            raise RuntimeError(f"probX shape {probX.shape} != X shape {X.shape}")

        bs, n, _ = probX.shape
        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = (
            self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        )

        limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
        )

        # NOTE: F.kl_div(input=log_q, target=p) computes KL(p || q), so this
        # is KL(limit_dist || q(z_T|x_0)) -- the reverse of the standard VLB
        # direction KL(q(z_T|x_0) || p(z_T)).  We keep this to match the
        # original DiGress implementation (Vignac et al. 2023).  In practice
        # the term is near-zero for well-tuned schedules, so the direction
        # makes negligible difference.
        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction="none"
        )

        return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E)

    def _compute_Lt(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        pred: PlaceHolder,
        noisy_data: dict[str, Tensor],
        node_mask: Tensor,
    ) -> Tensor:
        """Diffusion KL loss term L_t, scaled by T.

        Computes KL between the true posterior q(z_{t-1} | z_t, x_0) and the
        predicted posterior q(z_{t-1} | z_t, hat{x}_0), where hat{x}_0 comes
        from the neural network prediction.

        Returns shape ``(bs,)``.
        """
        # pred already contains softmaxed probabilities
        pred_probs_X = pred.X
        pred_probs_E = pred.E
        pred_probs_y = F.softmax(pred.y, dim=-1) if pred.y.size(-1) > 0 else pred.y

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

        bs, n, d = X.shape

        # True posterior: q(z_{t-1} | z_t, x_0)
        prob_true = posterior_distributions(
            X=X,
            E=E,
            y=y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_true_E = prob_true.E.reshape((bs, n, n, -1))

        # Predicted posterior: q(z_{t-1} | z_t, hat{x}_0)
        prob_pred = posterior_distributions(
            X=pred_probs_X,
            E=pred_probs_E,
            y=pred_probs_y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_pred_E = prob_pred.E.reshape((bs, n, n, -1))

        # Mask distributions to exclude invalid positions
        true_X, true_E, pred_X, pred_E = mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true_E,
            pred_X=prob_pred.X,
            pred_E=prob_pred_E,
            node_mask=node_mask,
        )

        kl_x = F.kl_div(input=torch.log(pred_X), target=true_X, reduction="none")
        kl_e = F.kl_div(input=torch.log(pred_E), target=true_E, reduction="none")

        return self.T * (sum_except_batch(kl_x) + sum_except_batch(kl_e))

    def _reconstruction_logp(
        self, t: Tensor, X: Tensor, E: Tensor, node_mask: Tensor
    ) -> PlaceHolder:
        """Predict clean-data distribution at t=0 for the reconstruction term.

        Applies minimal noise (one step from clean), runs the denoiser, and
        returns predicted probabilities with masked positions set to uniform
        ones so they contribute zero log-probability.
        """
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X
        probE0 = E @ Q0.E.unsqueeze(1)

        sampled0 = sample_discrete_features(
            probX=probX0, probE=probE0, node_mask=node_mask
        )

        X0 = F.one_hot(sampled0.X, num_classes=self.dx).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime
        E0 = F.one_hot(sampled0.E, num_classes=self.de).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime
        y0 = sampled0.y
        if X.shape != X0.shape or E.shape != E0.shape:
            raise RuntimeError(
                f"Shape mismatch after one_hot: X {X.shape} vs X0 {X0.shape}, "
                f"E {E.shape} vs E0 {E0.shape}"
            )

        sampled_0 = PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1).type_as(y0),
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # pred0.X/E are already softmaxed. Set masked positions to 1 so
        # log(1)=0 and they contribute nothing to the reconstruction loss.
        probX0_out = pred0.X.clone()
        probE0_out = pred0.E.clone()

        probX0_out[~node_mask] = torch.ones(self.dx, device=X.device, dtype=X.dtype)
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0_out[~edge_mask] = torch.ones(self.de, device=X.device, dtype=X.dtype)

        diag_mask = (
            torch.eye(probE0_out.size(1), device=X.device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(probE0_out.size(0), -1, -1)
        )
        probE0_out[diag_mask] = torch.ones(self.de, device=X.device, dtype=X.dtype)

        return PlaceHolder(X=probX0_out, E=probE0_out, y=pred0.y)

    @staticmethod
    def _reconstruction_loss(
        X: Tensor, E: Tensor, prob0: PlaceHolder, node_mask: Tensor
    ) -> Tensor:
        """Compute reconstruction log-probability log p(x | z_0).

        Returns shape ``(bs,)``.
        """
        loss_X = sum_except_batch(X * prob0.X.log())
        loss_E = sum_except_batch(E * prob0.E.log())
        return loss_X + loss_E

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        num_nodes: int | Tensor | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Generate graphs via ancestral categorical reverse diffusion.

        Parameters
        ----------
        batch_size
            Number of graphs to generate.
        num_nodes
            Fixed node count (int), per-graph counts (Tensor), or ``None``
            to use the datamodule's ``num_nodes``.

        Returns
        -------
        list[tuple[Tensor, Tensor]]
            ``(node_types, edge_types)`` integer tensors trimmed to the
            real node count per graph.
        """
        if num_nodes is None:
            dm = getattr(self.trainer, "datamodule", None) if self.trainer else None
            if dm is not None and hasattr(dm, "num_nodes"):
                num_nodes = dm.num_nodes  # pyright: ignore[reportAttributeAccessIssue]  # DataModule protocol
            else:
                raise ValueError(
                    "num_nodes must be provided when no datamodule is attached"
                )

        if isinstance(num_nodes, int):
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.long
            )
        else:
            assert isinstance(num_nodes, Tensor)
            n_nodes = num_nodes.to(self.device)

        n_max = int(n_nodes.max().item())

        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample from the limit distribution
        z_T = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        if not (torch.transpose(E, 1, 2) == E).all():
            raise RuntimeError("Initial noise E not symmetric")

        # Reverse diffusion: iteratively sample p(z_s | z_t)
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1), device=self.device).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s = self._sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Collapse one-hot to class indices and trim to real nodes
        final = PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
        X_final, E_final = final.X, final.E

        results: list[tuple[Tensor, Tensor]] = []
        for i in range(batch_size):
            n = int(n_nodes[i].item())
            results.append((X_final[i, :n].cpu(), E_final[i, :n, :n].cpu()))

        return results

    def _sample_p_zs_given_zt(
        self,
        s: Tensor,
        t: Tensor,
        X_t: Tensor,
        E_t: Tensor,
        y_t: Tensor,
        node_mask: Tensor,
    ) -> PlaceHolder:
        """One reverse diffusion step: sample z_s ~ p(z_s | z_t).

        Computes the posterior p(z_s | z_t) = sum_{x_0} p(x_0|z_t) *
        q(z_s | z_t, x_0), where p(x_0|z_t) is the neural network's
        prediction and q(z_s | z_t, x_0) is the analytic transition
        posterior.
        """
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Predict p(x_0 | z_t) via neural net
        noisy_data: dict[str, Tensor] = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # pred.X, pred.E are already softmaxed
        pred_X = pred.X  # (bs, n, d0)
        pred_E = pred.E  # (bs, n, n, d0)

        # Compute p(z_s, z_t | x_0) for each possible x_0 value
        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
        )

        # Weight by predicted p(x_0) and marginalise over x_0
        # X: (bs, n, d0, d_{t-1})
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)  # (bs, n, d_{t-1})
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )

        pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        if not ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all():
            raise RuntimeError(
                f"prob_X not normalised: max dev "
                f"{(prob_X.sum(dim=-1) - 1).abs().max():.2e}"
            )
        if not ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all():
            raise RuntimeError(
                f"prob_E not normalised: max dev "
                f"{(prob_E.sum(dim=-1) - 1).abs().max():.2e}"
            )

        # Sample from the reverse posterior
        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.dx).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime
        E_s = F.one_hot(sampled_s.E, num_classes=self.de).float()  # pyright: ignore[reportAttributeAccessIssue]  # F.one_hot exists at runtime

        if not (E_s == torch.transpose(E_s, 1, 2)).all():
            raise RuntimeError("Sampled E_s not symmetric")
        if X_t.shape != X_s.shape or E_t.shape != E_s.shape:
            raise RuntimeError(
                f"Shape mismatch after sampling: X_t {X_t.shape} vs X_s {X_s.shape}, "
                f"E_t {E_t.shape} vs E_s {E_s.shape}"
            )

        out = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0).type_as(y_t))
        return out.mask(node_mask).type_as(y_t)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Log model parameter count at training start."""
        log_parameter_count(self.model, self.get_model_name(), self.logger)

    def get_model_name(self) -> str:
        """Return model name for logging."""
        return "DiscreteDiffusion"
