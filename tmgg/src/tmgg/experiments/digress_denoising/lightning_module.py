"""PyTorch Lightning module for Digress GraphTransformer-based denoising."""

from typing import Any, override

import torch

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.experiment_utils.mmd_evaluator import MMDEvaluator
from tmgg.experiment_utils.mmd_metrics import adjacency_to_networkx
from tmgg.experiment_utils.sampling import get_noise_schedule
from tmgg.models.base import DenoisingModel
from tmgg.models.digress.transformer_model import GraphTransformer


class DigressDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for Digress GraphTransformer-based graph denoising.

    Eigenvector extraction (when use_eigenvectors=True) is handled internally
    by the GraphTransformer model, matching the architecture of spectral denoisers.

    Includes MMD evaluation during validation: accumulates clean reference
    graphs, generates graphs via iterative denoising at epoch end, and logs
    degree/clustering/spectral MMD for cross-paradigm comparison with the
    generative discrete diffusion experiment.
    """

    def __init__(
        self,
        use_eigenvectors: bool = False,
        k: int = 20,
        n_layers: int = 4,
        hidden_mlp_dims: dict[str, int] | None = None,
        hidden_dims: dict[str, int] | None = None,
        output_dims: dict[str, int] | None = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: dict[str, Any] | None = None,
        eval_noise_levels: list[float] | None = None,
        noise_type: str = "digress",
        rotation_k: int = 20,
        seed: int | None = None,
        # MMD evaluation
        eval_num_samples: int = 128,
        mmd_kernel: str = "gaussian_tv",
        mmd_sigma: float = 1.0,
        # Legacy parameter name (deprecated, use k instead)
        node_feature_dim: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ):
        # Handle legacy parameter name
        if node_feature_dim is not None:
            k = node_feature_dim

        # Populate self.hparams before super().__init__() so that subclass
        # params are available during _make_model if needed.
        # See the Template Method contract on _make_model.
        self.save_hyperparameters()

        super().__init__(
            use_eigenvectors=use_eigenvectors,
            k=k,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            eval_noise_levels=eval_noise_levels,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
            **kwargs,
        )

        # MMD evaluation — same interface as DiscreteDiffusionLightningModule
        self.mmd_evaluator = MMDEvaluator(
            eval_num_samples=eval_num_samples,
            kernel=mmd_kernel,
            sigma=mmd_sigma,
        )
        self._mmd_num_nodes: int | None = None

    # forward() is inherited from base class - model handles eigenvector extraction

    # ------------------------------------------------------------------
    # MMD evaluation hooks
    # ------------------------------------------------------------------

    @override
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Run base validation and accumulate reference graphs for MMD."""
        result = super().validation_step(batch, batch_idx)

        # Accumulate clean graphs as NetworkX for MMD reference set.
        # batch shape: (B, N, N) — binary adjacency matrices.
        if self._mmd_num_nodes is None:
            self._mmd_num_nodes = batch.shape[1]
            self.mmd_evaluator.set_num_nodes(batch.shape[1])

        for i in range(batch.size(0)):
            adj_np = batch[i].detach().cpu().numpy()
            self.mmd_evaluator.accumulate(adjacency_to_networkx(adj_np))

        return result

    @override
    def on_validation_epoch_end(self) -> None:
        """Generate graphs via iterative denoising and compute MMD metrics."""
        # Run MMD evaluation before base visualizations
        if (
            self.global_step > 0
            and self.mmd_evaluator.num_ref_graphs >= 2
            and self._mmd_num_nodes is not None
        ):
            num_to_generate = min(
                self.mmd_evaluator.num_ref_graphs,
                self.mmd_evaluator.eval_num_samples,
            )
            gen_adjs = self.sample(
                num_graphs=num_to_generate,
                num_nodes=self._mmd_num_nodes,
            )
            gen_graphs = [adjacency_to_networkx(adj.cpu().numpy()) for adj in gen_adjs]
            results = self.mmd_evaluator.evaluate(gen_graphs)
            if results is not None:
                self.log("val/degree_mmd", results.degree_mmd, prog_bar=True)
                self.log("val/clustering_mmd", results.clustering_mmd)
                self.log("val/spectral_mmd", results.spectral_mmd)
        else:
            self.mmd_evaluator.clear()

        self._mmd_num_nodes = None
        super().on_validation_epoch_end()

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        """Run base test step and accumulate reference graphs for MMD."""
        result = super().test_step(batch, batch_idx)

        if self._mmd_num_nodes is None:
            self._mmd_num_nodes = batch.shape[1]
            self.mmd_evaluator.set_num_nodes(batch.shape[1])

        for i in range(batch.size(0)):
            adj_np = batch[i].detach().cpu().numpy()
            self.mmd_evaluator.accumulate(adjacency_to_networkx(adj_np))

        return result

    @override
    def on_test_epoch_end(self) -> None:
        """Generate graphs and compute MMD metrics at end of test."""
        if self.mmd_evaluator.num_ref_graphs >= 2 and self._mmd_num_nodes is not None:
            num_to_generate = min(
                self.mmd_evaluator.num_ref_graphs,
                self.mmd_evaluator.eval_num_samples,
            )
            gen_adjs = self.sample(
                num_graphs=num_to_generate,
                num_nodes=self._mmd_num_nodes,
            )
            gen_graphs = [adjacency_to_networkx(adj.cpu().numpy()) for adj in gen_adjs]
            results = self.mmd_evaluator.evaluate(gen_graphs)
            if results is not None:
                self.log("test/degree_mmd", results.degree_mmd)
                self.log("test/clustering_mmd", results.clustering_mmd)
                self.log("test/spectral_mmd", results.spectral_mmd)
        else:
            self.mmd_evaluator.clear()

        self._mmd_num_nodes = None
        super().on_test_epoch_end()

    def _make_model(
        self,
        *args: Any,
        n_layers: int = 4,
        hidden_mlp_dims: dict[str, int] | None = None,
        hidden_dims: dict[str, int] | None = None,
        output_dims: dict[str, int] | None = None,
        k: int = 20,
        use_eigenvectors: bool = False,
        **kwargs: Any,
    ) -> DenoisingModel:
        """Create the DiGress GraphTransformer model.

        Parameters
        ----------
        n_layers
            Number of transformer layers.
        hidden_mlp_dims
            Hidden MLP dimensions. Defaults to {"X": 256, "E": 64, "y": 256}.
        hidden_dims
            Transformer dimensions. Defaults to {"dx": 128, "de": 32, "dy": 128, "n_head": 4}.
        output_dims
            Output dimensions. Defaults to {"X": 0, "E": 1, "y": 0}.
        k
            Number of eigenvectors to extract when use_eigenvectors=True.
        use_eigenvectors
            If True, model extracts eigenvectors from adjacency internally.
            If False, model receives adjacency matrix directly.
        """
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"X": 256, "E": 64, "y": 256}
        if hidden_dims is None:
            hidden_dims = {"dx": 128, "de": 32, "dy": 128, "n_head": 4}
        if output_dims is None:
            output_dims = {"X": 0, "E": 1, "y": 0}
        input_dims = {
            "X": k if use_eigenvectors else 1,
            "E": 1,
            "y": 0,
        }

        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            use_eigenvectors=use_eigenvectors,
            k=k if use_eigenvectors else None,
        )
        assert isinstance(model, DenoisingModel)
        return model

    def get_model_name(self) -> str:
        return "Digress"

    @torch.no_grad()
    def sample(
        self,
        num_graphs: int,
        num_nodes: int,
        num_steps: int = 100,
        noise_schedule: str = "cosine",
    ) -> list[torch.Tensor]:
        """Generate graphs by iterative denoising from noise.

        Implements discrete denoising diffusion sampling: starts from random
        binary graphs and iteratively denoises using the learned model.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.
        num_nodes
            Number of nodes per graph.
        num_steps
            Number of denoising steps.
        noise_schedule
            Noise schedule type: "linear", "cosine", or "quadratic".

        Returns
        -------
        list[torch.Tensor]
            List of generated adjacency matrices, each (num_nodes, num_nodes).
        """
        self.eval()
        device = next(self.parameters()).device

        schedule = get_noise_schedule(noise_schedule, num_steps)

        # Start from random binary graphs (50% edge probability)
        z_t = (
            torch.rand(num_graphs, num_nodes, num_nodes, device=device) > 0.5
        ).float()
        # Make symmetric
        z_t = (z_t + z_t.transpose(-2, -1)) / 2
        z_t = (z_t > 0.5).float()
        # Zero diagonal (no self-loops)
        z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Iterative denoising
        for step in reversed(range(num_steps)):
            eps = schedule[step]

            # Forward pass to get model prediction
            output = self.forward(z_t)
            predictions = self.model.predict(output)

            # Interpolate between prediction and current state based on noise level
            # High noise (step near num_steps) → trust model less
            # Low noise (step near 0) → trust model more
            alpha = 1.0 - eps
            z_t = alpha * predictions + (1 - alpha) * z_t

            # Binarize for discrete graph
            z_t = (z_t > 0.5).float()
            # Enforce symmetry
            z_t = (z_t + z_t.transpose(-2, -1)) / 2
            z_t = (z_t > 0.5).float()
            # Enforce zero diagonal
            z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Final prediction
        output = self.forward(z_t)
        final = self.model.logits_to_graph(output)
        # Enforce symmetry (DiGress-style averaging)
        final = (final + final.transpose(-2, -1)) / 2
        final = (final > 0.5).float()
        # Enforce zero diagonal
        final = final * (1 - torch.eye(num_nodes, device=device))

        return [final[i] for i in range(num_graphs)]
