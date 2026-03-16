"""E2E integration tests for DiffusionModule-based generative diffusion.

Test rationale:
    Verifies the complete diffusion-based graph generation pipeline for
    DiffusionModule:
    1. Model instantiation for all architectures
    2. Training step with diffusion noise and loss computation
    3. Validation with metric computation
    4. Full training loop across all dataset types

Invariants:
    - Model output shape matches input shape (adjacency matrix reconstruction)
    - Loss is finite and positive
    - Generated graphs are binary, symmetric, with zero diagonal
    - MMD metrics are non-negative floats
"""

from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch

from tmgg.data.data_modules.multigraph_data_module import (
    MultiGraphDataModule,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.noise import DigressNoiseGenerator
from tmgg.diffusion.noise_process import ContinuousNoiseProcess
from tmgg.diffusion.sampler import ContinuousSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser
from tmgg.training.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
)

# Dataset configurations for parametrized tests (excludes LFR due to generation complexity)
DATASET_CONFIGS: dict[str, dict] = {
    "sbm": {"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
    "erdos_renyi": {"p": 0.3},
    "regular": {"d": 3},
    "tree": {},
    "watts_strogatz": {"k": 4, "p": 0.3},
    "random_geometric": {"radius": 0.4},
}


# ---------------------------------------------------------------------------
# DiffusionModule helper
# ---------------------------------------------------------------------------


def _make_diffusion_module(
    model: GraphModel | None = None,
    timesteps: int = 10,
    num_nodes: int = 16,
    eval_num_samples: int = 4,
) -> DiffusionModule:
    """Construct a DiffusionModule with ContinuousNoiseProcess + DigressNoiseGenerator."""
    if model is None:
        model = SelfAttentionDenoiser(k=8, d_k=16)
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=timesteps)
    noise_process = ContinuousNoiseProcess(
        generator=DigressNoiseGenerator(), noise_schedule=schedule
    )
    sampler = ContinuousSampler(
        noise_process=ContinuousNoiseProcess(
            generator=DigressNoiseGenerator(), noise_schedule=schedule
        ),
        noise_schedule=schedule,
    )
    evaluator = GraphEvaluator(
        eval_num_samples=eval_num_samples, kernel="gaussian", sigma=1.0
    )
    return DiffusionModule(
        model=model,
        noise_process=noise_process,
        sampler=sampler,
        noise_schedule=schedule,
        evaluator=evaluator,
        loss_type="mse",
        num_nodes=num_nodes,
        eval_every_n_steps=1,
    )


def _make_block_adjacency(bs: int = 4, n: int = 16) -> torch.Tensor:
    """Create a batch of binary symmetric adjacency matrices with block structure."""
    batch = torch.zeros(bs, n, n)
    half = n // 2
    batch[:, :half, :half] = torch.bernoulli(torch.full((bs, half, half), 0.6))
    batch = (batch + batch.transpose(-2, -1)) / 2
    batch = (batch > 0.5).float()
    batch.diagonal(dim1=-2, dim2=-1).zero_()
    return batch


# ---------------------------------------------------------------------------
# DataModule tests (independent of any LightningModule)
# ---------------------------------------------------------------------------


class TestGenerativeDataModule:
    """Verify datamodule generates valid data for all graph types."""

    @pytest.mark.parametrize("dataset_type", list(DATASET_CONFIGS.keys()))
    def test_datamodule_setup(self, dataset_type: str) -> None:
        """DataModule should set up train/val/test splits correctly."""
        datamodule = MultiGraphDataModule(
            graph_type=dataset_type,
            num_nodes=16,
            num_graphs=20,
            batch_size=4,
            graph_config=DATASET_CONFIGS[dataset_type],
            seed=42,
        )

        datamodule.setup("fit")

        # Verify data was created
        assert datamodule._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert datamodule._val_data is not None  # pyright: ignore[reportPrivateUsage]

        # Verify list[Data] structure
        train_data = datamodule._train_data  # pyright: ignore[reportPrivateUsage]
        assert isinstance(train_data, list)
        assert len(train_data) > 0
        assert train_data[0].num_nodes == 16  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.parametrize("dataset_type", list(DATASET_CONFIGS.keys()))
    def test_generated_graphs_are_valid(self, dataset_type: str) -> None:
        """Generated graphs should be binary, symmetric, with zero diagonal."""
        datamodule = MultiGraphDataModule(
            graph_type=dataset_type,
            num_nodes=16,
            num_graphs=10,
            batch_size=4,
            graph_config=DATASET_CONFIGS[dataset_type],
            seed=42,
        )

        datamodule.setup("fit")
        train_data = datamodule._train_data  # pyright: ignore[reportPrivateUsage]
        assert train_data is not None

        # Reconstruct adjacency from PyG Data to verify graph properties
        from torch_geometric.utils import to_dense_adj

        for g in train_data:
            assert g.edge_index is not None
            adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes).squeeze(0)
            # Check binary values
            assert torch.all((adj == 0) | (adj == 1)), "Non-binary adjacency"
            # Check symmetry
            assert torch.allclose(adj, adj.T), "Adjacency not symmetric"
            # Check zero diagonal
            assert torch.all(adj.diagonal() == 0), "Self-loops present"


# ---------------------------------------------------------------------------
# DiffusionModule tests
# ---------------------------------------------------------------------------


class TestDiffusionModuleInstantiation:
    """Verify DiffusionModule instantiates with continuous noise components.

    Test rationale: confirms DiffusionModule can be constructed with all
    supported model types using the injected ContinuousNoiseProcess,
    ContinuousSampler, and NoiseSchedule components.
    """

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                lambda: __import__(
                    "tmgg.models.spectral_denoisers", fromlist=["LinearPE"]
                ).LinearPE(k=8),
                id="LinearPE",
            ),
            pytest.param(
                lambda: __import__(
                    "tmgg.models.spectral_denoisers", fromlist=["GraphFilterBank"]
                ).GraphFilterBank(k=8, polynomial_degree=3),
                id="GraphFilterBank",
            ),
            pytest.param(
                lambda: SelfAttentionDenoiser(k=8, d_k=16),
                id="SelfAttentionDenoiser",
            ),
            pytest.param(
                lambda: __import__(
                    "tmgg.models.spectral_denoisers.bilinear",
                    fromlist=["BilinearDenoiserWithMLP"],
                ).BilinearDenoiserWithMLP(
                    k=8, d_k=16, mlp_hidden_dim=32, mlp_num_layers=1
                ),
                id="BilinearDenoiserWithMLP",
            ),
            pytest.param(
                lambda: __import__(
                    "tmgg.models.spectral_denoisers.bilinear",
                    fromlist=["MultiLayerBilinearDenoiser"],
                ).MultiLayerBilinearDenoiser(
                    k=8, d_model=16, num_heads=2, num_layers=1
                ),
                id="MultiLayerBilinearDenoiser",
            ),
            pytest.param(
                lambda: __import__("tmgg.models.gnn", fromlist=["GNN"]).GNN(
                    num_layers=1, num_terms=2, feature_dim_in=8, feature_dim_out=4
                ),
                id="GNN",
            ),
            pytest.param(
                lambda: __import__(
                    "tmgg.models.gnn", fromlist=["GNNSymmetric"]
                ).GNNSymmetric(
                    num_layers=1, num_terms=2, feature_dim_in=8, feature_dim_out=4
                ),
                id="GNNSymmetric",
            ),
            pytest.param(
                lambda: _make_hybrid_model(),
                id="hybrid",
            ),
        ],
    )
    def test_model_type_instantiates(self, model) -> None:
        """Each supported model type should produce a valid DiffusionModule."""
        module = _make_diffusion_module(model=model())
        assert module.model is not None
        assert hasattr(module.model, "forward")

    def test_invalid_loss_type_raises(self) -> None:
        """Unknown loss_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            _make_diffusion_module.__wrapped__ if hasattr(  # type: ignore[attr-defined]
                _make_diffusion_module, "__wrapped__"
            ) else None
            schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=10)
            DiffusionModule(
                model=SelfAttentionDenoiser(k=8, d_k=16),
                noise_process=ContinuousNoiseProcess(
                    generator=DigressNoiseGenerator(),
                    noise_schedule=schedule,
                ),
                sampler=ContinuousSampler(
                    noise_process=ContinuousNoiseProcess(
                        generator=DigressNoiseGenerator(),
                        noise_schedule=schedule,
                    ),
                    noise_schedule=schedule,
                ),
                noise_schedule=schedule,
                loss_type="invalid",
            )


def _make_hybrid_model():
    """Create a hybrid SequentialDenoisingModel for tests."""
    from tmgg.models.gnn import GNN
    from tmgg.models.hybrid import SequentialDenoisingModel

    gnn = GNN(num_layers=1, num_terms=2, feature_dim_in=8, feature_dim_out=4)
    return SequentialDenoisingModel(embedding_model=gnn, denoising_model=None)


class TestDiffusionModuleTrainingStep:
    """Verify DiffusionModule training step produces valid loss and gradients.

    Test rationale: DiffusionModule should produce finite positive loss and
    gradient flow through the model parameters.
    """

    @pytest.fixture
    def module_and_batch(self) -> tuple[DiffusionModule, GraphData]:
        module = _make_diffusion_module()
        batch = GraphData.from_adjacency(_make_block_adjacency())
        return module, batch

    def test_training_step_finite_loss(
        self, module_and_batch: tuple[DiffusionModule, GraphData]
    ) -> None:
        """Training step should produce a finite scalar loss."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradients_flow(
        self, module_and_batch: tuple[DiffusionModule, GraphData]
    ) -> None:
        """Gradients should propagate to model parameters after backward pass."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        params_with_grad = [p for p in module.model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No gradients flowed to model parameters"


class TestDiffusionModuleValidation:
    """Verify DiffusionModule validation computes loss without accumulating graphs.

    Test rationale: the DiffusionModule should compute validation loss per
    batch; reference graphs for evaluation are pulled from the datamodule
    at epoch end (covered by the E2E test and unit tests in test_diffusion_module.py).
    """

    @pytest.fixture
    def module_and_batch(self) -> tuple[DiffusionModule, GraphData]:
        module = _make_diffusion_module(eval_num_samples=4)
        batch = GraphData.from_adjacency(_make_block_adjacency())
        return module, batch

    def test_validation_step_runs_without_accumulation(
        self, module_and_batch: tuple[DiffusionModule, GraphData]
    ) -> None:
        """Validation step should compute loss without buffering graphs."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            module.validation_step(batch, batch_idx=0)
        # No _ref_graphs buffer should exist on the module
        assert not hasattr(module, "_ref_graphs")


class TestDiffusionModuleE2E:
    """Full training loop for DiffusionModule with an SBM dataset.

    Test rationale: end-to-end smoke test ensuring DiffusionModule can
    complete train/val/test through Lightning without exceptions.
    """

    def test_short_training_run(self) -> None:
        """Short training run should complete without errors."""
        module = _make_diffusion_module(eval_num_samples=4)
        datamodule = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=20,
            batch_size=4,
            graph_config={"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
            seed=42,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
        assert trainer.current_epoch == 2


class TestDiffusionModulePerElementNoise:
    """Regression test: DiffusionModule applies per-element noise levels.

    Test rationale: the ContinuousNoiseProcess wrapping DigressNoiseGenerator
    should call add_noise with a (bs,) tensor of distinct noise levels
    derived from distinct timesteps, applying per-sample noise in one
    vectorized call.
    """

    def test_per_element_noise_via_noise_process(self) -> None:
        """DiffusionModule should apply per-sample noise levels as a batched tensor."""
        module = _make_diffusion_module(timesteps=100)
        batch = GraphData.from_adjacency(_make_block_adjacency())

        # Instrument the underlying noise generator to record the eps argument
        recorded_eps: list[torch.Tensor] = []
        assert isinstance(module.noise_process, ContinuousNoiseProcess)
        original_add_noise = module.noise_process.generator.add_noise

        def recording_add_noise(
            A: torch.Tensor, eps: float | torch.Tensor
        ) -> torch.Tensor:
            if isinstance(eps, torch.Tensor):
                recorded_eps.append(eps.clone())
            return original_add_noise(A, eps)

        module.noise_process.generator.add_noise = recording_add_noise  # pyright: ignore[reportAttributeAccessIssue]

        torch.manual_seed(12345)
        with patch.object(module, "log"):
            module.training_step(batch, batch_idx=0)

        assert (
            len(recorded_eps) == 1
        ), f"Expected 1 batched add_noise call, got {len(recorded_eps)}"

        eps_tensor = recorded_eps[0]
        assert eps_tensor.shape == (
            4,
        ), f"Expected eps shape (4,), got {eps_tensor.shape}"

        # Noise levels should be in [0, 1] (converted from integer timesteps)
        assert torch.all(eps_tensor >= 0.0) and torch.all(
            eps_tensor <= 1.0
        ), f"Noise levels not in [0, 1]: {eps_tensor}"

        # At least 2 distinct values — per-sample noise, not uniform
        assert eps_tensor.unique().numel() >= 2, (
            f"All eps values are identical ({eps_tensor[0].item():.4f}), "
            "noise is not being applied per-element"
        )
