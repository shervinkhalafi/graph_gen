"""Shared pytest fixtures for integration tests.

These fixtures provide common setup for testing experiment runners and
config composition without running full training.
"""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def quick_training_overrides(tmp_path: Path) -> list[str]:
    """Hydra overrides for minimal training runs.

    Configures experiments to complete in ~10 seconds by limiting steps,
    disabling logging, and using minimal data on CPU.
    """
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        "++data.num_samples_per_graph=4",
        "++data.num_train_samples=8",
        "++data.num_val_samples=4",
        "++data.num_test_samples=4",
        "++data.dataset_config.num_graphs=4",
        f"hydra.run.dir={tmp_path}",
    ]


@pytest.fixture
def sample_adjacency_batch() -> torch.Tensor:
    """Generate a small batch of adjacency matrices for forward pass tests.

    Creates 4 random symmetric adjacency matrices (20x20) suitable for
    testing model forward passes.
    """
    batch_size = 4
    n_nodes = 20

    matrices = []
    for _ in range(batch_size):
        # Random symmetric matrix with ~30% edge density
        A = torch.bernoulli(torch.full((n_nodes, n_nodes), 0.3))
        A = (A + A.T) / 2
        A.fill_diagonal_(0)
        A = (A > 0.5).float()
        matrices.append(A)

    return torch.stack(matrices)


@pytest.fixture
def config_path() -> Path:
    """Path to the experiment configs directory."""
    return Path(__file__).parent.parent / "src" / "tmgg" / "exp_configs"
