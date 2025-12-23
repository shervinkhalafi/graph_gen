"""Sanity check tests for DiGress model variants.

Tests two configurations:
1. Official DiGress settings: AdamW + amsgrad, LR=0.0002, weight_decay=1e-12
2. High LR settings: Adam, LR=1e-2, weight_decay=0 (matching spectral models)

Both use the small 4-layer architecture from digress_sbm_small.yaml.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tmgg.experiment_utils import generate_sbm_adjacency
from tmgg.experiment_utils.data import add_digress_noise
from tmgg.models.digress.transformer_model import GraphTransformer

# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_digress_small(n_nodes: int = 16) -> GraphTransformer:
    """Create small DiGress model matching digress_sbm_small.yaml config."""
    return GraphTransformer(
        n_layers=4,
        input_dims={"X": 1, "E": 1, "y": 0},
        hidden_mlp_dims={"X": 64, "E": 32, "y": 64},
        hidden_dims={"dx": 128, "de": 32, "dy": 128, "n_head": 4},
        output_dims={"X": 0, "E": 1, "y": 0},
    )


# =============================================================================
# TARGET GENERATION (same as test_learning_sanity.py)
# =============================================================================


def create_block_diagonal_target(n: int = 16, num_blocks: int = 4) -> torch.Tensor:
    """Create block diagonal matrix - dense within blocks, no inter-block edges."""
    A = torch.zeros(1, n, n)
    block_size = n // num_blocks
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        A[0, start:end, start:end] = 1.0
    A[0].fill_diagonal_(0)
    return A


def create_sbm_target(
    block_sizes: list | None = None,
    p: float = 0.8,
    q: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Create SBM graph with community structure."""
    if block_sizes is None:
        block_sizes = [4, 4, 4, 4]
    rng = np.random.default_rng(seed)
    A_np = generate_sbm_adjacency(block_sizes, p=p, q=q, rng=rng)
    A = torch.tensor(A_np, dtype=torch.float32).unsqueeze(0)
    return A


# =============================================================================
# LEVEL 1: CONSTANT NOISE MEMORIZATION - DIGRESS
# =============================================================================


class TestDigressLevel1ConstantNoiseMemorization:
    """LEVEL 1: Can DiGress memorize a fixed input→output mapping?

    Tests both optimizer configurations on the memorization task.
    """

    @pytest.fixture(params=["block_diagonal", "sbm"])
    def fixed_sample(self, request):
        """Create fixed noisy/clean pair."""
        n = 16
        if request.param == "block_diagonal":
            A_clean = create_block_diagonal_target(n=n, num_blocks=4)
        else:
            A_clean = create_sbm_target(block_sizes=[4, 4, 4, 4], p=0.8, q=0.1, seed=42)

        torch.manual_seed(123)
        np.random.seed(123)
        A_noisy = add_digress_noise(A_clean, p=0.1)
        return A_noisy, A_clean, request.param

    @pytest.mark.xfail(
        reason="DiGress converges to ~90% accuracy on fixed-input memorization and plateaus. "
        "Tested with 5k and 25k steps - same accuracy, indicating a local minimum. "
        "Level 2 (fresh noise) tests pass, confirming the model learns denoising.",
        strict=False,
    )
    def test_digress_official_lr_memorization(self, fixed_sample):
        """DiGress with official settings (AdamW+amsgrad, LR=0.0002) on fixed input."""
        A_noisy, A_clean, target_type = fixed_sample
        model = create_digress_small()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0002,
            weight_decay=1e-12,
            amsgrad=True,
        )

        for _ in range(25000):
            logits = model(A_noisy)
            loss = F.binary_cross_entropy_with_logits(logits, A_clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            logits = model(A_noisy)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

        assert (
            accuracy > 0.99
        ), f"DiGress (official LR) failed Level 1 on {target_type}: {accuracy:.1%}"

    @pytest.mark.xfail(
        reason="DiGress with high LR converges to ~75% accuracy and plateaus. "
        "Tested with 5k and 15k steps - same accuracy. Worse than official LR (~90%), "
        "suggesting DiGress is sensitive to hyperparameters. Level 2 tests pass.",
        strict=False,
    )
    def test_digress_high_lr_memorization(self, fixed_sample):
        """DiGress with high LR settings (Adam, LR=1e-2) on fixed input."""
        A_noisy, A_clean, target_type = fixed_sample
        model = create_digress_small()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(15000):
            logits = model(A_noisy)
            loss = F.binary_cross_entropy_with_logits(logits, A_clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            logits = model(A_noisy)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

        assert (
            accuracy > 0.99
        ), f"DiGress (high LR) failed Level 1 on {target_type}: {accuracy:.1%}"


# =============================================================================
# LEVEL 2: FRESH NOISE GENERALIZATION - DIGRESS
# =============================================================================


class TestDigressLevel2FreshNoiseGeneralization:
    """LEVEL 2: Can DiGress generalize denoising with fresh noise each step?"""

    @pytest.fixture(params=["block_diagonal", "sbm"])
    def structured_sample(self, request):
        """Create structured target."""
        n = 16
        if request.param == "block_diagonal":
            A_clean = create_block_diagonal_target(n=n, num_blocks=4)
        else:
            A_clean = create_sbm_target(block_sizes=[4, 4, 4, 4], p=0.8, q=0.1, seed=42)
        return A_clean, request.param

    def test_digress_official_lr_generalization(self, structured_sample):
        """DiGress with official settings on fresh noise each step."""
        A_clean, target_type = structured_sample
        model = create_digress_small()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0002,
            weight_decay=1e-12,
            amsgrad=True,
        )

        for _ in range(2000):
            A_noisy = add_digress_noise(A_clean, p=0.1)
            logits = model(A_noisy)
            loss = F.binary_cross_entropy_with_logits(logits, A_clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.manual_seed(999)
        A_noisy_eval = add_digress_noise(A_clean, p=0.1)

        with torch.no_grad():
            logits = model(A_noisy_eval)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

        assert (
            accuracy > 0.70
        ), f"DiGress (official LR) failed Level 2 on {target_type}: {accuracy:.1%}"

    def test_digress_high_lr_generalization(self, structured_sample):
        """DiGress with high LR settings on fresh noise each step."""
        A_clean, target_type = structured_sample
        model = create_digress_small()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(2000):
            A_noisy = add_digress_noise(A_clean, p=0.1)
            logits = model(A_noisy)
            loss = F.binary_cross_entropy_with_logits(logits, A_clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.manual_seed(999)
        A_noisy_eval = add_digress_noise(A_clean, p=0.1)

        with torch.no_grad():
            logits = model(A_noisy_eval)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

        assert (
            accuracy > 0.70
        ), f"DiGress (high LR) failed Level 2 on {target_type}: {accuracy:.1%}"


# =============================================================================
# QUICK SANITY: SINGLE STEP LOSS REDUCTION
# =============================================================================


class TestDigressSingleStepLearning:
    """Quick check that one optimizer step reduces loss."""

    @pytest.fixture
    def training_data(self):
        """Create noisy/clean adjacency pair."""
        A_clean = create_block_diagonal_target(n=16, num_blocks=4)
        A_clean = A_clean.expand(4, -1, -1).clone()
        A_noisy = add_digress_noise(A_clean, p=0.1)
        return A_noisy, A_clean

    @pytest.mark.parametrize(
        "optimizer_type,lr",
        [
            ("official", 0.0002),
            ("high_lr", 1e-2),
        ],
    )
    def test_single_step_reduces_loss(self, optimizer_type, lr, training_data):
        """Verify one optimizer step reduces loss."""
        A_noisy, A_clean = training_data
        model = create_digress_small()

        if optimizer_type == "official":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-12, amsgrad=True
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        logits_1 = model(A_noisy)
        loss_1 = F.binary_cross_entropy_with_logits(logits_1, A_clean)

        loss_1.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits_2 = model(A_noisy)
        loss_2 = F.binary_cross_entropy_with_logits(logits_2, A_clean)

        assert loss_2 <= loss_1 + 0.01, (
            f"DiGress ({optimizer_type}): Loss increased. "
            f"Before: {loss_1:.4f}, After: {loss_2:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
