"""Tests for TrainLossDiscrete.

Test rationale: The training loss directly determines gradient signal quality.
Incorrect masking or normalization leads to silent training failures where the
model appears to train but produces degenerate samples. These tests verify
the loss computation against known analytical results and confirm that
masking correctly excludes invalid positions.
"""

import math

import pytest
import torch

from tmgg.experiments._shared_utils.lightning_modules.train_loss_discrete import (
    TrainLossDiscrete,
)

BS = 4
N = 8
DX = 3
DE = 2


def _make_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer indices to one-hot float tensors."""
    return torch.nn.functional.one_hot(indices, num_classes).float()


@pytest.fixture()
def loss_fn() -> TrainLossDiscrete:
    return TrainLossDiscrete(lambda_E=5.0)


@pytest.fixture()
def full_mask() -> torch.Tensor:
    """All nodes are valid."""
    return torch.ones(BS, N, dtype=torch.bool)


class TestPerfectPrediction:
    """When predicted == true, the cross-entropy should be near zero."""

    def test_perfect_nodes_and_edges(
        self, loss_fn: TrainLossDiscrete, full_mask: torch.Tensor
    ) -> None:
        """Identical one-hot pred and true yields near-zero loss."""
        true_X = _make_one_hot(torch.randint(0, DX, (BS, N)), DX)
        true_E = _make_one_hot(torch.randint(0, DE, (BS, N, N)), DE)

        loss = loss_fn(
            pred_X=true_X.clone(),
            pred_E=true_E.clone(),
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        # mask_distributions adds eps ~ 1e-7 and renormalizes, so loss won't
        # be exactly 0 but should be very small
        assert loss.item() < 0.01, f"Perfect prediction loss too high: {loss.item()}"


class TestUniformPrediction:
    """Uniform prediction vs one-hot target should yield loss ~ log(K)."""

    def test_uniform_node_prediction(self, full_mask: torch.Tensor) -> None:
        """With lambda_E=0, uniform node prediction gives loss ~ log(DX)."""
        loss_fn = TrainLossDiscrete(lambda_E=0.0)
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        pred_X = torch.ones(BS, N, DX) / DX
        # Edges don't matter since lambda_E=0
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_E = true_E.clone()

        loss = loss_fn(
            pred_X=pred_X,
            pred_E=pred_E,
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        expected = math.log(DX)
        assert (
            abs(loss.item() - expected) < 0.1
        ), f"Uniform node loss {loss.item():.4f} != expected log({DX})={expected:.4f}"

    def test_uniform_edge_prediction(self, full_mask: torch.Tensor) -> None:
        """With lambda_E=1 and perfect node pred, uniform edge prediction
        gives edge contribution ~ log(DE).
        """
        loss_fn = TrainLossDiscrete(lambda_E=1.0)
        # Perfect node prediction
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        pred_X = true_X.clone()
        # Uniform edge prediction
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_E = torch.ones(BS, N, N, DE) / DE

        loss = loss_fn(
            pred_X=pred_X,
            pred_E=pred_E,
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        expected_edge = math.log(DE)
        # Node loss is near zero, so total ~ 1.0 * log(DE)
        assert (
            abs(loss.item() - expected_edge) < 0.15
        ), f"Uniform edge loss {loss.item():.4f} != expected log({DE})={expected_edge:.4f}"


class TestMasking:
    """Masked positions should not contribute to the loss."""

    def test_masked_nodes_do_not_affect_loss(self, loss_fn: TrainLossDiscrete) -> None:
        """Loss should be the same whether extra nodes are masked or absent.

        We create two scenarios: (A) N nodes all valid, (B) N+2 nodes where
        the last 2 are masked. Both should produce the same loss because the
        masked nodes carry garbage values that should be ignored.
        """
        n_valid = 6
        node_mask_a = torch.ones(BS, n_valid, dtype=torch.bool)

        true_X_a = _make_one_hot(torch.randint(0, DX, (BS, n_valid)), DX)
        true_E_a = _make_one_hot(torch.randint(0, DE, (BS, n_valid, n_valid)), DE)
        pred_X_a = torch.ones(BS, n_valid, DX) / DX
        pred_E_a = torch.ones(BS, n_valid, n_valid, DE) / DE

        loss_a = loss_fn(
            pred_X=pred_X_a.clone(),
            pred_E=pred_E_a.clone(),
            true_X=true_X_a.clone(),
            true_E=true_E_a.clone(),
            node_mask=node_mask_a,
        )

        # Scenario B: pad with 2 extra garbage nodes, masked out
        n_padded = n_valid + 2
        node_mask_b = torch.ones(BS, n_padded, dtype=torch.bool)
        node_mask_b[:, n_valid:] = False

        true_X_b = torch.randn(BS, n_padded, DX)  # garbage in padded positions
        true_X_b[:, :n_valid] = true_X_a
        true_E_b = torch.randn(BS, n_padded, n_padded, DE)
        true_E_b[:, :n_valid, :n_valid] = true_E_a
        pred_X_b = torch.randn(BS, n_padded, DX)
        pred_X_b[:, :n_valid] = pred_X_a
        pred_E_b = torch.randn(BS, n_padded, n_padded, DE)
        pred_E_b[:, :n_valid, :n_valid] = pred_E_a

        loss_b = loss_fn(
            pred_X=pred_X_b,
            pred_E=pred_E_b,
            true_X=true_X_b,
            true_E=true_E_b,
            node_mask=node_mask_b,
        )

        assert (
            abs(loss_a.item() - loss_b.item()) < 0.01
        ), f"Masked loss {loss_b.item():.4f} differs from unmasked {loss_a.item():.4f}"


class TestLambdaEWeighting:
    """Verify that lambda_E scales the edge loss contribution."""

    def test_higher_lambda_increases_loss(self, full_mask: torch.Tensor) -> None:
        """Doubling lambda_E should increase total loss when edge prediction
        is imperfect (uniform) but node prediction is perfect.
        """
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_X = true_X.clone()  # perfect node prediction
        pred_E = torch.ones(BS, N, N, DE) / DE  # uniform edge prediction

        loss_low = TrainLossDiscrete(lambda_E=1.0)(
            pred_X=pred_X.clone(),
            pred_E=pred_E.clone(),
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        loss_high = TrainLossDiscrete(lambda_E=2.0)(
            pred_X=pred_X.clone(),
            pred_E=pred_E.clone(),
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        # Edge loss contribution doubles, so total loss should roughly double
        # (node loss is near-zero)
        assert loss_high.item() > loss_low.item(), (
            f"Higher lambda_E did not increase loss: "
            f"lambda_E=2.0 -> {loss_high.item():.4f}, "
            f"lambda_E=1.0 -> {loss_low.item():.4f}"
        )
        ratio = loss_high.item() / max(loss_low.item(), 1e-10)
        assert 1.8 < ratio < 2.2, f"Expected loss ratio ~2.0 but got {ratio:.2f}"
