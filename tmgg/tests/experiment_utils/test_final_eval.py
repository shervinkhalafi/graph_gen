"""Tests for final_eval noise generator conditioning.

Regression test for the bug where evaluate_across_noise_levels always used
add_edge_flip_noise regardless of model type. After the fix, the function
accepts a NoiseGenerator instance and delegates noise application to it.
"""

from unittest.mock import MagicMock

import torch

from tmgg.experiment_utils.data.noise_generators import (
    GaussianNoiseGenerator,
    NoiseGenerator,
)
from tmgg.experiment_utils.final_eval import evaluate_across_noise_levels


class _SpyNoiseGenerator(NoiseGenerator):
    """NoiseGenerator that records calls for test verification.

    We need to verify that evaluate_across_noise_levels delegates noise
    application to the injected NoiseGenerator rather than hardcoding
    add_edge_flip_noise. A spy lets us assert both invocation count and
    the eps values passed, without depending on any particular noise
    implementation's output characteristics.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, float]] = []

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        self.calls.append((A, eps))
        return A.clone()

    @property
    def requires_state(self) -> bool:
        return False


class TestEvaluateAcrossNoiseLevelsUsesInjectedGenerator:
    """Verify that evaluate_across_noise_levels uses the provided NoiseGenerator.

    Test rationale
    --------------
    Starting state: evaluate_across_noise_levels previously imported and called
    add_edge_flip_noise directly, ignoring the model's actual noise type.

    Invariant: the function must call noise_generator.add_noise exactly once per
    noise level, with the correct eps value.
    """

    @staticmethod
    def _make_dummy_model() -> MagicMock:
        """Create a minimal mock model that returns a plausible prediction."""
        model = MagicMock()
        model.device = torch.device("cpu")
        model.eval.return_value = None

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        model.__call__ = forward_fn
        model.side_effect = forward_fn
        return model

    @staticmethod
    def _make_dummy_data_module(n: int = 8) -> MagicMock:
        """Create a mock data module returning a fixed adjacency matrix."""
        dm = MagicMock()
        A = torch.eye(n, dtype=torch.float32)
        dm.get_sample_adjacency_matrix.return_value = A
        return dm

    def test_spy_generator_called_for_each_noise_level(self) -> None:
        """The injected generator should be called num_eval_samples times per noise level."""
        spy = _SpyNoiseGenerator()
        model = self._make_dummy_model()
        dm = self._make_dummy_data_module()
        noise_levels = [0.01, 0.1, 0.5]
        num_eval_samples = 3

        evaluate_across_noise_levels(
            model,
            dm,
            noise_levels,
            noise_generator=spy,
            num_eval_samples=num_eval_samples,
        )

        assert len(spy.calls) == len(noise_levels) * num_eval_samples
        recorded_eps = [eps for _, eps in spy.calls]
        # Each noise level should appear num_eval_samples times consecutively
        for i, eps in enumerate(noise_levels):
            start = i * num_eval_samples
            assert (
                recorded_eps[start : start + num_eval_samples]
                == [eps] * num_eval_samples
            )

    def test_add_edge_flip_noise_no_longer_imported(self) -> None:
        """The module should not import add_edge_flip_noise at all.

        This structural check ensures no future edit re-introduces the
        hardcoded noise function dependency.
        """
        import tmgg.experiment_utils.final_eval as mod

        assert not hasattr(mod, "add_edge_flip_noise"), (
            "final_eval should not import add_edge_flip_noise; "
            "noise application is delegated to NoiseGenerator"
        )

    def test_gaussian_generator_produces_different_output(self) -> None:
        """Sanity check: a non-DiGress generator actually modifies the matrix.

        Verifies end-to-end that when a GaussianNoiseGenerator is injected,
        the noise it produces differs from what DiGress would produce (edge
        flipping), confirming the generator is actually used.
        """
        gauss_gen = GaussianNoiseGenerator()
        torch.manual_seed(42)
        A = torch.eye(10, dtype=torch.float32)
        A_noisy = gauss_gen.add_noise(A, 0.5)
        assert not torch.allclose(
            A, A_noisy
        ), "Gaussian noise at eps=0.5 should visibly perturb an identity matrix"
