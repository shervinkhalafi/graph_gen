"""Tests for DiffusionModule multi-step diffusion training loop.

DiffusionModule extends BaseGraphModule with forward/reverse diffusion
by composing NoiseProcess, Sampler, NoiseSchedule, and GraphEvaluator.
These tests verify that the composition wires correctly and that each
lifecycle hook (training_step, validation_step, on_validation_epoch_end)
behaves as specified.

The tests use a GNN model instantiated directly, paired with a
ContinuousNoiseProcess wrapping GaussianNoise and a
ContinuousSampler, since those require no external dependencies and
work with simple binary-topology graphs lifted into edge-state space.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    ExactDensityNoiseProcess,
)
from tmgg.diffusion.sampler import CategoricalSampler, ContinuousSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.models.base import GraphModel

# -----------------------------------------------------------------------
# Shared fixtures and helpers
# -----------------------------------------------------------------------
from tmgg.models.gnn import GNN as _GNN
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
    _categorical_reconstruction_log_prob,
)
from tmgg.utils.noising.noise import GaussianNoise

_MODEL_CONFIG: dict[str, Any] = {
    "num_layers": 2,
    "num_terms": 2,
    "feature_dim_in": 10,
    "feature_dim_out": 10,
}


def _make_default_gnn() -> _GNN:
    """Instantiate the default test GNN model."""
    return _GNN(**_MODEL_CONFIG)


_TIMESTEPS = 10
_NUM_NODES = 6
_BATCH_SIZE = 3


def _make_schedule() -> NoiseSchedule:
    return NoiseSchedule("linear_ddpm", timesteps=_TIMESTEPS)


def _make_noise_process() -> ContinuousNoiseProcess:
    return ContinuousNoiseProcess(GaussianNoise(), schedule=_make_schedule())


def _make_sampler() -> ContinuousSampler:
    return ContinuousSampler()


def _make_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphData:
    """Create a continuous edge-state batch from a binary topology."""
    adj = torch.zeros(bs, n, n)
    for i in range(bs):
        # Random symmetric adjacency with no self-loops
        upper = (torch.rand(n, n) > 0.5).float()
        sym = upper.triu(diagonal=1)
        adj[i] = sym + sym.t()
    return GraphData.from_edge_state(adj)


def _make_categorical_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphData:
    """Create a categorical binary-topology batch for discrete tests."""
    adj = _make_batch(bs=bs, n=n).to_edge_state()
    return GraphData.from_binary_adjacency(adj)


def _make_module(
    evaluator: GraphEvaluator | None = None,
    loss_type: str = "mse",
    **overrides: Any,
) -> DiffusionModule:
    """Build a DiffusionModule with sensible test defaults."""
    schedule = _make_schedule()
    noise_process = _make_noise_process()
    sampler = _make_sampler()

    model = overrides.pop("model", _make_default_gnn())
    kwargs: dict[str, Any] = {
        "model": model,
        "noise_process": noise_process,
        "sampler": sampler,
        "noise_schedule": schedule,
        "evaluator": evaluator,
        "loss_type": loss_type,
        "num_nodes": _NUM_NODES,
        "eval_every_n_steps": 1,
    }
    kwargs.update(overrides)
    return DiffusionModule(**kwargs)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestInstantiation:
    """DiffusionModule can be created with all required components."""

    def test_creates_with_all_components(self) -> None:
        """Instantiation succeeds and stores every injected component.

        The module must hold references to the noise process, sampler,
        schedule, and evaluator that were passed at construction time.
        """
        evaluator = GraphEvaluator(eval_num_samples=10)
        module = _make_module(evaluator=evaluator)

        assert isinstance(module.noise_process, ContinuousNoiseProcess)
        assert isinstance(module.sampler, ContinuousSampler)
        assert isinstance(module.noise_schedule, NoiseSchedule)
        assert module.evaluator is evaluator

    def test_creates_without_evaluator(self) -> None:
        """Instantiation works when no evaluator is provided.

        The evaluator is optional; generative evaluation is simply
        skipped when it is None.
        """
        module = _make_module(evaluator=None)
        assert module.evaluator is None

    def test_T_property(self) -> None:
        """The ``T`` property reflects the schedule's timestep count."""
        module = _make_module()
        assert module.T == _TIMESTEPS

    def test_unknown_loss_type_raises(self) -> None:
        """An unrecognised loss_type must raise ValueError immediately.

        We fail fast rather than deferring the error to the first
        training step.
        """
        with pytest.raises(ValueError, match="Unknown loss_type"):
            _make_module(loss_type="huber")

    def test_cross_entropy_criterion(self) -> None:
        """loss_type='cross_entropy' installs CrossEntropyLoss."""
        module = _make_module(loss_type="cross_entropy")
        assert isinstance(module.criterion, torch.nn.CrossEntropyLoss)

    def test_mse_criterion(self) -> None:
        """loss_type='mse' installs MSELoss."""
        module = _make_module(loss_type="mse")
        assert isinstance(module.criterion, torch.nn.MSELoss)

    def test_bce_logits_criterion(self) -> None:
        """loss_type='bce_logits' installs BCEWithLogitsLoss."""
        module = _make_module(loss_type="bce_logits")
        assert isinstance(module.criterion, torch.nn.BCEWithLogitsLoss)


class TestForward:
    """Verify forward() delegates to the underlying model."""

    def test_forward_returns_graph_data(self) -> None:
        """forward() must return GraphData with the same batch shape.

        It is a thin wrapper around self.model(data, t=t), so the
        output shape should match what the GNN model produces.
        """
        module = _make_module()
        batch = _make_batch(bs=2, n=_NUM_NODES)
        result = module.forward(batch)
        assert isinstance(result, GraphData)
        assert result.E.shape[0] == 2

    def test_forward_passes_timestep(self) -> None:
        """forward() relays the ``t`` argument to the model.

        We patch the model's forward to capture the call and confirm
        the timestep tensor arrives unmodified.
        """
        module = _make_module()
        batch = _make_batch(bs=2, n=_NUM_NODES)
        t = torch.tensor([0.5, 0.8])

        original_forward = module.model.forward

        calls: list[tuple[GraphData, torch.Tensor | None]] = []

        def spy(data: GraphData, t: torch.Tensor | None = None) -> GraphData:
            calls.append((data, t))
            return original_forward(data, t=t)

        module.model.forward = spy  # type: ignore[assignment]
        module.forward(batch, t=t)

        assert len(calls) == 1
        passed_t = calls[0][1]
        assert passed_t is not None
        assert torch.allclose(passed_t, t)


class TestTrainingStep:
    """training_step must produce a finite scalar loss with gradient."""

    def test_uses_noise_process_condition_vector_for_model_input(self) -> None:
        """training_step should get model conditioning from the process.

        Test rationale: Step 1 removes module-owned conditioning semantics.
        The module should ask the noise process for the condition vector and
        pass that exact tensor to the model, rather than reconstructing
        ``t / T`` locally.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        fixed_t = torch.tensor([2, 5, 7], dtype=torch.long)
        expected_condition = torch.tensor([0.13, 0.57, 0.91], dtype=torch.float32)

        observed_t: list[torch.Tensor] = []
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: torch.Tensor) -> torch.Tensor:
            observed_t.append(t.detach().clone())
            return expected_condition.to(device=t.device)

        original_forward = module.model.forward

        def spy_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
        ) -> GraphData:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t)

        module.noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        module.model.forward = spy_forward  # type: ignore[assignment]

        with patch(
            "tmgg.training.lightning_modules.diffusion_module.torch.randint",
            return_value=fixed_t.clone(),
        ):
            module.training_step(batch, batch_idx=0)

        assert len(observed_t) == 1
        torch.testing.assert_close(observed_t[0], fixed_t)
        assert len(observed_condition) == 1
        assert observed_condition[0] is not None
        torch.testing.assert_close(observed_condition[0], expected_condition)

    def test_loss_is_finite_and_has_grad(self) -> None:
        """A single training step on a synthetic batch must return a
        finite scalar tensor that retains its computation graph (i.e.
        ``requires_grad`` is True). This confirms the forward noise,
        model prediction, and loss computation all compose correctly.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, "loss must be a scalar"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        assert loss.requires_grad, "loss must retain grad for backprop"

    def test_mse_loss_is_finite(self) -> None:
        """MSE loss variant also produces a finite scalar.

        MSE compares edge features directly rather than via argmax,
        so this exercises a different code path in _compute_loss.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)

        assert torch.isfinite(loss)
        assert loss.requires_grad


class TestComputeLoss:
    """_compute_loss handles both categorical and continuous modes."""

    def test_cross_entropy_with_one_hot_target(self) -> None:
        """CrossEntropyLoss path: converts one-hot targets to class
        indices and computes edge + node loss. Both must contribute
        to a finite scalar.
        """
        module = _make_module(loss_type="cross_entropy")
        batch = _make_categorical_batch(bs=2, n=4)
        # Simulate model prediction: same shape but with logits
        pred = GraphData(
            X=torch.randn_like(batch.X),
            E=torch.randn_like(batch.E),
            y=batch.y,
            node_mask=batch.node_mask,
        )
        loss = module._compute_loss(pred, batch)
        assert torch.isfinite(loss)

    def test_mse_loss_finite(self) -> None:
        """MSE path: compares edge tensors element-wise. The result
        must be finite and non-negative.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch(bs=2, n=4)
        pred = GraphData(
            X=torch.randn_like(batch.X),
            E=torch.randn_like(batch.E),
            y=batch.y,
            node_mask=batch.node_mask,
        )
        loss = module._compute_loss(pred, batch)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


class TestValidationStep:
    """validation_step computes loss and VLB but does not accumulate graphs."""

    def test_runs_without_evaluator(self) -> None:
        """When no evaluator is attached, validation_step should
        still run without error.
        """
        module = _make_module(evaluator=None)
        batch = _make_batch()
        # Should not raise
        module.validation_step(batch, batch_idx=0)

    def test_runs_with_evaluator(self) -> None:
        """With an evaluator attached, validation_step should still
        run without error and not accumulate any per-batch state.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator)
        batch = _make_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)
        # No _ref_graphs buffer should exist
        assert not hasattr(module, "_ref_graphs")


class TestOnValidationEpochEnd:
    """on_validation_epoch_end pulls refs from datamodule and evaluates."""

    def test_skips_when_no_evaluator(self) -> None:
        """on_validation_epoch_end is a no-op when evaluator is None.

        Must not raise even if called repeatedly.
        """
        module = _make_module(evaluator=None)
        # Should not raise
        module.on_validation_epoch_end()

    def test_skips_when_not_eval_step(self) -> None:
        """If global_step is not a multiple of eval_every_n_steps,
        the method returns without running generation.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=5)

        mock_trainer = MagicMock()
        mock_trainer.global_step = 2  # not a multiple of 5
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        # Should not raise — skips generation because global_step % 5 != 0
        module.on_validation_epoch_end()

    def test_calls_get_reference_graphs_on_datamodule(self) -> None:
        """At epoch end, on_validation_epoch_end should call
        get_reference_graphs on the datamodule to obtain refs.
        """
        import networkx as nx

        evaluator = GraphEvaluator(eval_num_samples=4)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=1)

        mock_dm = MagicMock()
        mock_dm.get_reference_graphs.return_value = [nx.path_graph(5) for _ in range(4)]

        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with patch.object(
            module, "generate_graphs", return_value=[nx.path_graph(5) for _ in range(4)]
        ):
            module.on_validation_epoch_end()

        mock_dm.get_reference_graphs.assert_called_once_with("val", 4)

    def test_skips_when_too_few_ref_graphs(self) -> None:
        """Generative evaluation needs at least 2 reference graphs for
        MMD. If the datamodule returns fewer, the method returns
        without attempting to generate or evaluate.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=1)

        mock_dm = MagicMock()
        mock_dm.get_reference_graphs.return_value = []  # no refs

        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        # Should not raise and should not call generate_graphs
        with patch.object(module, "generate_graphs") as mock_gen:
            module.on_validation_epoch_end()
        mock_gen.assert_not_called()


class TestTestStep:
    """test_step delegates to validation_step."""

    def test_test_step_runs_without_error(self) -> None:
        """test_step delegates to validation_step and should run
        without error, computing loss but not accumulating graphs.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator)
        batch = _make_batch(bs=2)

        # Should not raise
        module.test_step(batch, batch_idx=0)


class TestGenerateGraphs:
    """generate_graphs should use the module-owned sampler inputs."""

    def test_passes_module_noise_process_to_sampler(self) -> None:
        """Generation must pass the module's process instance through.

        Test rationale: Step 2 removes sampler-owned process state. The
        sampler should receive the exact ``module.noise_process`` object so
        setup-time initialization cannot diverge between training and
        generation.
        """
        module = _make_module()
        observed: dict[str, Any] = {}

        def fake_sample(
            *,
            model: GraphModel,
            noise_process: ContinuousNoiseProcess,
            num_graphs: int,
            num_nodes: int | torch.Tensor,
            device: torch.device,
            collector: Any = None,
        ) -> list[GraphData]:
            observed["model"] = model
            observed["noise_process"] = noise_process
            observed["num_graphs"] = num_graphs
            observed["num_nodes"] = num_nodes
            observed["device"] = device
            observed["collector"] = collector
            graph = GraphData.from_binary_adjacency(torch.zeros(_NUM_NODES, _NUM_NODES))
            return [graph for _ in range(num_graphs)]

        module.sampler.sample = fake_sample  # type: ignore[assignment]

        generated = module.generate_graphs(2)

        assert len(generated) == 2
        assert observed["model"] is module.model
        assert observed["noise_process"] is module.noise_process
        assert observed["num_graphs"] == 2


# -----------------------------------------------------------------------
# VLB wiring tests
# -----------------------------------------------------------------------

_DX = 2  # node classes (matches from_binary_adjacency output)
_DE = 2  # edge classes


class _CategoricalLogitModel(GraphModel):
    """Minimal categorical model stub for DiffusionModule tests."""

    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        del t
        node_logits = torch.where(
            data.X > 0,
            torch.full_like(data.X, 2.0),
            torch.full_like(data.X, -2.0),
        )
        edge_logits = torch.where(
            data.E > 0,
            torch.full_like(data.E, 2.0),
            torch.full_like(data.E, -2.0),
        )
        return GraphData(
            X=node_logits,
            E=edge_logits,
            y=data.y,
            node_mask=data.node_mask,
        )

    def get_config(self) -> dict[str, Any]:
        return {"type": "categorical_logit_stub"}


def _make_categorical_module(
    limit_distribution: Literal["uniform", "empirical_marginal"] = "uniform",
) -> DiffusionModule:
    """Build a DiffusionModule backed by a CategoricalNoiseProcess.

    Uses a categorical-logit stub model so the discrete loss path does not
    depend on a continuous edge-state model.
    """
    schedule = NoiseSchedule("cosine_iddpm", timesteps=_TIMESTEPS)
    cat_np = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=_DX,
        e_classes=_DE,
        limit_distribution=limit_distribution,
    )
    cat_sampler = CategoricalSampler()
    return DiffusionModule(
        model=_CategoricalLogitModel(),
        noise_process=cat_np,
        sampler=cat_sampler,
        noise_schedule=schedule,
        loss_type="cross_entropy",
        num_nodes=_NUM_NODES,
        eval_every_n_steps=1,
    )


class TestVLBWiring:
    """DiffusionModule computes VLB when noise_process is CategoricalNoiseProcess."""

    def test_categorical_module_uses_exact_density_process(self) -> None:
        """Categorical modules should expose exact-density VLB semantics."""
        module = _make_categorical_module()
        assert isinstance(module.noise_process, ExactDensityNoiseProcess)

    def test_continuous_module_is_not_exact_density(self) -> None:
        """Continuous modules should skip exact-density VLB bookkeeping."""
        module = _make_module()
        assert not isinstance(module.noise_process, ExactDensityNoiseProcess)

    def test_vlb_accumulators_start_empty(self) -> None:
        """All VLB accumulator lists should be empty at construction."""
        module = _make_categorical_module()
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0
        assert len(module._vlb_kl_diffusion) == 0
        assert len(module._vlb_reconstruction) == 0

    def test_validation_step_populates_vlb_accumulators(self) -> None:
        """After a validation step on a categorical module, each VLB
        accumulator should contain exactly one entry (one per batch).
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 1
        assert len(module._vlb_kl_prior) == 1
        assert len(module._vlb_kl_diffusion) == 1
        assert len(module._vlb_reconstruction) == 1

    def test_vlb_values_finite_with_soft_predictions(self) -> None:
        """All VLB components must be finite when the model outputs proper
        probability distributions (non-zero everywhere). The GNN model
        produces hard one-hot via ``from_adjacency``, which causes NaN in
        log-space. To test finiteness of the VLB pipeline we patch the
        model to return uniform probabilities.
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)

        original_forward = module.model.forward

        def soft_forward(data: GraphData, t: torch.Tensor | None = None) -> GraphData:
            result = original_forward(data, t=t)
            # Convert one-hot output to soft probabilities (uniform + epsilon)
            soft_X = torch.ones_like(result.X) / _DX
            soft_E = torch.ones_like(result.E) / _DE
            return GraphData(X=soft_X, E=soft_E, y=result.y, node_mask=result.node_mask)

        module.model.forward = soft_forward  # type: ignore[assignment]
        module.validation_step(batch, batch_idx=0)

        assert torch.isfinite(module._vlb_nll[0]), f"NLL is {module._vlb_nll[0]}"
        assert torch.isfinite(module._vlb_kl_prior[0])
        assert torch.isfinite(module._vlb_kl_diffusion[0])
        assert torch.isfinite(module._vlb_reconstruction[0])

    def test_continuous_validation_step_skips_vlb(self) -> None:
        """A continuous-noise module's validation step must not populate
        the VLB accumulators; they should remain empty.
        """
        module = _make_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0

    def test_on_validation_epoch_start_clears_accumulators(self) -> None:
        """``on_validation_epoch_start`` must reset all VLB accumulators."""
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 1
        module.on_validation_epoch_start()
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0
        assert len(module._vlb_kl_diffusion) == 0
        assert len(module._vlb_reconstruction) == 0

    def test_epoch_end_logs_vlb_and_clears(self) -> None:
        """``on_validation_epoch_end`` should log VLB metrics and clear
        accumulators for a categorical module with accumulated data.
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        # Mock trainer for current_epoch and log
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        logged: dict[str, float] = {}
        original_log = module.log

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
            logged[name] = float(value)
            original_log(name, value, **kwargs)

        module.log = capture_log  # type: ignore[assignment]
        module.on_validation_epoch_end()

        assert "val/epoch_NLL" in logged
        assert "val/kl_prior" in logged
        assert "val/kl_diffusion" in logged
        assert "val/reconstruction_logp" in logged

        # Accumulators must be cleared after logging
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0

    def test_epoch_end_no_vlb_when_continuous(self) -> None:
        """A continuous-noise module should not log VLB metrics at all."""
        module = _make_module(evaluator=None)
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        logged: dict[str, float] = {}
        original_log = module.log

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
            logged[name] = float(value)
            original_log(name, value, **kwargs)

        module.log = capture_log  # type: ignore[assignment]
        module.on_validation_epoch_end()

        assert "val/epoch_NLL" not in logged


class TestCategoricalReconstructionLogProb:
    """Module-local categorical reconstruction helper stays finite and masked."""

    def test_output_shape(self) -> None:
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        pred_probs = GraphData(
            X=torch.ones_like(batch.X) / _DX,
            E=torch.ones_like(batch.E) / _DE,
            y=batch.y,
            node_mask=batch.node_mask,
        )
        result = _categorical_reconstruction_log_prob(batch, pred_probs)
        assert result.shape == (_BATCH_SIZE,)

    def test_perfect_prediction_near_zero(self) -> None:
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        eps = 1e-7
        x_safe = batch.X + eps
        x_safe = x_safe / x_safe.sum(dim=-1, keepdim=True)
        e_safe = batch.E + eps
        e_safe = e_safe / e_safe.sum(dim=-1, keepdim=True)
        pred_probs = GraphData(X=x_safe, E=e_safe, y=batch.y, node_mask=batch.node_mask)
        result = _categorical_reconstruction_log_prob(batch, pred_probs)
        assert torch.isfinite(result).all()
        assert (result.abs() < 0.1).all()


class TestReconstructionMasking:
    """M-6: reconstruction log-prob must mask invalid positions."""

    def test_reconstruction_at_t1_variable_size_batch(self) -> None:
        """Reconstruction log-prob must be finite even with masked-out nodes.

        Test rationale: without masking, invalid positions can produce NaN
        via 0 * log(small) = 0 * -inf = NaN in reconstruction_logp. The fix
        sets pred[invalid] = 1 (so log(1) = 0) and clean[invalid] = 0.
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)

        # Mask out the last node in each graph to simulate variable sizes
        batch.node_mask[:, -1] = False
        # Put garbage at invalid positions to ensure masking actually works
        batch.X[:, -1] = torch.rand_like(batch.X[:, -1])
        batch.E[:, -1, :] = torch.rand_like(batch.E[:, -1, :])
        batch.E[:, :, -1] = torch.rand_like(batch.E[:, :, -1])

        # Patch model to return soft probabilities (uniform) so log-space
        # is well-defined at valid positions.
        original_forward = module.model.forward

        def soft_forward(data: GraphData, t: torch.Tensor | None = None) -> GraphData:
            result = original_forward(data, t=t)
            soft_X = torch.ones_like(result.X) / _DX
            soft_E = torch.ones_like(result.E) / _DE
            return GraphData(X=soft_X, E=soft_E, y=result.y, node_mask=result.node_mask)

        module.model.forward = soft_forward  # type: ignore[assignment]

        result = module._compute_reconstruction_at_t1(batch)  # pyright: ignore[reportPrivateUsage]
        assert result.shape == (_BATCH_SIZE,)
        assert torch.isfinite(
            result
        ).all(), f"Non-finite reconstruction log-prob: {result}"


class TestSetup:
    """DiffusionModule.setup() initialises categorical stationary PMFs."""

    def test_setup_with_uniform_is_noop(self) -> None:
        """Uniform categorical setup should skip marginal counting."""
        module = _make_categorical_module(limit_distribution="uniform")
        assert isinstance(module.noise_process, CategoricalNoiseProcess)
        assert module.noise_process._limit_x is not None
        assert module.noise_process._limit_e is not None
        original_x = module.noise_process._limit_x.clone()
        original_e = module.noise_process._limit_e.clone()

        mock_dm = MagicMock()
        mock_dm.get_size_distribution.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]
        module.setup(stage="fit")

        torch.testing.assert_close(module.noise_process._limit_x, original_x)
        torch.testing.assert_close(module.noise_process._limit_e, original_e)
        mock_dm.train_dataloader.assert_not_called()
        mock_dm.get_size_distribution.assert_called_once_with("train")

    def test_setup_continuous_is_noop(self) -> None:
        """For a continuous noise process, setup() should be a no-op."""
        module = _make_module()
        mock_trainer = MagicMock()
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]
        # Should not raise
        module.setup(stage="fit")

    def test_setup_marginal_initialises_transition(self) -> None:
        """Empirical-marginal setup should initialise from the train loader."""
        module = _make_categorical_module(limit_distribution="empirical_marginal")
        assert isinstance(module.noise_process, CategoricalNoiseProcess)
        assert module.noise_process._limit_x is None
        assert module.noise_process._limit_e is None

        mock_dm = MagicMock()
        mock_dm.train_dataloader.return_value = [
            _make_categorical_batch(bs=_BATCH_SIZE)
        ]

        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        module.setup(stage="fit")
        assert module.noise_process._limit_x is not None
        assert module.noise_process._limit_e is not None
        assert module.noise_process.is_initialized() is True
        mock_dm.get_size_distribution.assert_called_once_with("train")

    def test_setup_marginal_no_dm_raises(self) -> None:
        """setup() must raise RuntimeError if the datamodule is None but
        a marginal transition needs initialising.
        """
        module = _make_categorical_module(limit_distribution="empirical_marginal")

        mock_trainer = MagicMock()
        mock_trainer.datamodule = None
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(RuntimeError, match="DataModule required"):
            module.setup(stage="fit")
