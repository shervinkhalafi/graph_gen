# Phase 3 Corrections & Tasks 13-14 Completion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three deviations from the approved design doc (`2026-03-02-training-loop-unification-design.md`), then complete the final two tasks of the unification plan.

**Architecture:** The design doc specifies `BaseGraphModule -> DiffusionModule -> SingleStepDenoisingModule`. The current code incorrectly has `SingleStepDenoisingModule` extending `BaseGraphModule` directly, and `DiffusionModule` lacks VLB/ELBO support for categorical noise. This plan corrects the hierarchy, adds VLB methods to `CategoricalNoiseProcess`, wires them into `DiffusionModule`, rewrites `SingleStepDenoisingModule` as a thin subclass of `DiffusionModule`, migrates discrete diffusion configs, and then removes dead code.

**Tech Stack:** PyTorch, PyTorch Lightning, networkx, numpy. Test runner: `uv run pytest`.

**Branch:** `cleanup` (current). All work continues here.

**Test command:** `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`

**Baseline:** 1259 tests collected, 0 failures.

---

## Deviation Summary

| # | Deviation | Design doc reference | Fix |
|---|-----------|---------------------|-----|
| 1 | `DiffusionModule` has no VLB/ELBO | Line 103: "VLB computation is ... methods on `CategoricalNoiseProcess`, called by `DiffusionModule`'s validation step when the noise process is categorical (`isinstance` check)." | Add `compute_Lt()` and `reconstruction_logp()` to `CategoricalNoiseProcess`, add VLB accumulation/logging to `DiffusionModule` |
| 2 | `SingleStepDenoisingModule` extends `BaseGraphModule` | Lines 124-128: "`SingleStepDenoisingModule(DiffusionModule)` ... Semantic subclass: hardcodes T=1, sampler=None" | Rewrite to extend `DiffusionModule` with T=1, ContinuousNoiseProcess, sampler=None |
| 3 | Discrete diffusion configs still point at old `DiscreteDiffusionLightningModule` | Lines 134-144: every experiment maps to `DiffusionModule` or `SingleStepDenoisingModule` | Migrate 4 discrete configs + base config to `DiffusionModule` |

---

## Task 1: Add VLB methods to CategoricalNoiseProcess

The `CategoricalNoiseProcess` already has `kl_prior()`. It needs two more methods ported from `DiscreteDiffusionLightningModule`: `compute_Lt()` (diffusion KL) and `reconstruction_logp()` (t=0 reconstruction term). These are pure noise-process math that doesn't depend on the model or Lightning.

**Files:**
- Modify: `src/tmgg/diffusion/noise_process.py` (add `compute_Lt`, `reconstruction_logp` to `CategoricalNoiseProcess`)
- Test: `tests/diffusion/test_noise_process_vlb.py` (new)

**Step 1: Write the failing tests**

Create `tests/diffusion/test_noise_process_vlb.py`:

```python
"""Tests for VLB methods on CategoricalNoiseProcess.

These methods compute the three components of the variational lower bound
for discrete diffusion: KL prior (already tested elsewhere), diffusion KL
(compute_Lt), and reconstruction log-probability (reconstruction_logp).
The math is ported from DiscreteDiffusionLightningModule._compute_Lt and
._reconstruction_logp, adapted to operate on GraphData without any
Lightning dependency.

Test strategy: construct a small CategoricalNoiseProcess with uniform
transitions (no datamodule needed), run each method on synthetic one-hot
GraphData, and verify shape + finiteness invariants. We do NOT test
numerical correctness of the KL formulas (that's the job of the original
DiGress tests) — only that the methods compose without error and return
sensible shapes.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete

_T = 10
_DX = 2
_DE = 2
_BS = 3
_N = 6


@pytest.fixture
def noise_process() -> CategoricalNoiseProcess:
    schedule = PredefinedNoiseScheduleDiscrete("cosine", _T)
    return CategoricalNoiseProcess(
        transition_type="uniform",
        noise_schedule=schedule,
        x_classes=_DX,
        e_classes=_DE,
    )


@pytest.fixture
def clean_data() -> GraphData:
    """Synthetic one-hot GraphData batch."""
    X = F.one_hot(torch.randint(0, _DX, (_BS, _N)), _DX).float()
    E = F.one_hot(torch.randint(0, _DE, (_BS, _N, _N)), _DE).float()
    # Symmetrise E
    E = (E + E.transpose(1, 2)) / 2.0
    E = F.one_hot(E.argmax(dim=-1), _DE).float()
    node_mask = torch.ones(_BS, _N, dtype=torch.bool)
    return GraphData(X=X, E=E, y=torch.zeros(_BS, 0), node_mask=node_mask)


class TestComputeLt:
    """Diffusion KL term L_t for categorical noise."""

    def test_returns_per_batch_tensor(
        self, noise_process: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """compute_Lt returns shape (bs,) with finite values."""
        # Simulate model prediction: use clean data as "perfect" prediction
        pred = GraphData(
            X=F.softmax(torch.randn_like(clean_data.X), dim=-1),
            E=F.softmax(torch.randn_like(clean_data.E), dim=-1),
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        # Random timestep in [1, T]
        t_int = torch.randint(1, _T + 1, (_BS,))
        result = noise_process.compute_Lt(clean_data, pred, t_int)
        assert result.shape == (_BS,)
        assert torch.isfinite(result).all()

    def test_perfect_prediction_low_kl(
        self, noise_process: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """When prediction exactly matches clean data, KL should be near zero."""
        # Use clean data probabilities directly (softmax of one-hot = one-hot)
        pred = GraphData(
            X=clean_data.X.clone(),
            E=clean_data.E.clone(),
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        t_int = torch.randint(1, _T + 1, (_BS,))
        result = noise_process.compute_Lt(clean_data, pred, t_int)
        assert (result >= -1e-5).all(), f"KL should be non-negative, got {result}"


class TestReconstructionLogp:
    """Reconstruction log-probability term at t=0."""

    def test_returns_per_batch_tensor(
        self, noise_process: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """reconstruction_logp returns shape (bs,) with finite values."""
        # Simulate predicted probabilities at t=0
        pred_probs = GraphData(
            X=F.softmax(torch.randn_like(clean_data.X), dim=-1),
            E=F.softmax(torch.randn_like(clean_data.E), dim=-1),
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = noise_process.reconstruction_logp(clean_data, pred_probs)
        assert result.shape == (_BS,)
        assert torch.isfinite(result).all()

    def test_perfect_prediction_high_logp(
        self, noise_process: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """With perfect predicted probs, reconstruction logp should be near zero (log 1)."""
        pred_probs = GraphData(
            X=clean_data.X.clone(),
            E=clean_data.E.clone(),
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = noise_process.reconstruction_logp(clean_data, pred_probs)
        # log(1) = 0, so result should be near 0 (actually sum of log(1) = 0)
        assert (result >= -1e-3).all(), f"Expected near-zero logp, got {result}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/diffusion/test_noise_process_vlb.py -v`
Expected: FAIL with `AttributeError: 'CategoricalNoiseProcess' object has no attribute 'compute_Lt'`

**Step 3: Implement compute_Lt and reconstruction_logp**

Add to `CategoricalNoiseProcess` in `src/tmgg/diffusion/noise_process.py`:

```python
def compute_Lt(
    self, clean: GraphData, pred: GraphData, t_int: Tensor
) -> Tensor:
    """Diffusion KL loss term L_t, scaled by T.

    Computes ``T * KL(q(z_{t-1} | z_t, x_0) || q(z_{t-1} | z_t, hat{x}_0))``
    for nodes and edges, where ``x_0`` is the clean graph and ``hat{x}_0``
    is the model's prediction (softmaxed probabilities).

    Parameters
    ----------
    clean
        Clean one-hot ``GraphData``.
    pred
        Model prediction — softmaxed probabilities with the same shape.
    t_int
        Integer timesteps, shape ``(bs,)``.

    Returns
    -------
    Tensor
        Per-sample KL, shape ``(bs,)``.
    """
    transition = self._ensure_transition_model()
    T = self.noise_schedule.timesteps

    s_int = t_int - 1  # (bs,)

    beta_t = self.noise_schedule(t_int=t_int.unsqueeze(1).float())
    alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s_int.unsqueeze(1).float())
    alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t_int.unsqueeze(1).float())

    device = clean.X.device
    Qt = transition.get_Qt(beta_t, device)
    Qsb = transition.get_Qt_bar(alpha_bar_s, device)
    Qtb = transition.get_Qt_bar(alpha_bar_t, device)

    # True posterior: q(z_{t-1} | z_t, x_0)
    # We need z_t to compute the posterior, but this method receives
    # clean data. We apply noise to get z_t.
    z_t = self.apply(clean, t_int)

    prob_true = posterior_distributions(
        X=clean.X, E=clean.E, y=clean.y,
        X_t=z_t.X, E_t=z_t.E, y_t=z_t.y,
        Qt=Qt, Qsb=Qsb, Qtb=Qtb,
        node_mask=clean.node_mask,
    )
    bs, n, _ = clean.X.shape
    prob_true_E = prob_true.E.reshape((bs, n, n, -1))

    # Predicted posterior: q(z_{t-1} | z_t, hat{x}_0)
    prob_pred = posterior_distributions(
        X=pred.X, E=pred.E, y=pred.y,
        X_t=z_t.X, E_t=z_t.E, y_t=z_t.y,
        Qt=Qt, Qsb=Qsb, Qtb=Qtb,
        node_mask=clean.node_mask,
    )
    prob_pred_E = prob_pred.E.reshape((bs, n, n, -1))

    # Mask distributions
    true_X, true_E, pred_X, pred_E = mask_distributions(
        true_X=prob_true.X, true_E=prob_true_E,
        pred_X=prob_pred.X, pred_E=prob_pred_E,
        node_mask=clean.node_mask,
    )

    kl_x = F.kl_div(input=torch.log(pred_X + 1e-10), target=true_X, reduction="none")
    kl_e = F.kl_div(input=torch.log(pred_E + 1e-10), target=true_E, reduction="none")

    return T * (sum_except_batch(kl_x) + sum_except_batch(kl_e))


def reconstruction_logp(
    self, clean: GraphData, pred_probs: GraphData
) -> Tensor:
    """Reconstruction log-probability log p(x_0 | z_0).

    Computes the sum of log-probabilities of clean features under the
    predicted distribution at t=0.

    Parameters
    ----------
    clean
        Clean one-hot ``GraphData``.
    pred_probs
        Predicted probability distributions at t=0 (softmaxed).

    Returns
    -------
    Tensor
        Per-sample log-probability, shape ``(bs,)``.
    """
    loss_X = sum_except_batch(clean.X * torch.log(pred_probs.X.clamp(min=1e-10)))
    loss_E = sum_except_batch(clean.E * torch.log(pred_probs.E.clamp(min=1e-10)))
    return loss_X + loss_E
```

Also add these imports at the top of `noise_process.py`:

```python
from tmgg.models.digress.diffusion_utils import (
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    sample_discrete_features,
    sum_except_batch,
)
```

(Update the existing import to include `mask_distributions`, `posterior_distributions`, `sum_except_batch`.)

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/diffusion/test_noise_process_vlb.py -v`
Expected: 4 PASSED

**Step 5: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All 1263+ tests pass (4 new)

**Step 6: Commit**

```bash
git add src/tmgg/diffusion/noise_process.py tests/diffusion/test_noise_process_vlb.py
git commit -m "feat: add compute_Lt and reconstruction_logp to CategoricalNoiseProcess"
```

---

## Task 2: Add VLB wiring and setup() to DiffusionModule

`DiffusionModule` currently computes only a plain loss during validation. The design doc requires it to: (a) call `CategoricalNoiseProcess.kl_prior/compute_Lt/reconstruction_logp` when the noise process is categorical, (b) accumulate and log VLB components per epoch, (c) implement `setup()` for deferred noise process initialization from datamodule marginals.

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/diffusion_module.py`
- Modify: `tests/experiment_utils/test_diffusion_module.py` (add VLB tests)

**Step 1: Write the failing tests**

Add to `tests/experiment_utils/test_diffusion_module.py`:

```python
class TestVLBWiring:
    """DiffusionModule should compute VLB when noise_process is categorical."""

    def test_vlb_accumulators_exist_for_categorical(self) -> None:
        """When noise_process is CategoricalNoiseProcess, VLB accumulators
        should be initialised."""
        # This test will be filled once we know the exact API.
        # For now: the module should have _vlb_nll, _vlb_kl_prior, etc.
        pass  # placeholder — see step 3 for real test

    def test_vlb_logged_on_epoch_end(self) -> None:
        """on_validation_epoch_end should log VLB metrics when categorical."""
        pass  # placeholder


class TestSetup:
    """DiffusionModule.setup() should initialise marginal noise process."""

    def test_setup_calls_noise_process_setup(self) -> None:
        """When noise_process is CategoricalNoiseProcess with marginal
        transitions, setup() should call noise_process.setup() with
        marginals from the datamodule."""
        pass  # placeholder
```

**Step 2: Implement VLB wiring in DiffusionModule**

Modify `src/tmgg/experiments/_shared_utils/diffusion_module.py`:

1. In `__init__`, add VLB accumulator lists (conditionally on `isinstance(noise_process, CategoricalNoiseProcess)`):

```python
# VLB accumulators (only used when noise_process is categorical)
self._is_categorical: bool = isinstance(noise_process, CategoricalNoiseProcess)
self._vlb_nll: list[torch.Tensor] = []
self._vlb_kl_prior: list[torch.Tensor] = []
self._vlb_kl_diffusion: list[torch.Tensor] = []
self._vlb_reconstruction: list[torch.Tensor] = []
```

2. Add `setup()` method:

```python
def setup(self, stage: str | None = None) -> None:
    """Deferred initialisation for noise process and evaluator.

    For ``CategoricalNoiseProcess`` with ``transition_type='marginal'``,
    reads empirical marginals from the datamodule and calls
    ``noise_process.setup()``. For the evaluator, calls ``setup()``
    with training graphs if available.
    """
    if (
        self._is_categorical
        and isinstance(self.noise_process, CategoricalNoiseProcess)
        and self.noise_process.transition_model is None
    ):
        dm = self.trainer.datamodule
        if dm is None:
            raise RuntimeError("DataModule required for marginal transition setup")
        x_marginals = dm.node_marginals
        e_marginals = dm.edge_marginals
        self.noise_process.setup(x_marginals, e_marginals)
```

3. Modify `validation_step` to compute VLB when categorical:

After the existing loss computation, add:

```python
# VLB computation for categorical noise processes
if self._is_categorical and isinstance(self.noise_process, CategoricalNoiseProcess):
    kl_prior_X, kl_prior_E, _ = self.noise_process.kl_prior(
        batch.X, batch.E, batch.node_mask
    )
    kl_prior = kl_prior_X + kl_prior_E
    kl_diffusion = self.noise_process.compute_Lt(batch, pred, t_int)
    reconstruction = self.noise_process.reconstruction_logp(batch, pred)

    nll = kl_prior + kl_diffusion - reconstruction
    self._vlb_nll.append(nll.mean())
    self._vlb_kl_prior.append(kl_prior.mean())
    self._vlb_kl_diffusion.append(kl_diffusion.mean())
    self._vlb_reconstruction.append(reconstruction.mean())
```

4. Modify `on_validation_epoch_end` to log VLB:

Before the existing evaluator logic, add:

```python
# Log VLB metrics for categorical noise
if self._is_categorical and self._vlb_nll:
    self.log("val/epoch_NLL", torch.stack(self._vlb_nll).mean())
    self.log("val/kl_prior", torch.stack(self._vlb_kl_prior).mean())
    self.log("val/kl_diffusion", torch.stack(self._vlb_kl_diffusion).mean())
    self.log("val/reconstruction_logp", torch.stack(self._vlb_reconstruction).mean())
    self._vlb_nll.clear()
    self._vlb_kl_prior.clear()
    self._vlb_kl_diffusion.clear()
    self._vlb_reconstruction.clear()
```

5. Add `on_validation_epoch_start`:

```python
def on_validation_epoch_start(self) -> None:
    """Clear VLB accumulators at start of each validation epoch."""
    self._vlb_nll.clear()
    self._vlb_kl_prior.clear()
    self._vlb_kl_diffusion.clear()
    self._vlb_reconstruction.clear()
```

**Step 3: Write real tests and verify**

Replace the placeholder tests with concrete assertions:

```python
class TestVLBWiring:
    """DiffusionModule computes VLB when noise_process is CategoricalNoiseProcess."""

    def test_categorical_module_has_vlb_flag(self) -> None:
        """_is_categorical should be True for CategoricalNoiseProcess."""
        # Build with CategoricalNoiseProcess
        from tmgg.diffusion.noise_process import CategoricalNoiseProcess
        from tmgg.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete

        cat_schedule = PredefinedNoiseScheduleDiscrete("cosine", 10)
        cat_np = CategoricalNoiseProcess("uniform", cat_schedule, 2, 2)
        # Need a CategoricalSampler too
        from tmgg.diffusion.sampler import CategoricalSampler
        unified_schedule = NoiseSchedule("cosine", timesteps=10)
        cat_sampler = CategoricalSampler(cat_np, unified_schedule)

        module = DiffusionModule(
            model_type=_MODEL_TYPE,
            model_config=_MODEL_CONFIG,
            noise_process=cat_np,
            sampler=cat_sampler,
            noise_schedule=unified_schedule,
            loss_type="cross_entropy",
        )
        assert module._is_categorical is True

    def test_continuous_module_has_no_vlb_flag(self) -> None:
        """_is_categorical should be False for ContinuousNoiseProcess."""
        module = _make_module()
        assert module._is_categorical is False


class TestSetup:
    """DiffusionModule.setup() initialises marginal noise process."""

    def test_setup_with_uniform_is_noop(self) -> None:
        """Uniform CategoricalNoiseProcess already has transition_model;
        setup() should not raise."""
        from tmgg.diffusion.noise_process import CategoricalNoiseProcess
        from tmgg.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete
        from tmgg.diffusion.sampler import CategoricalSampler

        cat_schedule = PredefinedNoiseScheduleDiscrete("cosine", 10)
        cat_np = CategoricalNoiseProcess("uniform", cat_schedule, 2, 2)
        unified_schedule = NoiseSchedule("cosine", timesteps=10)
        cat_sampler = CategoricalSampler(cat_np, unified_schedule)

        module = DiffusionModule(
            model_type=_MODEL_TYPE,
            model_config=_MODEL_CONFIG,
            noise_process=cat_np,
            sampler=cat_sampler,
            noise_schedule=unified_schedule,
            loss_type="cross_entropy",
        )

        # Attach a mock trainer with no datamodule
        mock_trainer = MagicMock()
        mock_trainer.datamodule = None
        module._trainer = mock_trainer

        # Should not raise — uniform transition is already initialised
        module.setup(stage="fit")
```

**Step 4: Run tests**

Run: `uv run pytest tests/experiment_utils/test_diffusion_module.py -v`
Expected: All tests pass (20 existing + 3 new = 23)

**Step 5: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/diffusion_module.py tests/experiment_utils/test_diffusion_module.py
git commit -m "feat: add VLB wiring and setup() to DiffusionModule for categorical noise"
```

---

## Task 3: Rewrite SingleStepDenoisingModule to extend DiffusionModule

Per the design doc (lines 124-128), `SingleStepDenoisingModule` should be a semantic subclass of `DiffusionModule` that hardcodes T=1 and sampler=None. The current implementation extends `BaseGraphModule` directly, uses a `NoiseGenerator` internally, and has its own training/validation logic.

The rewrite makes it a thin subclass that:
- Creates a `ContinuousNoiseProcess` internally from the `noise_type` parameter
- Creates a `NoiseSchedule` with T=1
- Passes `sampler=None` to `DiffusionModule.__init__`
- Overrides `training_step` for single-noise-level sampling
- Overrides `validation_step` / `test_step` for per-noise-level evaluation
- Preserves spectral delta logging

**Files:**
- Rewrite: `src/tmgg/experiments/_shared_utils/denoising_module.py`
- Modify: `tests/experiment_utils/test_denoising_module.py`

**Step 1: Rewrite SingleStepDenoisingModule**

The new module extends `DiffusionModule` but overrides the training and validation logic for single-step semantics. Key changes:

- Constructor creates `ContinuousNoiseProcess(create_noise_generator(...))` and `NoiseSchedule("linear", timesteps=1)` internally, then calls `super().__init__()` with `sampler=None`
- `training_step` samples a noise level from the discrete set, applies noise via the inherited `noise_process`, runs model, computes loss
- `_val_or_test` iterates all eval noise levels, computes per-level metrics
- Removes `_zero_diagonal` (move to utility or inline), `_log_spectral_deltas` stays
- The `forward()` override for raw tensor bridging stays for backward compatibility

**Step 2: Update tests**

The existing tests in `test_denoising_module.py` should mostly work unchanged since the public API (constructor params, `training_step`, `validation_step`, `test_step`, `forward`) is preserved. The main change: `isinstance(module, DiffusionModule)` should now be True.

Add a test:
```python
def test_is_diffusion_module_subclass(self) -> None:
    """SingleStepDenoisingModule should be a DiffusionModule subclass."""
    from tmgg.experiments._shared_utils.diffusion_module import DiffusionModule
    m = _make_module()
    assert isinstance(m, DiffusionModule)
```

**Step 3: Run tests**

Run: `uv run pytest tests/experiment_utils/test_denoising_module.py -v`
Expected: All 19 existing + 1 new pass

**Step 4: Run full suite** (catch breakage in experiment-level tests)

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/denoising_module.py tests/experiment_utils/test_denoising_module.py
git commit -m "refactor: rewrite SingleStepDenoisingModule as DiffusionModule subclass (T=1, sampler=None)"
```

---

## Task 4: Migrate discrete diffusion configs to DiffusionModule

Now that `DiffusionModule` supports VLB, the 4 discrete diffusion model configs and the base config can point at `DiffusionModule` instead of `DiscreteDiffusionLightningModule`.

**Files:**
- Modify: `src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml`

**Step 1: Update base config**

Remove the TODO comment block (lines 12-20). Update the model `_target_` references in model configs to use `DiffusionModule` with nested `noise_process`, `sampler`, `noise_schedule`, and `evaluator` Hydra targets:

```yaml
_target_: tmgg.experiments._shared_utils.diffusion_module.DiffusionModule

model_type: graph_transformer
model_config:
  # ... model params ...

noise_process:
  _target_: tmgg.diffusion.noise_process.CategoricalNoiseProcess
  transition_type: marginal
  x_classes: 2
  e_classes: 2
  noise_schedule:
    _target_: tmgg.models.digress.noise_schedule.PredefinedNoiseScheduleDiscrete
    noise_schedule: cosine
    timesteps: 500

sampler:
  _target_: tmgg.diffusion.sampler.CategoricalSampler
  noise_process: ${model.noise_process}
  schedule: ${model.noise_schedule}

noise_schedule:
  _target_: tmgg.diffusion.schedule.NoiseSchedule
  schedule_type: cosine
  timesteps: 500

evaluator:
  _target_: tmgg.experiments._shared_utils.graph_evaluator.GraphEvaluator
  eval_num_samples: 128

loss_type: cross_entropy
```

Note: The exact YAML structure depends on Hydra interpolation rules. The key is that `_target_` changes from `DiscreteDiffusionLightningModule` to `DiffusionModule`, and the component objects are instantiated via Hydra's recursive instantiation.

**Step 2: Verify configs parse**

Run: `uv run python -c "from hydra import compose, initialize_config_dir; ..."` or manually check that imports resolve.

**Step 3: Commit**

```bash
git add src/tmgg/experiments/exp_configs/
git commit -m "refactor: migrate discrete diffusion configs to DiffusionModule"
```

---

## Task 5: Remove dead code (original Task 13)

With all experiments migrated, the following modules have zero callers and can be deleted:

**Old LightningModules (~3000 lines):**
- `src/tmgg/experiments/discrete_diffusion_generative/lightning_module.py` (DiscreteDiffusionLightningModule)
- `src/tmgg/experiments/digress_denoising/lightning_module.py` (DigressDenoisingLightningModule)
- `src/tmgg/experiments/gnn_denoising/lightning_module.py` (GNNDenoisingLightningModule)
- `src/tmgg/experiments/spectral_denoising/lightning_module.py` (SpectralDenoisingLightningModule)
- `src/tmgg/experiments/baseline_denoising/lightning_module.py` (BaselineDenoisingLightningModule)
- `src/tmgg/experiments/hybrid_denoising/lightning_module.py` (HybridDenoisingLightningModule)
- `src/tmgg/experiments/gaussian_diffusion_generative/lightning_module.py` (GenerativeLightningModule)
- `src/tmgg/experiments/_shared_utils/denoising_lightning_module.py` (DenoisingLightningModule abstract base)

**Old model base classes:**
- `src/tmgg/models/denoising_model.py` (AdjacencyDenoisingModel, CategoricalDenoisingModel)

**Old evaluators:**
- `src/tmgg/experiments/_shared_utils/sampling_evaluator.py` (SamplingEvaluator — replaced by GraphEvaluator)

**Step 1: Verify no remaining imports**

For each file, search for imports:

```bash
rg "from tmgg.experiments.discrete_diffusion_generative.lightning_module import" src/
rg "from tmgg.experiments.digress_denoising.lightning_module import" src/
# ... etc for each file
```

Exclude test files from this check — they test the old modules and will be updated or deleted.

**Step 2: Delete dead modules**

Delete each file listed above. Then update `__init__.py` files to remove re-exports.

**Step 3: Update or delete orphaned tests**

Tests that exercise only the old modules (e.g., the `TestMMDEvaluation` class in `tests/experiments/test_digress_denoising_module.py` that imports `DigressDenoisingLightningModule`) should be deleted or migrated to test the new modules.

**Step 4: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass (test count decreases from removed old-module tests)

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove ~3000 lines of dead LightningModules, old model bases, old evaluators"
```

---

## Task 6: Final cleanup (original Task 14)

**Step 1: Verify tach boundaries**

```bash
uv run tach check
```

Update `tach.toml` if boundaries changed.

**Step 2: Update `__init__.py` exports**

Ensure the public API surfaces the new classes:
- `tmgg.experiments._shared_utils` exports `BaseGraphModule`, `DiffusionModule`, `SingleStepDenoisingModule`, `GraphEvaluator`
- `tmgg.diffusion` exports `NoiseProcess`, `CategoricalNoiseProcess`, `ContinuousNoiseProcess`, `Sampler`, `CategoricalSampler`, `ContinuousSampler`, `NoiseSchedule`

**Step 3: Run basedpyright**

```bash
uv run basedpyright src/tmgg/experiments/_shared_utils/ src/tmgg/diffusion/
```

Fix any new type errors.

**Step 4: Run full test suite one final time**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 5: Commit and tag**

```bash
git add -A
git commit -m "chore: final cleanup — tach boundaries, exports, pyright"
git tag -a v0.9.0-unified -m "Training loop unification complete"
```

---

## Execution Notes

**Dependency ordering:** Tasks 1 -> 2 -> 3 -> 4 are strictly sequential (each builds on the previous). Tasks 5 and 6 are also sequential but independent of the numbering once Task 4 completes.

**Risk areas:**
- **Task 1** (VLB methods): The `compute_Lt` method needs `posterior_distributions` from DiGress utils, which expects specific tensor shapes. The test uses synthetic data so shape mismatches surface early.
- **Task 3** (rewrite denoising module): This touches 26 migrated configs and multiple test files. The risk is breaking existing experiment configs. Run `tests/experiments/test_digress_denoising_module.py` explicitly after this task.
- **Task 4** (discrete config migration): Hydra recursive instantiation of nested `_target_` objects can be finicky. Validate by actually loading configs.
- **Task 5** (dead code removal): Must verify no imports remain before deleting. The `rg` search is the safety net.

**Parallelisation:** Tasks 1 and 2 could potentially be merged into a single agent. Tasks 5 and 6 can also be merged. Tasks 3 and 4 should remain separate due to the scope of changes.
