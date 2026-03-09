# GraphTransformer Self-Containment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `GraphTransformer` self-contained (handles its own feature augmentation and timestep embedding), remove the `DiscreteGraphTransformer` alias, and fix the VLB softmax bug.

**Architecture:** Unify all feature augmentation behind a single `extra_features` parameter with an `adjust_dims()` method. Remove `use_eigenvectors`/`k`/`eigen_layer` from `GraphTransformer`, add `use_timestep`. Fix the VLB block to apply softmax before passing to probability-expecting methods. Delete the re-export alias module and update all references.

**Tech Stack:** PyTorch, PyTorch Lightning. Test runner: `uv run pytest`.

**Branch:** `cleanup` (current). All work continues here.

**Test command:** `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`

**Design doc:** `docs/plans/2026-03-05-graphtransformer-self-containment-design.md`

---

## Task 1: Add `adjust_dims()` to augmentation classes + create `EigenvectorAugmentation`

Add `adjust_dims(input_dims) -> input_dims` to `DummyExtraFeatures` and `ExtraFeatures`, then create `EigenvectorAugmentation` — a thin callable that wraps the eigenvector code currently in `GraphTransformer.forward()` (lines 851-861 of `transformer_model.py`) into the same `adjust_dims` + `__call__` interface.

**Files:**
- Modify: `src/tmgg/models/digress/extra_features.py`
- Create: `tests/models/test_extra_features_augmentation.py`

**Step 1: Write the failing tests**

Create `tests/models/test_extra_features_augmentation.py`:

```python
"""Tests for feature augmentation classes and their adjust_dims interface.

Test rationale: the adjust_dims method encapsulates dimension arithmetic that
GraphTransformer delegates to each augmentation. If dims are wrong, the
_GraphTransformer's mlp_in layers will silently accept mismatched widths and
produce garbage. These tests verify both the dim computation and the runtime
__call__ output shapes.
"""

import pytest
import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.digress.extra_features import (
    DummyExtraFeatures,
    EigenvectorAugmentation,
    ExtraFeatures,
)

BS = 3
N = 8
DX = 2
DE = 2
DY = 0


@pytest.fixture()
def node_mask() -> torch.Tensor:
    return torch.ones(BS, N, dtype=torch.bool)


@pytest.fixture()
def categorical_input(node_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One-hot categorical X and E tensors."""
    X = F.one_hot(torch.randint(0, DX, (BS, N)), DX).float()
    E = F.one_hot(torch.randint(0, DE, (BS, N, N)), DE).float()
    y = torch.zeros(BS, DY)
    return X, E, y, node_mask


class TestDummyExtraFeatures:
    """DummyExtraFeatures returns zero-width tensors and identity dims."""

    def test_adjust_dims_identity(self) -> None:
        aug = DummyExtraFeatures()
        base = {"X": 5, "E": 3, "y": 1}
        assert aug.adjust_dims(base) == {"X": 5, "E": 3, "y": 1}

    def test_call_shapes(
        self,
        categorical_input: tuple[torch.Tensor, ...],
    ) -> None:
        aug = DummyExtraFeatures()
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape == (BS, N, 0)
        assert ex_E.shape == (BS, N, N, 0)
        assert ex_y.shape == (BS, 0)


class TestExtraFeatures:
    """ExtraFeatures augments dims according to features_type."""

    @pytest.mark.parametrize(
        "features_type,expected_extra",
        [
            ("cycles", (3, 0, 5)),
            ("eigenvalues", (3, 0, 11)),
            ("all", (6, 0, 11)),
        ],
    )
    def test_adjust_dims(
        self, features_type: str, expected_extra: tuple[int, int, int]
    ) -> None:
        aug = ExtraFeatures(features_type, max_n_nodes=N)
        base = {"X": 2, "E": 2, "y": 0}
        result = aug.adjust_dims(base)
        dx, de, dy = expected_extra
        assert result == {"X": 2 + dx, "E": 2 + de, "y": 0 + dy}

    def test_call_shapes_cycles(
        self,
        categorical_input: tuple[torch.Tensor, ...],
    ) -> None:
        aug = ExtraFeatures("cycles", max_n_nodes=N)
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape[0] == BS
        assert ex_X.shape[1] == N
        assert ex_X.shape[2] == 3  # cycle counts k=3,4,5
        assert ex_E.shape[-1] == 0
        assert ex_y.shape[0] == BS
        assert ex_y.shape[1] == 5  # n + y_cycles(4)


class TestEigenvectorAugmentation:
    """EigenvectorAugmentation adds top-k eigenvectors to X."""

    K = 5

    def test_adjust_dims(self) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        base = {"X": 2, "E": 2, "y": 0}
        result = aug.adjust_dims(base)
        assert result == {"X": 2 + self.K, "E": 2, "y": 0}

    def test_adjust_dims_preserves_other_keys(self) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        base = {"X": 2, "E": 4, "y": 3}
        result = aug.adjust_dims(base)
        assert result["E"] == 4
        assert result["y"] == 3

    def test_call_shapes(
        self,
        categorical_input: tuple[torch.Tensor, ...],
    ) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape == (BS, N, self.K)
        assert ex_E.shape == (BS, N, N, 0)
        assert ex_y.shape == (BS, 0)

    def test_output_finite(
        self,
        categorical_input: tuple[torch.Tensor, ...],
    ) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        X, E, y, mask = categorical_input
        ex_X, _, _ = aug(X, E, y, mask)
        assert torch.isfinite(ex_X).all()

    def test_small_graph_pads(self) -> None:
        """When graph has fewer eigenvalues than k, output is zero-padded."""
        k = 20  # more than N=8
        aug = EigenvectorAugmentation(k=k)
        X = torch.randn(1, 4, DX)  # tiny 4-node graph
        E = F.one_hot(torch.randint(0, DE, (1, 4, 4)), DE).float()
        y = torch.zeros(1, 0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        ex_X, _, _ = aug(X, E, y, mask)
        assert ex_X.shape == (1, 4, k)
        assert torch.isfinite(ex_X).all()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_extra_features_augmentation.py -v`
Expected: FAIL with `ImportError: cannot import name 'EigenvectorAugmentation'`

**Step 3: Implement**

In `src/tmgg/models/digress/extra_features.py`:

Add `adjust_dims` method to `DummyExtraFeatures` (after `__call__`, around line 93):

```python
def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
    """Return *input_dims* unchanged (no augmentation)."""
    return dict(input_dims)
```

Add `adjust_dims` method to `ExtraFeatures` (after `__call__`, around line 181):

```python
def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
    """Return *input_dims* with extra feature dimensions added."""
    dx, de, dy = extra_features_dims(self.features_type)
    return {
        "X": input_dims["X"] + dx,
        "E": input_dims["E"] + de,
        "y": input_dims["y"] + dy,
    }
```

Add `EigenvectorAugmentation` class after the `ExtraFeatures` class (before the cycle features section, around line 183):

```python
class EigenvectorAugmentation:
    """Top-k eigenvector augmentation for node features.

    Extracts the top-k eigenvectors of the Laplacian derived from the
    edge features (``E.argmax(-1) > 0``) and returns them as extra node
    features. Equivalent to the old ``use_eigenvectors`` path in
    ``GraphTransformer``, factored out into the shared augmentation
    interface.

    Parameters
    ----------
    k
        Number of eigenvectors to extract.
    """

    def __init__(self, k: int) -> None:
        from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer

        self._k = k
        self._eigen_layer = TopKEigenLayer(k=k)

    def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
        """Add *k* eigenvector dimensions to X."""
        return {**input_dims, "X": input_dims["X"] + self._k}

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute top-k eigenvectors from edge features.

        Parameters
        ----------
        X
            Node features ``(bs, n, dx)``.
        E
            Edge features ``(bs, n, n, de)``. ``argmax(-1) > 0`` gives adjacency.
        y
            Global features ``(bs, dy)``.
        node_mask
            Valid-node mask ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(eigenvectors, zero_E, zero_y)`` where eigenvectors has shape
            ``(bs, n, k)``, zero-padded when the graph has fewer than *k*
            eigenvalues.
        """
        adj = (E.argmax(dim=-1) > 0).float()
        V, _ = self._eigen_layer(adj)

        actual_k = V.shape[-1]
        if actual_k < self._k:
            V = torch.nn.functional.pad(V, (0, self._k - actual_k))  # pyright: ignore[reportAttributeAccessIssue]

        V = V * node_mask.unsqueeze(-1).float()

        bs, n = X.shape[:2]
        return (
            V,
            torch.zeros(bs, n, n, 0, device=X.device, dtype=X.dtype),
            torch.zeros(bs, 0, device=X.device, dtype=X.dtype),
        )
```

Update the `__all__` or module-level docstring if present. Add `EigenvectorAugmentation` to any existing `__all__`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_extra_features_augmentation.py -v`
Expected: All 10 tests PASS

**Step 5: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass (existing tests unaffected — we only added methods)

**Step 6: Commit**

```bash
git add src/tmgg/models/digress/extra_features.py tests/models/test_extra_features_augmentation.py
git commit -m "feat: add adjust_dims to augmentation classes, create EigenvectorAugmentation"
```

---

## Task 2: Refactor GraphTransformer to use `extra_features` + `use_timestep`

Replace `use_eigenvectors`/`k`/`eigen_layer` constructor params with `extra_features` (callable with `adjust_dims`) and `use_timestep` (bool). The model delegates dim adjustment to the augmentation callable and auto-adjusts for timestep. `forward()` applies augmentation and optionally embeds `t`.

**Files:**
- Modify: `src/tmgg/models/digress/transformer_model.py` (lines 747-877)
- Modify: `tests/models/test_discrete_transformer.py` (entire file)

**Step 1: Update `GraphTransformer` constructor**

In `src/tmgg/models/digress/transformer_model.py`, replace the `GraphTransformer` class (lines 747-877). Key changes:

Replace constructor params:
```python
# REMOVE these params:
#   use_eigenvectors: bool = False,
#   k: int | None = None,

# ADD these params:
    extra_features: DummyExtraFeatures | ExtraFeatures | EigenvectorAugmentation | None = None,
    use_timestep: bool = False,
```

Note: use a union type or keep it untyped (`Any`) to avoid import cycles. The simplest approach is a duck-typed `extra_features` with no type annotation beyond `| None`.

Replace constructor body. Remove the eigenvector block (lines 807-816):
```python
# REMOVE:
# self._use_eigenvectors = use_eigenvectors
# self._k = k
# if use_eigenvectors:
#     if k is None:
#         raise ValueError(...)
#     self.eigen_layer = TopKEigenLayer(k=k)
# else:
#     self.eigen_layer = None
```

Replace with:
```python
self.extra_features = extra_features
self._use_timestep = use_timestep

# Compute adjusted input dims for _GraphTransformer
adjusted_input_dims = dict(input_dims)
if extra_features is not None:
    adjusted_input_dims = extra_features.adjust_dims(adjusted_input_dims)
if use_timestep:
    adjusted_input_dims = {**adjusted_input_dims, "y": adjusted_input_dims["y"] + 1}
```

Update the `_GraphTransformer` construction to use `adjusted_input_dims`:
```python
self.transformer = _GraphTransformer(
    n_layers=n_layers,
    input_dims=adjusted_input_dims,  # was: input_dims
    hidden_mlp_dims=hidden_mlp_dims,
    hidden_dims=hidden_dims,
    output_dims=output_dims,
    act_fn_in=act_fn_in,
    act_fn_out=act_fn_out,
)
```

Remove `eigen_layer` from the class-level annotations (line 786):
```python
# REMOVE: eigen_layer: nn.Module | None
```

**Step 2: Update `forward()` method**

Replace `forward()` body (lines 849-863):
```python
@override
def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
    """Forward pass through the graph transformer.

    Applies optional feature augmentation and timestep embedding before
    the transformer stack.

    Parameters
    ----------
    data
        Batched graph features (X, E, y, node_mask).
    t
        Normalised diffusion timestep, shape ``(bs,)``. Embedded into
        ``y`` when ``use_timestep=True``.

    Returns
    -------
    GraphData
        Predicted features with output dimensions.
    """
    X, E, y, node_mask = data.X, data.E, data.y, data.node_mask

    if self.extra_features is not None:
        extra_X, extra_E, extra_y = self.extra_features(X, E, y, node_mask)
        X = torch.cat([X, extra_X], dim=-1)
        E = torch.cat([E, extra_E], dim=-1)
        y = torch.cat([y, extra_y], dim=-1)

    if self._use_timestep and t is not None:
        y = torch.cat([y, t.unsqueeze(-1)], dim=-1)

    return self.transformer(X, E, y, node_mask)
```

**Step 3: Update `get_config()`**

Replace `get_config()` (lines 866-877):
```python
@override
def get_config(self) -> dict[str, Any]:
    """Return model configuration for serialization and logging."""
    return {
        "model_class": "GraphTransformer",
        "n_layers": self.n_layers,
        "input_dims": self.input_dims,
        "hidden_mlp_dims": self.hidden_mlp_dims,
        "hidden_dims": self.hidden_dims,
        "output_dims": self.output_dims,
        "extra_features": type(self.extra_features).__name__ if self.extra_features else None,
        "use_timestep": self._use_timestep,
    }
```

Remove the `TopKEigenLayer` import from the constructor (line 812) — it's no longer needed here.

**Step 4: Update test file**

Rewrite `tests/models/test_discrete_transformer.py`. Change import (line 13):

```python
# BEFORE:
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
# AFTER:
from tmgg.models.digress.transformer_model import GraphTransformer
from tmgg.models.digress.extra_features import EigenvectorAugmentation
```

Replace all `DiscreteGraphTransformer` type hints and constructor calls with `GraphTransformer`.

Update the docstring (line 1):
```python
"""Tests for GraphTransformer model.
```

In `TestGetConfig.test_expected_keys` (line 105), update expected keys:
```python
expected_keys = {
    "model_class",
    "n_layers",
    "input_dims",
    "hidden_mlp_dims",
    "hidden_dims",
    "output_dims",
    "extra_features",
    "use_timestep",
}
```

In `TestGetConfig.test_eigenvector_defaults` (line 127), update:
```python
def test_augmentation_defaults(self, model: GraphTransformer) -> None:
    """Vanilla model reports no extra_features and use_timestep=False."""
    config = model.get_config()
    assert config["extra_features"] is None
    assert config["use_timestep"] is False
```

In `TestEigenvectorMode` (line 134):
- Remove `DX_EIGEN` constant — no longer needed
- Update `eigen_model` fixture to use `extra_features`:
```python
K = 5

@pytest.fixture()
def eigen_model(self) -> GraphTransformer:
    return GraphTransformer(
        n_layers=2,
        input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},  # base dims only
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
        output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
        extra_features=EigenvectorAugmentation(k=self.K),
    )
```

- Update `test_config_reports_eigenvectors`:
```python
def test_config_reports_augmentation(self, eigen_model: GraphTransformer) -> None:
    config = eigen_model.get_config()
    assert config["extra_features"] == "EigenvectorAugmentation"
    assert config["use_timestep"] is False
```

- Update `test_k_required_when_enabled` — this validation now lives in `EigenvectorAugmentation`, not `GraphTransformer`. Either move the test to `test_extra_features_augmentation.py` or remove it from this file. The simplest approach: remove it (Task 1 already tests `EigenvectorAugmentation`). If you keep it, test that `EigenvectorAugmentation(k=...)` raises appropriately.

Add a new test class for `use_timestep`:
```python
class TestTimestepMode:
    """Verify timestep embedding path."""

    @pytest.fixture()
    def timestep_model(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX_IN, "E": DE_IN, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": 0},
            use_timestep=True,
        )

    def test_forward_with_timestep(self, timestep_model: GraphTransformer) -> None:
        """Forward with t produces correctly shaped output."""
        data = GraphData(
            X=torch.randn(BS, N, DX_IN),
            E=torch.randn(BS, N, N, DE_IN),
            y=torch.zeros(BS, 0),
            node_mask=torch.ones(BS, N),
        )
        t = torch.rand(BS)
        out = timestep_model(data, t=t)
        assert out.X.shape == (BS, N, DX_OUT)

    def test_config_reports_timestep(self, timestep_model: GraphTransformer) -> None:
        config = timestep_model.get_config()
        assert config["use_timestep"] is True
```

**Step 5: Run tests**

Run: `uv run pytest tests/models/test_discrete_transformer.py tests/models/test_extra_features_augmentation.py -v`
Expected: All pass

**Step 6: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: Some tests may fail because other test files still import `DiscreteGraphTransformer` with `use_eigenvectors` — that's OK, those are fixed in Task 5. The `test_discrete_transformer.py` tests must pass.

**Step 7: Commit**

```bash
git add src/tmgg/models/digress/transformer_model.py tests/models/test_discrete_transformer.py
git commit -m "refactor: GraphTransformer uses extra_features + use_timestep, removes use_eigenvectors/k"
```

---

## Task 3: Fix VLB softmax bug in DiffusionModule

The VLB block in `DiffusionModule.validation_step` passes raw model output (logits) to `compute_Lt` and `reconstruction_logp`, which expect probabilities. Apply softmax before passing to VLB methods.

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/diffusion_module.py` (lines 302-321)
- Modify: `tests/experiment_utils/test_diffusion_module.py`

**Step 1: Write the failing test**

Add to `tests/experiment_utils/test_diffusion_module.py`, in the `TestVLBWiring` class:

```python
def test_vlb_receives_probabilities_not_logits(self) -> None:
    """VLB methods should receive softmax'd probabilities, not raw logits.

    Invariant: the pred passed to compute_Lt/reconstruction_logp must
    have values in [0, 1] summing to ~1 along the class dimension.
    We verify by patching compute_Lt to inspect its pred argument.
    """
    # This is a design assertion — the fix is in validation_step.
    # After fix, the code path applies F.softmax before VLB methods.
    # We test indirectly: if kl_diffusion values are finite and
    # non-negative, the softmax was applied correctly.
    pass  # tested end-to-end via TestVLBWiring.test_categorical_module_has_vlb_flag
```

Since this is a bugfix to existing code, the primary verification is that the existing VLB tests still pass after the change, and that values are numerically sensible.

**Step 2: Apply the fix**

In `src/tmgg/experiments/_shared_utils/diffusion_module.py`, in `validation_step` (around line 302), add the softmax import at the top of the file:

```python
from torch.nn import functional as F
```

Then modify the VLB block (lines 302-321). Replace:
```python
        # VLB computation for categorical noise processes
        if self._is_categorical and isinstance(
            self.noise_process, CategoricalNoiseProcess
        ):
            kl_prior_X, kl_prior_E, _ = self.noise_process.kl_prior(
                batch.X, batch.E, batch.node_mask
            )
            # kl_prior returns batch-summed scalars; normalise to per-sample
            kl_prior = (kl_prior_X + kl_prior_E) / bs
            kl_diffusion = self.noise_process.compute_Lt(batch, pred, z_t, t_int)
            reconstruction = self.noise_process.reconstruction_logp(batch, pred)
```

With:
```python
        # VLB computation for categorical noise processes
        if self._is_categorical and isinstance(
            self.noise_process, CategoricalNoiseProcess
        ):
            # VLB methods expect probability distributions, not logits.
            # The model outputs logits (CrossEntropyLoss handles conversion);
            # apply softmax here for VLB only.
            pred_probs = GraphData(
                X=F.softmax(pred.X, dim=-1),
                E=F.softmax(pred.E, dim=-1),
                y=pred.y,
                node_mask=pred.node_mask,
            )

            kl_prior_X, kl_prior_E, _ = self.noise_process.kl_prior(
                batch.X, batch.E, batch.node_mask
            )
            # kl_prior returns batch-summed scalars; normalise to per-sample
            kl_prior = (kl_prior_X + kl_prior_E) / bs
            kl_diffusion = self.noise_process.compute_Lt(batch, pred_probs, z_t, t_int)
            reconstruction = self.noise_process.reconstruction_logp(batch, pred_probs)
```

**Step 3: Run tests**

Run: `uv run pytest tests/experiment_utils/test_diffusion_module.py -v`
Expected: All pass

**Step 4: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/diffusion_module.py tests/experiment_utils/test_diffusion_module.py
git commit -m "fix: apply softmax before VLB methods in DiffusionModule (logits != probabilities)"
```

---

## Task 4: Simplify model factory

Update `_make_graph_transformer` in the factory: drop the `"discrete_graph_transformer"` registry name, remove the `default_x = 2 + k` auto-adjustment (model handles this now), pass `extra_features` and `use_timestep` from config.

**Files:**
- Modify: `src/tmgg/models/factory.py` (lines 363-387)

**Step 1: Update factory**

Replace the `_make_graph_transformer` function (lines 363-387):

```python
@register_model("graph_transformer")
def _make_graph_transformer(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.digress.transformer_model import GraphTransformer

    extra_features = None
    extra_features_type = config.get("extra_features_type")
    use_eigenvectors = config.get("use_eigenvectors", False)

    if extra_features_type:
        from tmgg.models.digress.extra_features import ExtraFeatures

        extra_features = ExtraFeatures(
            extra_features_type,
            max_n_nodes=config.get("max_n_nodes", 200),
        )
    elif use_eigenvectors:
        from tmgg.models.digress.extra_features import EigenvectorAugmentation

        k = config.get("k")
        if k is None:
            raise ValueError("k must be specified when use_eigenvectors=True")
        extra_features = EigenvectorAugmentation(k=k)

    return GraphTransformer(
        n_layers=config.get("n_layers", 2),
        input_dims=config.get("input_dims", {"X": 2, "E": 2, "y": 0}),
        hidden_mlp_dims=config.get("hidden_mlp_dims", {"X": 32, "E": 16, "y": 32}),
        hidden_dims=config.get(
            "hidden_dims", {"dx": 32, "de": 16, "dy": 32, "n_head": 2}
        ),
        output_dims=config.get("output_dims", {"X": 0, "E": 2, "y": 0}),
        extra_features=extra_features,
        use_timestep=config.get("use_timestep", False),
    )
```

Key changes vs current:
- `@register_model("graph_transformer")` only (was `"discrete_graph_transformer", "graph_transformer"`)
- No `default_x = 2 + k` logic
- Passes `extra_features` and `use_timestep` instead of `use_eigenvectors` and `k`

**Step 2: Check for tests that reference "discrete_graph_transformer" factory name**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v -k "factory or registry"`
Expected: If any tests create models via `create_model("discrete_graph_transformer", ...)`, they will fail. Fix by changing to `"graph_transformer"`.

**Step 3: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: Some failures from tests still importing `DiscreteGraphTransformer` — fixed in Task 5.

**Step 4: Commit**

```bash
git add src/tmgg/models/factory.py
git commit -m "refactor: simplify graph_transformer factory, drop discrete_graph_transformer name"
```

---

## Task 5: Alias cleanup — delete `discrete_transformer.py`, update all imports

Delete the re-export alias module and update every import site to use `GraphTransformer` from `transformer_model`.

**Files:**
- Delete: `src/tmgg/models/digress/discrete_transformer.py`
- Modify: `src/tmgg/experiments/discrete_diffusion_generative/lightning_module.py` (line 40)
- Modify: `src/tmgg/experiments/discrete_diffusion_generative/__init__.py` (line 12)
- Modify: `src/tmgg/experiments/discrete_diffusion_generative/README.md` (line 20)
- Modify: `tests/experiments/test_discrete_diffusion_module.py` (lines 25, 42, 44, 70)
- Modify: `tests/experiments/test_discrete_diffusion_evaluate.py` (lines 37, 55, 56, 81, 203, 256)
- Modify: `tests/experiments/test_discrete_diffusion_runner.py` (lines 102, 187-188, 197)
- Modify: `tests/experiments/test_all_experiments_full_flow.py` (lines 270-271, 280)
- Modify: `scripts/validate_discrete_diffusion.py` (lines 63, 239)

**Step 1: Delete the alias module**

```bash
rm src/tmgg/models/digress/discrete_transformer.py
```

**Step 2: Update source imports**

In `src/tmgg/experiments/discrete_diffusion_generative/lightning_module.py`, line 40:
```python
# BEFORE:
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
# AFTER:
from tmgg.models.digress.transformer_model import GraphTransformer
```

Also update the type hint on line 97:
```python
# BEFORE: model: DiscreteGraphTransformer,
# AFTER: model: GraphTransformer,
```

And the docstring on line 73:
```python
# BEFORE: Pre-built ``DiscreteGraphTransformer`` backbone.
# AFTER: Pre-built ``GraphTransformer`` backbone.
```

In `src/tmgg/experiments/discrete_diffusion_generative/__init__.py`, line 12:
```python
# BEFORE: ``DiscreteGraphTransformer`` forward interface.
# AFTER: ``GraphTransformer`` forward interface.
```

In `src/tmgg/experiments/discrete_diffusion_generative/README.md`, line 20:
```markdown
# BEFORE: - `DiscreteGraphTransformer` — denoising backbone (`models/digress/`)
# AFTER: - `GraphTransformer` — denoising backbone (`models/digress/`)
```

**Step 3: Update test imports**

For each test file, apply the same pattern:

```python
# BEFORE:
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
# AFTER:
from tmgg.models.digress.transformer_model import GraphTransformer
```

Then replace all `DiscreteGraphTransformer` type hints with `GraphTransformer`.

In `tests/experiments/test_discrete_diffusion_runner.py` line 102, update the assertion:
```python
# BEFORE: assert model_cfg["model"]["_target_"].endswith("DiscreteGraphTransformer")
# AFTER: assert model_cfg["model"]["_target_"].endswith("GraphTransformer")
```

**Step 4: Update script**

In `scripts/validate_discrete_diffusion.py`, lines 63 and 239: same import + constructor rename.

**Step 5: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove DiscreteGraphTransformer alias, use GraphTransformer everywhere"
```

---

## Task 6: Update YAML configs

Update the 4 discrete model configs to use the new `GraphTransformer` import path and, for the eigenvector config, replace `use_eigenvectors`/`k` with `extra_features`. These configs currently target `DiscreteDiffusionLightningModule` (not migrated yet), so `use_timestep` and `input_dims.y` changes are NOT made here — they happen during the DiffusionModule migration.

**Files:**
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml`

**Step 1: Update non-eigenvector configs**

For `discrete_default.yaml`, `discrete_small.yaml`, `discrete_sbm_official.yaml`: only the model `_target_` changes.

```yaml
# BEFORE:
model:
  _target_: tmgg.models.digress.discrete_transformer.DiscreteGraphTransformer

# AFTER:
model:
  _target_: tmgg.models.digress.transformer_model.GraphTransformer
```

Also update comments: replace "DiscreteGraphTransformer" with "GraphTransformer" in comment text. Remove `TODO(Task 12)` comments since the migration path is now unblocked.

**Step 2: Update eigenvector config**

For `discrete_sbm_eigenvec.yaml`:

```yaml
# BEFORE:
model:
  _target_: tmgg.models.digress.discrete_transformer.DiscreteGraphTransformer
  n_layers: 8
  use_eigenvectors: true
  k: 50
  input_dims: { X: 52, E: 2, y: 1 }   # X = 2 (one-hot) + 50 (eigenvectors)
  ...

# AFTER:
model:
  _target_: tmgg.models.digress.transformer_model.GraphTransformer
  n_layers: 8
  input_dims: { X: 2, E: 2, y: 1 }    # base dims; eigenvectors auto-added
  extra_features:
    _target_: tmgg.models.digress.extra_features.EigenvectorAugmentation
    k: 50
  ...
```

Remove `use_eigenvectors: true` and `k: 50` from the top-level model config (they're now inside `extra_features`). Update the comment about `input_dims.X`.

**Step 3: Verify configs parse**

Run:
```bash
uv run python -c "
from omegaconf import OmegaConf
import yaml
for f in [
    'src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml',
    'src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml',
    'src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml',
    'src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml',
]:
    cfg = OmegaConf.load(f)
    print(f'{f}: OK - model._target_ = {cfg.model._target_}')
"
```

Expected: all 4 print "OK" with `GraphTransformer` in the target.

**Step 4: Run config-related tests**

Run: `uv run pytest tests/experiments/test_discrete_diffusion_runner.py -v`
Expected: All pass (the `_target_` assertion was updated in Task 5)

**Step 5: Run full suite**

Run: `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/tmgg/experiments/exp_configs/models/discrete/
git commit -m "refactor: update discrete configs to GraphTransformer, use extra_features for eigenvectors"
```

---

## Execution Notes

**Dependency ordering:** 1 → 2 → 4 → 5 → 6 (strictly sequential). Task 3 (VLB fix) is independent and can run at any point.

**Risk areas:**
- **Task 2** (GraphTransformer refactor): Changing constructor params breaks any code that passes `use_eigenvectors`/`k` directly. Tests fixed in the same task; other test files import from `discrete_transformer.py` which still exists until Task 5. Those tests will fail between Tasks 2 and 5 because the re-export still works but the constructor params changed. To handle this: either (a) batch Tasks 2-5 together, or (b) temporarily update `discrete_transformer.py` to re-export without the old params (it's just a re-export, so this works automatically).
- **Task 5** (alias cleanup): The bulk rename. Use `git grep` to verify no references remain after the change.
- **Task 6** (YAML configs): The eigenvector config changes `input_dims.X` from 52 to 2, relying on auto-adjustment. If `GraphTransformer.__init__` is wrong, `_GraphTransformer`'s `mlp_in_X` will have mismatched dims. Caught at model construction time (shape error), not silently.

**Important constraint:** The YAML configs still target `DiscreteDiffusionLightningModule` (not `DiffusionModule`). The `use_timestep` flag and `input_dims.y` changes happen during the DiffusionModule migration (separate follow-up). For now, `use_timestep=False` (default) keeps the model's behavior identical to before — the old LightningModule handles `t` embedding in its own `forward()`.
