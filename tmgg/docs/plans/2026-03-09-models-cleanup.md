# Models Package Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the reverse dependency in `layers/`, make `EigenEmbedding` private, remove experimental shrinkage wrappers, and relocate diffusion math/sampling/types from `models/digress/` to `tmgg/diffusion/`.

**Architecture:** Four independent refactoring streams that touch different files, executed sequentially to keep the test suite green after each commit. The diffusion split (Task 4) is the largest because it rewires the most import paths, but all changes are mechanical import updates with no behavioral changes.

**Tech Stack:** Python 3.12, PyTorch, pytest, tach (module boundaries)

---

### Task 1: Move `topk_eigen.py` from `spectral_denoisers/` to `layers/`

This fixes the reverse dependency where `layers/eigen_embedding.py` imports from `spectral_denoisers/topk_eigen.py`.

**Files:**
- Move: `src/tmgg/models/spectral_denoisers/topk_eigen.py` -> `src/tmgg/models/layers/topk_eigen.py`
- Modify: `src/tmgg/models/layers/__init__.py`
- Modify: `src/tmgg/models/layers/eigen_embedding.py`
- Modify: `src/tmgg/models/spectral_denoisers/__init__.py`
- Modify: `src/tmgg/models/spectral_denoisers/base_spectral.py`

**Step 1: Move the file**

```bash
git mv src/tmgg/models/spectral_denoisers/topk_eigen.py src/tmgg/models/layers/topk_eigen.py
```

**Step 2: Update `layers/__init__.py`**

Add to imports:
```python
from .topk_eigen import EigenDecompositionError, TopKEigenLayer
```
Add to `__all__`:
```python
"EigenDecompositionError",
"TopKEigenLayer",
```

**Step 3: Update `layers/eigen_embedding.py`**

Change import from:
```python
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer
```
to:
```python
from .topk_eigen import TopKEigenLayer
```

**Step 4: Update `spectral_denoisers/__init__.py`**

Change the topk_eigen import from:
```python
from tmgg.models.spectral_denoisers.topk_eigen import (
    EigenDecompositionError,
    TopKEigenLayer,
)
```
to:
```python
from tmgg.models.layers.topk_eigen import (
    EigenDecompositionError,
    TopKEigenLayer,
)
```

**Step 5: Update `spectral_denoisers/base_spectral.py`**

Change import from:
```python
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer
```
to:
```python
from tmgg.models.layers.topk_eigen import TopKEigenLayer
```

**Step 6: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v -k "eigen or topk or spectral or gnn"
```
Expected: All pass.

**Step 7: Commit**

```bash
git add -A && git commit -m "refactor: move topk_eigen.py from spectral_denoisers/ to layers/

Eliminates the reverse dependency where layers/ imported from
spectral_denoisers/. TopKEigenLayer is a computation primitive that
belongs alongside other layer building blocks."
```

---

### Task 2: Make `EigenEmbedding` private

`EigenEmbedding` is only used internally by `TruncatedEigenEmbedding`. External code should use the truncated wrapper. Tests that reference `EigenEmbedding` directly should import from the file, not the package namespace.

**Files:**
- Modify: `src/tmgg/models/layers/eigen_embedding.py` (rename class)
- Modify: `src/tmgg/models/layers/__init__.py` (remove export)
- Modify: `src/tmgg/models/__init__.py` (remove export)
- Modify: `tests/models/test_gnn.py` (update import)
- Modify: `tests/models/test_gnn_properties.py` (update import)
- Modify: `tests/test_audit_fixes.py` (update import)

**Step 1: Rename class in `eigen_embedding.py`**

Rename `EigenEmbedding` -> `_EigenEmbedding` throughout the file. Update the docstring module header and `TruncatedEigenEmbedding`'s docstring references accordingly.

In `eigen_embedding.py`:
- `class EigenEmbedding(nn.Module):` -> `class _EigenEmbedding(nn.Module):`
- `self._inner = EigenEmbedding(...)` -> `self._inner = _EigenEmbedding(...)` in TruncatedEigenEmbedding
- Update docstrings that reference `EigenEmbedding` to `_EigenEmbedding`

**Step 2: Remove from `layers/__init__.py`**

Remove `EigenEmbedding` from the import line and from `__all__`. Keep `TruncatedEigenEmbedding`.

Change:
```python
from .eigen_embedding import EigenEmbedding, TruncatedEigenEmbedding
```
to:
```python
from .eigen_embedding import TruncatedEigenEmbedding
```

Remove `"EigenEmbedding"` from `__all__`.

**Step 3: Remove from `models/__init__.py`**

Remove `EigenEmbedding` from the layers import and from `__all__`.

Change:
```python
from .layers import (
    EigenEmbedding,
    GraphConvolutionLayer,
    ...
)
```
to:
```python
from .layers import (
    GraphConvolutionLayer,
    ...
)
```

Remove `"EigenEmbedding"` from `__all__`.

**Step 4: Update test imports**

In `tests/models/test_gnn.py`, `tests/models/test_gnn_properties.py`, `tests/test_audit_fixes.py`:

Change imports from:
```python
from tmgg.models.layers import EigenEmbedding
# or
from tmgg.models.layers.eigen_embedding import EigenEmbedding
# or
from tmgg.models import EigenEmbedding
```
to:
```python
from tmgg.models.layers.eigen_embedding import _EigenEmbedding as EigenEmbedding
```

This preserves the test variable name while importing the now-private class directly. Tests validating EigenEmbedding's behavior are still valid, they're just reaching into an internal.

**Step 5: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v -k "eigen or gnn or audit"
```
Expected: All pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: make EigenEmbedding private (_EigenEmbedding)

Only TruncatedEigenEmbedding is used by external code. The base class
is now an implementation detail, removed from package-level exports."
```

---

### Task 3: Remove shrinkage wrappers

Shrinkage wrappers are experimental, never used in sweeps, and add dead weight.

**Files:**
- Delete: `src/tmgg/models/spectral_denoisers/shrinkage_wrapper.py`
- Delete: `tests/models/test_shrinkage_wrapper.py`
- Modify: `src/tmgg/models/spectral_denoisers/__init__.py`
- Modify: `src/tmgg/models/factory.py`
- Modify: `tests/models/test_factory_registry.py`
- Modify: `tests/test_training_integration.py`

**Step 1: Delete the source file**

```bash
git rm src/tmgg/models/spectral_denoisers/shrinkage_wrapper.py
```

**Step 2: Delete the test file**

```bash
git rm tests/models/test_shrinkage_wrapper.py
```

**Step 3: Update `spectral_denoisers/__init__.py`**

Remove the entire shrinkage import block:
```python
from tmgg.models.spectral_denoisers.shrinkage_wrapper import (
    RelaxedShrinkageWrapper,
    ShrinkageSVDLayer,
    ShrinkageWrapper,
    StrictShrinkageWrapper,
)
```

Remove from `__all__`:
```python
"ShrinkageWrapper",
"StrictShrinkageWrapper",
"RelaxedShrinkageWrapper",
"ShrinkageSVDLayer",
```

Remove the "Shrinkage Wrappers (Experimental)" section from the module docstring.

**Step 4: Remove factory registrations from `factory.py`**

Delete the two factory functions and their `@register_model` decorators:
- `_make_strict_shrinkage` (registered as `"self_attention_strict_shrinkage"`)
- `_make_relaxed_shrinkage` (registered as `"self_attention_relaxed_shrinkage"`)

**Step 5: Update `tests/models/test_factory_registry.py`**

Remove `"self_attention_strict_shrinkage"` and `"self_attention_relaxed_shrinkage"` from the expected model types list/parametrize.

**Step 6: Update `tests/test_training_integration.py`**

Remove:
- The shrinkage entries from parametrize lists (lines ~104-109)
- The entire `TestShrinkageModels` class (starting ~line 292)
- References to shrinkage in the module docstring (line ~5)

**Step 7: Run full test suite**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```
Expected: All pass.

**Step 8: Commit**

```bash
git add -A && git commit -m "refactor: remove experimental shrinkage wrappers

StrictShrinkageWrapper and RelaxedShrinkageWrapper were never used in
experiment sweeps. Removes the module, factory registrations, and all
associated tests."
```

---

### Task 4: Move diffusion utilities from `models/digress/` to `tmgg/diffusion/`

Three files in `models/digress/` are diffusion primitives, not neural network models: `graph_types.py`, `diffusion_math.py`, `diffusion_sampling.py`. They belong in `tmgg/diffusion/`.

One complication: `transformer_model.py` (which stays in `models/digress/`) imports `assert_correctly_masked` from `diffusion_sampling`. This 3-line function is a mask validation utility. Moving it to `tmgg/diffusion/` would create a circular dependency (`models -> diffusion -> models`). Instead, inline it in `transformer_model.py`.

**Files:**
- Move: `src/tmgg/models/digress/graph_types.py` -> `src/tmgg/diffusion/graph_types.py`
- Move: `src/tmgg/models/digress/diffusion_math.py` -> `src/tmgg/diffusion/diffusion_math.py`
- Move: `src/tmgg/models/digress/diffusion_sampling.py` -> `src/tmgg/diffusion/diffusion_sampling.py`
- Modify: `src/tmgg/diffusion/__init__.py` (add exports)
- Modify: `src/tmgg/diffusion/diffusion_sampling.py` (fix relative import after move)
- Modify: `src/tmgg/diffusion/protocols.py` (update import)
- Modify: `src/tmgg/diffusion/transitions.py` (update import)
- Modify: `src/tmgg/diffusion/sampler.py` (update import)
- Modify: `src/tmgg/diffusion/noise_process.py` (update import)
- Modify: `src/tmgg/diffusion/schedule.py` (update import)
- Modify: `src/tmgg/experiments/_shared_utils/lightning_modules/train_loss_discrete.py` (update import)
- Modify: `src/tmgg/models/digress/transformer_model.py` (inline assert_correctly_masked, remove import)
- Modify: `tests/models/test_graph_types.py` (update import)
- Modify: `tests/test_beta_schedule_edge_classes.py` (update import)

**Step 1: Move the three files**

```bash
git mv src/tmgg/models/digress/graph_types.py src/tmgg/diffusion/graph_types.py
git mv src/tmgg/models/digress/diffusion_math.py src/tmgg/diffusion/diffusion_math.py
git mv src/tmgg/models/digress/diffusion_sampling.py src/tmgg/diffusion/diffusion_sampling.py
```

**Step 2: Fix relative import in `diffusion_sampling.py`**

The file uses `from .graph_types import LimitDistribution, TransitionMatrices` — this still works since both files are now in `tmgg/diffusion/`. No change needed.

**Step 3: Update `diffusion/__init__.py`**

Add exports for the relocated types:
```python
from .graph_types import LimitDistribution, TransitionMatrices
```

Add to `__all__`:
```python
"LimitDistribution",
"TransitionMatrices",
```

**Step 4: Update `diffusion/protocols.py`**

Change:
```python
from tmgg.models.digress.graph_types import LimitDistribution, TransitionMatrices
```
to:
```python
from .graph_types import LimitDistribution, TransitionMatrices
```

**Step 5: Update `diffusion/transitions.py`**

Change:
```python
from tmgg.models.digress.graph_types import LimitDistribution, TransitionMatrices
```
to:
```python
from .graph_types import LimitDistribution, TransitionMatrices
```

**Step 6: Update `diffusion/noise_process.py`**

Change:
```python
from tmgg.models.digress.diffusion_math import sum_except_batch
from tmgg.models.digress.diffusion_sampling import (
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    sample_discrete_features,
)
```
to:
```python
from .diffusion_math import sum_except_batch
from .diffusion_sampling import (
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    sample_discrete_features,
)
```

**Step 7: Update `diffusion/sampler.py`**

Change:
```python
from tmgg.models.digress.diffusion_sampling import (
    compute_batched_over0_posterior_distribution,
    sample_discrete_feature_noise,
    sample_discrete_features,
)
```
to:
```python
from .diffusion_sampling import (
    compute_batched_over0_posterior_distribution,
    sample_discrete_feature_noise,
    sample_discrete_features,
)
```

**Step 8: Update `diffusion/schedule.py`**

Change:
```python
from tmgg.models.digress.diffusion_math import (
    cosine_beta_schedule_discrete,
    custom_beta_schedule_discrete,
)
```
to:
```python
from .diffusion_math import (
    cosine_beta_schedule_discrete,
    custom_beta_schedule_discrete,
)
```

**Step 9: Update `experiments/_shared_utils/lightning_modules/train_loss_discrete.py`**

Change:
```python
from tmgg.models.digress.diffusion_sampling import mask_distributions
```
to:
```python
from tmgg.diffusion.diffusion_sampling import mask_distributions
```

**Step 10: Update `models/digress/transformer_model.py`**

Remove the import:
```python
from .diffusion_sampling import assert_correctly_masked
```

Add the inlined function near the top of the file (after imports):
```python
def _assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    """Verify that masked positions are near-zero."""
    max_val = (variable * (1 - node_mask.long())).abs().max().item()
    if not max_val < 1e-4:
        raise AssertionError("Variables not masked properly.")
```

Replace the 4 call sites of `assert_correctly_masked(...)` with `_assert_correctly_masked(...)`.

**Step 11: Update `tests/models/test_graph_types.py`**

Change:
```python
from tmgg.models.digress.graph_types import (
    LimitDistribution,
    TransitionMatrices,
)
```
to:
```python
from tmgg.diffusion.graph_types import (
    LimitDistribution,
    TransitionMatrices,
)
```

**Step 12: Update `tests/test_beta_schedule_edge_classes.py`**

Change all 3 occurrences of:
```python
from tmgg.models.digress.diffusion_math import custom_beta_schedule_discrete
```
to:
```python
from tmgg.diffusion.diffusion_math import custom_beta_schedule_discrete
```

**Step 13: Update `tach.toml`**

After the move, `tmgg.diffusion` no longer imports from `tmgg.models` for these utilities (they're now in-module). However `diffusion/sampler.py` still imports `GraphModel` from `tmgg.models.base`, so the `depends_on` entry stays. No tach change needed.

**Step 14: Run full test suite**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```
Expected: All pass.

**Step 15: Commit**

```bash
git add -A && git commit -m "refactor: move diffusion math/sampling/types from models/digress/ to tmgg/diffusion/

graph_types.py, diffusion_math.py, and diffusion_sampling.py are
diffusion pipeline primitives, not neural network models. Moving them
to tmgg.diffusion/ eliminates a false dependency where the diffusion
module appeared to depend on models/ for shared math utilities.

assert_correctly_masked (3-line mask validator used only by
transformer_model.py) is inlined to avoid a circular import."
```

---

### Post-completion: Run full suite and verify tach boundaries

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
uv run tach check
```
