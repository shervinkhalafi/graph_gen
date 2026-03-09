# GraphTransformer Self-Containment + Alias Cleanup

## Problem

Three responsibilities are split between the model (`GraphTransformer`) and the
training loop (`DiffusionModule` / old `DiscreteDiffusionLightningModule`):

1. **Feature augmentation** -- structural features (cycle counts, eigenvalues)
   computed from the noisy graph and concatenated with model input.
2. **Timestep embedding** -- normalised diffusion timestep concatenated with
   global features `y`.
3. **Softmax post-processing** -- converting model logits to probabilities
   for VLB and sampling.

The old `DiscreteDiffusionLightningModule.forward()` handled all three. The new
`DiffusionModule` handles none of them, creating a gap that blocked discrete
diffusion config migration. Rather than adding hooks to `DiffusionModule`
(a hack), we place each responsibility where it belongs:

| Responsibility | Belongs in | Rationale |
|---|---|---|
| Feature augmentation | Model | Same pattern as existing `use_eigenvectors` in GraphTransformer |
| Timestep embedding | Model | Model already accepts `t` param but ignores it |
| Softmax for VLB | Training loop (locally) | Standard: model outputs logits, consumer converts as needed |

## Design

### A. Unified feature augmentation

`GraphTransformer` currently has a special-case `use_eigenvectors + TopKEigenLayer`
path for eigenvector augmentation. `ExtraFeatures` from `extra_features.py` does
something more general (cycle counts + eigenvalues). Both follow the same pattern:
compute structural features from the input graph, concatenate with model input.

**Replace** `use_eigenvectors`/`k`/`eigen_layer` with a single `extra_features`
parameter. Three callables satisfy the interface:

| Callable | Computes | Dim adjustment | Used by |
|---|---|---|---|
| `DummyExtraFeatures` | Nothing | identity | SBM experiments (no augmentation) |
| `ExtraFeatures("all", max_n=20)` | Cycles + eigenvalues | X+6, E+0, y+11 | Molecular / published DiGress |
| `EigenvectorAugmentation(k=8)` | Top-k eigenvectors | X+k, E+0, y+0 | Spectral augmentation |

Each callable has two methods:
- `adjust_dims(input_dims) -> input_dims` -- takes pre-augmentation dims,
  returns adjusted dims. Encapsulates the dim arithmetic so the composing
  class never does manual `+=`.
- `__call__(X, E, y, node_mask) -> (extra_X, extra_E, extra_y)` -- computes
  extra features at runtime.

`EigenvectorAugmentation` is a new ~15-line class that moves the eigenvector
code currently in `GraphTransformer.forward()` into this interface.

### B. Timestep embedding

`GraphTransformer.forward()` accepts `t` but ignores it (docstring: "Currently
unused; accepted for interface compatibility"). The old code concatenates
normalised `t` as a 1D feature with `y`.

When `use_timestep=True`, `forward()` concatenates `t.unsqueeze(-1)` with `y`
before passing to `_GraphTransformer`. The `input_dims["y"]` is auto-adjusted
by +1 at construction.

### C. `input_dims` becomes base-only

Configs specify only raw feature dimensions. The model auto-adjusts by
delegating to `extra_features.adjust_dims()` and adding 1 for timestep:

```python
# GraphTransformer.__init__:
adjusted = dict(input_dims)
if extra_features is not None:
    adjusted = extra_features.adjust_dims(adjusted)
if use_timestep:
    adjusted["y"] += 1
self.transformer = _GraphTransformer(input_dims=adjusted, ...)
```

Config example (before/after):
```yaml
# Before (manual dim adjustment):
input_dims: { X: 10, E: 2, y: 0 }
use_eigenvectors: true
k: 8

# After (base dims, model auto-adjusts):
input_dims: { X: 2, E: 2, y: 0 }
extra_features:
  _target_: tmgg.models.digress.extra_features.EigenvectorAugmentation
  k: 8
```

### D. Softmax stays out of the model

The model outputs logits, always. `CrossEntropyLoss` expects logits. For VLB
computation (which expects probabilities), `DiffusionModule.validation_step`
applies softmax locally in the categorical block:

```python
pred = self.model(z_t, t=t_norm)          # logits
loss = self._compute_loss(pred, batch)    # CrossEntropyLoss handles it

if self._is_categorical:
    pred_probs = GraphData(
        X=F.softmax(pred.X, dim=-1),
        E=F.softmax(pred.E, dim=-1),
        y=pred.y, node_mask=pred.node_mask,
    )
    kl_diffusion = self.noise_process.compute_Lt(batch, pred_probs, z_t, t_int)
    reconstruction = self.noise_process.reconstruction_logp(batch, pred_probs)
```

This also fixes an existing bug: the current VLB block passes raw logits to
methods that expect probabilities.

### E. Alias cleanup

- Delete `src/tmgg/models/digress/discrete_transformer.py` (re-export alias)
- Factory: `@register_model("graph_transformer")` only (drop `"discrete_graph_transformer"`)
- All imports use `GraphTransformer` from `transformer_model`
- All YAML `_target_` use `tmgg.models.digress.transformer_model.GraphTransformer`

## Files touched

| File | Change |
|---|---|
| `models/digress/transformer_model.py` | Replace `use_eigenvectors`/`k`/`eigen_layer` with `extra_features`/`use_timestep`; auto-adjust `input_dims` via `adjust_dims()` |
| `models/digress/extra_features.py` | Add `adjust_dims()` to `ExtraFeatures`/`DummyExtraFeatures`; add `EigenvectorAugmentation` class |
| `models/digress/discrete_transformer.py` | **Delete** |
| `models/factory.py` | Simplify `_make_graph_transformer`; drop `"discrete_graph_transformer"`; remove dim auto-adjustment |
| `experiments/_shared_utils/diffusion_module.py` | Fix VLB block: softmax before VLB methods |
| 4 discrete YAML model configs | Update `_target_`; simplify `input_dims` to base dims |
| `base_config_discrete_diffusion_generative.yaml` | Remove TODO block about migration blockers |
| 5 test files + 1 script | `DiscreteGraphTransformer` -> `GraphTransformer` |

## What does NOT change

- `DiffusionModule` loop structure (no hooks, no pre/post-processing)
- `_GraphTransformer` internals (receives pre-processed tensors as before)
- `ExtraFeatures`/`DummyExtraFeatures` `__call__` signature
- `SingleStepDenoisingModule` (no extra features, no timestep, unaffected)
- `TopKEigenLayer` itself (still used by `_GraphTransformer` for spectral projections)
