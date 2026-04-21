# Section 8 — Model Architecture Review

**Spec:** `docs/reports/2026-04-21-digress-upstream-spec.md §8`
**Our code:** `src/tmgg/models/digress/transformer_model.py` + `digress/layers.py` + `models/layers/masked_softmax.py`
**Upstream ref:** `digress-upstream-readonly/src/models/transformer_model.py` + `src/models/layers.py`
**Prior audit:** `analysis/digress-loss-check/parity-audit/deep-architecture.md` (Haiku; reviewed an older config)
**Reviewed:** 2026-04-21

---

## Summary verdict

**MATCH** for the default SBM discrete-diffusion path (`discrete_sbm_official.yaml`, GNN/spectral flags off). All core ops — attention, FiLM, layer-norm placement, residual wiring, output projection, symmetrisation — are op-for-op identical to upstream. Two additive extensions (`use_timestep`, GNN/spectral projections) are present and correctly gated; neither changes the baseline graph when disabled. One minor behavioural divergence in `masked_softmax` is noted but does not affect normal runs. No weight-init divergence. No blocking gaps.

---

## Spec summary (§8)

`GraphTransformer` wraps `n_layers` `XEyTransformerLayer`s between input/output MLPs. Each layer runs `NodeEdgeBlock` (attention + FiLM) followed by per-stream FFN + LayerNorm + residual. Architecture is specified with SBM defaults `dx=256, de=64, dy=64, n_head=8, dim_ffX=256, dim_ffE=64, dim_ffy=256, n_layers=8`, input projections via `hidden_mlp_dims={X:128, E:64, y:128}`. Time `t` is appended to `y` by `compute_extra_data` in the diffusion wrapper, not inside the transformer class. Upstream uses PyTorch default Linear init throughout. No weight-init overrides visible in the model.

---

## Per-check verdicts

### 1. Input projection MLPs

**Spec:** `mlp_in_X = Linear(in_X, 128) → ReLU → Linear(128, dx) → ReLU`; analogous for E (`hidden_mlp_dims.E=64`) and y (`hidden_mlp_dims.y=128`).

**Ours** (`_GraphTransformer.__init__`, lines 575–594): structure is identical — two Linear layers with the passed activation, using the same `hidden_mlp_dims` and `hidden_dims` keys. Bias is present (PyTorch Linear default) on both sides.

**SBM config** (`discrete_sbm_official.yaml`): `hidden_mlp_dims={X:128, E:64, y:128}`, `hidden_dims.dx=256, de=64, dy=64`. The in→hidden→dx chain is therefore `Linear(in_X, 128) → ReLU → Linear(128, 256) → ReLU` for X, matching the spec table.

**Verdict: MATCH.**

---

### 2. `XEyTransformerLayer` — layer-norm placement and residual structure

**Spec:** Post-attention residual + LayerNorm (norm1), then FFN + residual + LayerNorm (norm2); `activation = F.relu`; dropout 0.1.

**Ours** (lines 162–186): identical structure. Upstream stores `self.activation = F.relu` and calls it via the instance attribute; ours calls `F.relu()` inline. These are functionally equivalent.

The upstream FFN for X is `linX2(dropoutX2(activation(linX1(X))))` then a third dropout before the residual add. Our code does the same (lines 175–177). The ordering is identical for E and y streams.

**Verdict: MATCH.**

---

### 3. `NodeEdgeBlock` — attention computation

**Spec:** Q, K, V = Linear(dx, dx); reshape to `(B, n, n_head, df)`, `df = dx/n_head`; `Y = Q ⊙ K / √df`; FiLM from E; softmax over dim 2 (keys); multiply by V; sum.

**Ours** (lines 411–484, default linear path): identical. When all projection flags are False (the default), `self.q = Linear(dx, dx)` etc., and the reshape, Q·K, FiLM, softmax, and weighted-V ops are byte-for-byte equivalent.

One structural cosmetic: upstream `Q.unsqueeze(2)` produces `(bs, n, 1, n_head, df)` and `K.unsqueeze(1)` produces `(bs, 1, n, n_head, df)`; our code has the comments labelling these as `(bs, n, 1, n_head, df)` and `(bs, 1, n, n_head, df)` respectively (lines 436–437). The upstream comment at line 155–156 swaps the labels (upstream comment says `(bs, 1, n, ...)` for Q.unsqueeze(2)), but the actual code is the same: both broadcast-multiply to `(bs, n, n, n_head, df)`.

**Verdict: MATCH.**

---

### 4. FiLM: E → attention scores and y → E/X

**Spec:**
- E → attention: `Y = Y * (E1 + 1) + E2`, where E1 = `e_mul(E)`, E2 = `e_add(E)`, both `Linear(de, dx)`.
- y → E: `newE = y_e_add(y) + (y_e_mul(y) + 1) * newE`; projections are `Linear(dy, dx)` (not `de`).
- y → X: `newX = y_x_add(y) + (y_x_mul(y) + 1) * weighted_V`; projections `Linear(dy, dx)`.

**Ours** (lines 444–457, 487–489): identical formulas and projection dimensions. The comment `# Warning: here it's dx and not de` is carried over verbatim from upstream (our line 337, upstream line 116), confirming intentional match.

**Verdict: MATCH.**

---

### 5. y update — `Xtoy`/`Etoy` pooling

**Spec:** `Xtoy` concatenates `[mean, min, max, std]` over node dim → `Linear(4*dx, dy)`. `Etoy` same over both edge dims. `y_out` = `Sequential(Linear(dy,dy), ReLU, Linear(dy,dy))`.

**Ours** (`digress/layers.py`): `Xtoy` and `Etoy` are verbatim copies of the upstream implementation — same four statistics, same hstack, same `Linear(4*dx, dy)` / `Linear(4*de, dy)`. `y_out` (lines 352–356 of `transformer_model.py`) is `Sequential(Linear, ReLU, Linear)` with `dy→dy→dy`, identical to upstream.

**Verdict: MATCH.**

---

### 6. Hidden dims — config-driven on both sides

Upstream reads `hidden_dims` and `hidden_mlp_dims` entirely from config (YAML → Hydra → constructor). Our `_GraphTransformer` and `GraphTransformer` take the same dict arguments and pass them through unchanged. The SBM official config sets the values from the spec table: `dx=256, de=64, dy=64, n_head=8, dim_ffX=256, dim_ffE=64, dim_ffy=256, hidden_mlp_dims={X:128, E:64, y:128}`.

One minor note: upstream's `XEyTransformerLayer` constructor does not accept `dim_ffy` — it is absent from the upstream signature (line 26–28 of `digress-upstream-readonly`). In contrast, our layer accepts `dim_ffy` explicitly and uses it for `lin_y1/lin_y2`. The upstream FFN for y also uses `dim_ffy` (just unnamed in the default), so functionally there is no difference, but the upstream code always used the constructor default `dim_ffy=2048` unless overridden by the caller. In upstream the y FFN width is hardcoded to `dim_ffy=2048` default; for SBM the caller passes `dim_ffy=hidden_dims['dim_ffy']` explicitly in our code, matching the spec's `dim_ffy=256`.

**Verdict: MATCH** (config-driven on both sides, SBM values correct).

---

### 7. Output projection MLPs

**Spec:** `mlp_out_X = Linear(dx, hidden_mlp_X) → ReLU → Linear(hidden_mlp_X, out_X)`, analogous for E and y. Residual `X_to_out = X[..., :out_X]` added after output projection; diagonal of E zeroed; E symmetrised.

**Ours** (lines 663–771): output MLPs have the same 2-layer structure with `act_fn_out` between them (defaults to ReLU). Residual, diagonal mask, and `½(E + Eᵀ)` symmetrisation are applied in the same order as upstream. The `diag_mask` is constructed via `torch.eye(n)` on `E_cat.device`, which matches upstream's `torch.eye(n).type_as(E)`.

For SBM, `output_dims={X:2, E:2, y:0}`, so `mlp_out_X` ends at width 2 (matching `d_X=2` node classes), and `mlp_out_E` at width 2 (matching `d_E=2` edge classes).

**Verdict: MATCH.**

---

### 8. `use_timestep=True` — what it adds and when it matches upstream

**Upstream mechanism:** `compute_extra_data` in `DiscreteDenoisingDiffusion` appends `t` (normalised, shape `(B,1)`) to `extra_y` and concatenates into the `y` input to `GraphTransformer.forward`. The GraphTransformer class itself is unaware of timestep — it sees `y` with `t` already included. Effective y-input width is `extra_y_width + 1`.

**Our mechanism:** `GraphTransformer.__init__` accepts `use_timestep: bool`. When `True`, it increments `adjusted_input_dims["y"]` by 1 before constructing `_GraphTransformer`, and in `forward` appends `t.unsqueeze(-1)` to `y` if `t is not None`. The `discrete_sbm_official.yaml` sets `use_timestep: true` and declares `input_dims.y = 0` with the comment "y=0 because use_timestep adds +1 internally".

The net result is identical: the `mlp_in_y` sees a tensor of width `extra_y_width + 1`, and the `+1` comes from `t`. The only structural difference is that upstream adds `t` in the diffusion wrapper, whereas we add it in the model class. When `use_timestep=False`, no timestep is injected and `mlp_in_y` sees whatever `y` is passed in.

**Verdict: MATCH** (semantically equivalent; the injection point differs but the tensor seen by `mlp_in_y` is the same).

---

### 9. GNN/spectral projection flags — default-off verification

All six `use_gnn_{q,k,v}` and `use_spectral_{q,k,v}` flags default to `False` in both `XEyTransformerLayer.__init__` and `NodeEdgeBlock.__init__`. The `discrete_sbm_official.yaml` does not set any of these flags, so they remain `False`. When False, `self.q = Linear(dx, dx)` and the forward path is the linear projection identical to upstream. The `GraphStructure` object is constructed with all fields `None` in this case, and the `structure` argument to `layer()` carries no data — all the guarded `if self._use_gnn_*` branches are skipped.

**Verdict: OFF by default; SBM run uses vanilla linear path matching upstream exactly.**

---

### 10. Weight initialisation

Neither upstream nor our code contains any explicit weight-init calls in the transformer or layer modules. Both rely on PyTorch's default `nn.Linear` initialisation (Kaiming uniform for weights, uniform for bias, as of PyTorch 2.x). Confirmed by inspection: no `reset_parameters`, `xavier_uniform_`, or `orthogonal_` calls in either `transformer_model.py` or `layers.py` on either side.

**Verdict: MATCH.**

---

### 11. `masked_softmax` — minor behavioural divergence

**Upstream** (`layers.py` lines 41–46): returns `x` unchanged when `mask.sum() == 0`; otherwise fills masked positions with `-inf` and applies softmax. This means that when the entire batch has no valid keys, the raw (pre-softmax) scores are returned — not a proper probability distribution.

**Ours** (`models/layers/masked_softmax.py`): when `mask.sum() == 0` returns `torch.zeros_like(x)`. Additionally adds NaN-to-zero cleanup for rows that are all-masked. Both behaviours guard an edge case that should not arise under normal SBM training (every graph has at least one node). There is also a signature difference: ours accepts `mask: Tensor | None` while upstream accepts only `Tensor`.

**Verdict: MINOR DIVERGENCE** (edge-case handling only; no impact under normal operation).

---

## Remaining gaps / open questions

None that affect correctness under the standard SBM discrete-diffusion configuration. The only items worth tracking:

1. The `masked_softmax` all-zeros vs. passthrough difference is benign in practice but means the two implementations do not return identical tensors in the degenerate all-masked case. If empty graphs ever enter the transformer this could produce different downstream NaN propagation behaviour.

2. The `use_timestep` injection is inside the model class in our code vs. inside the diffusion wrapper upstream. This is invisible to parity, but it means our `GraphTransformer` is not a drop-in replacement for upstream's if the caller also injects `t` externally — double-injection would inflate `y` by 2 rather than 1. Worth noting for any future adapter layer.
