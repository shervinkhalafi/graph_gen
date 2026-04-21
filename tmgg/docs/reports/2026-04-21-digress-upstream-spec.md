# DiGress Upstream Reference Specification

Scope: the `cvignac/DiGress` checkout at
`/home/igork/900_personal/900.000_research/900.003_shervin_graphgen/digress-upstream-readonly/`,
HEAD `780242b`. All file references below are relative to that
directory. This document describes the discrete variant
(`src/diffusion_model_discrete.py`) used for SBM / planar / QM9-no-H /
MOSES / GuacaMol. The continuous variant in `src/diffusion_model.py` is
out of scope.

---

## 1. Problem statement and graph representation

A graph with node set of size $n$ is encoded as a triple
$G = (X, E, y)$ where

- $X \in \{0, 1\}^{n \times d_X}$ is a one-hot node-type tensor;
- $E \in \{0, 1\}^{n \times n \times d_E}$ is a one-hot edge-type
  tensor, with class index $0$ reserved for "no edge" and the diagonal
  set to the all-zero vector;
- $y \in \mathbb{R}^{d_y}$ is a graph-level feature (empty for the
  non-molecular datasets).

A mini-batch is padded to the maximum number of nodes $n_\max$ in the
batch and accompanied by a boolean `node_mask` of shape `(B, n_max)`.

### Code

`src/utils.py:53-75` converts a PyG sparse batch into dense tensors:

```python
def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr,
                     max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask
```

`encode_no_edge` at `src/utils.py:65-75` forces every entry whose edge
one-hot sums to zero (absent edge or padded position) to the class-0
one-hot and then zeroes the diagonal:

```python
no_edge = torch.sum(E, dim=3) == 0
first_elt = E[:, :, :, 0]
first_elt[no_edge] = 1
E[:, :, :, 0] = first_elt
diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
E[diag] = 0
```

The `PlaceHolder` container at `src/utils.py:103-131` is a three-slot
struct `(X, E, y)` with a `.mask(node_mask, collapse=False)` method that
either masks out padded rows by multiplication or, with
`collapse=True`, argmaxes to integer labels and writes $-1$ into padded
positions.

### Shapes

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `X` | `(B, n_max, d_X)` | float (one-hot) |
| `E` | `(B, n_max, n_max, d_E)` | float (one-hot, symmetric) |
| `y` | `(B, d_y)` | float |
| `node_mask` | `(B, n_max)` | bool |

For the SPECTRE datasets (SBM, planar, comm20) there is a single node
class and two edge classes, i.e. $d_X = 1$, $d_E = 2$, $d_y = 0$
(see `src/datasets/spectre_dataset.py:85-89` and
`SpectreDatasetInfos.node_types = torch.tensor([1])` at
`src/datasets/spectre_dataset.py:133`).

### Hyperparameters

Only dataset-derived; no configurable knobs here. `max_n_nodes` is
derived by `AbstractDatasetInfos.complete_infos` at
`src/datasets/abstract_dataset.py:94-100` as the largest node count
seen in train+val.

---

## 2. Discrete diffusion process — formal definition

The forward chain on a single categorical variable $z \in
\{1, \ldots, K\}$ with one-hot encoding $\mathbf{z}$ is a Markov chain
with transition kernel

$$
q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \operatorname{Cat}(\mathbf{z}_t; \mathbf{z}_{t-1} Q_t),
$$

and cumulative kernel

$$
q(\mathbf{z}_t \mid \mathbf{z}_0) = \operatorname{Cat}(\mathbf{z}_t; \mathbf{z}_0 \bar{Q}_t), \qquad \bar{Q}_t = Q_1 Q_2 \cdots Q_t.
$$

DiGress runs one independent chain per node entry of $X$ and one per
off-diagonal entry of $E$; $y$ is carried unchanged. Time is indexed
such that $t = 0$ is the clean data and $t = T$ is the prior
(`self.T = cfg.model.diffusion_steps`, default 500 / 1000 for SBM per
`configs/experiment/sbm.yaml:21`). The continuous normalised time is
$t/T \in [0, 1]$.

Paper reference: Vignac et al., ICLR 2023, §3 (equations for $q(z_t |
z_{t-1})$ and $q(z_t | x_0)$).

---

## 3. Noise schedule

The cosine schedule of Nichol & Dhariwal is used. For discrete DiGress
the closed form implemented in `cosine_beta_schedule_discrete`
(`src/diffusion/diffusion_utils.py:65-74`) is

$$
\bar{\alpha}_t = \frac{\cos^2\!\Big( \tfrac{\pi}{2} \cdot \tfrac{t/T_{\text{ext}} + s}{1 + s} \Big)}{\cos^2\!\Big( \tfrac{\pi}{2} \cdot \tfrac{s}{1+s} \Big)}, \qquad \beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \qquad s = 0.008,
$$

with $T_{\text{ext}} = T + 2$. The code

```python
steps = timesteps + 2
x = np.linspace(0, steps, steps)
alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
betas = 1 - alphas
return betas.squeeze()
```

returns an array of length $T + 1$ (after `linspace(0, T+2, T+2)` and
the length-one shift).

`PredefinedNoiseScheduleDiscrete` (`src/diffusion/noise_schedule.py:44-79`)
precomputes:

```python
self.register_buffer('betas', torch.from_numpy(betas).float())
self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
log_alpha = torch.log(self.alphas)
log_alpha_bar = torch.cumsum(log_alpha, dim=0)
self.alphas_bar = torch.exp(log_alpha_bar)
```

so $\bar{\alpha}_t = \prod_{\tau \le t}(1 - \beta_\tau)$ is available by
table lookup. Both `forward(t_normalized)` (returns $\beta_t$) and
`get_alpha_bar(t_normalized)` round the normalised time to the nearest
integer index via `torch.round(t_normalized * self.timesteps)`.

A `custom_beta_schedule_discrete` is also defined
(`src/diffusion/diffusion_utils.py:77-97`) that floors $\beta_t$ during
the first steps so that roughly 1.2 edge flips happen per graph; it is
not selected by any shipped YAML.

### Hyperparameters

| Name | Default | Config |
|------|---------|--------|
| `diffusion_noise_schedule` | `'cosine'` | `configs/model/discrete.yaml:6` |
| `diffusion_steps` ($T$) | 500 (default), 1000 (sbm, planar) | `configs/model/discrete.yaml:5`, `configs/experiment/sbm.yaml:21`, `configs/experiment/planar.yaml:21` |

---

## 4. Limit distribution

Three limit variants are defined but only the first two ship:

- `uniform`: $\pi_X = \mathbf{1}/d_X$, $\pi_E = \mathbf{1}/d_E$.
- `marginal`: $\pi_X$ is the empirical node-type distribution,
  $\pi_E$ is the empirical edge-type distribution.
- `absorbing`: a fixed absorbing class
  (`AbsorbingStateTransition`, `src/diffusion/noise_schedule.py:190-223`);
  instantiated by no shipped config.

For the marginal variant, $\pi$ is estimated in
`AbstractDataModule.node_types` and `edge_counts`
(`src/datasets/abstract_dataset.py:34-72`). `edge_counts` counts ordered
pairs inside each graph:

```python
all_pairs = 0
for count in counts:
    all_pairs += count * (count - 1)
num_edges = data.edge_index.shape[1]
num_non_edges = all_pairs - num_edges
...
d[0] += num_non_edges
d[1:] += edge_types[1:]
```

and normalises, so the empirical $\pi_E$ counts each undirected edge
twice (because PyG stores both directions in `edge_index`) and
likewise for non-edges. For the SPECTRE datasets
`SpectreDatasetInfos.node_types = torch.tensor([1])` so only the edge
marginal is data-dependent.

The model binds $\pi$ in
`DiscreteDenoisingDiffusion.__init__` at
`src/diffusion_model_discrete.py:81-92`:

```python
elif cfg.model.transition == 'marginal':
    node_types = self.dataset_info.node_types.float()
    x_marginals = node_types / torch.sum(node_types)
    edge_types = self.dataset_info.edge_types.float()
    e_marginals = edge_types / torch.sum(edge_types)
    self.transition_model = MarginalUniformTransition(x_marginals=x_marginals,
                                                      e_marginals=e_marginals,
                                                      y_classes=self.ydim_output)
    self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                        y=torch.ones(self.ydim_output) / self.ydim_output)
```

### Hyperparameters

| Name | Default | Config |
|------|---------|--------|
| `transition` | `'marginal'` | `configs/model/discrete.yaml:3` |

---

## 5. Transition matrices

### Uniform (`DiscreteUniformTransition`, `src/diffusion/noise_schedule.py:82-135`)

$$
Q_t = (1 - \beta_t)\, I + \beta_t \, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top,
\qquad
\bar{Q}_t = \bar{\alpha}_t \, I + (1 - \bar{\alpha}_t)\, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top.
$$

```python
q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
...
q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
```

where `self.u_x = ones(1, K, K) / K`.

### Marginal (`MarginalUniformTransition`, `src/diffusion/noise_schedule.py:138-187`)

$$
Q_t = (1 - \beta_t)\, I + \beta_t\, \mathbf{1}\,\pi^\top,
\qquad
\bar{Q}_t = \bar{\alpha}_t\, I + (1 - \bar{\alpha}_t)\, \mathbf{1}\,\pi^\top,
$$

with each row equal to $\pi$ after the mixing step. The class stores

```python
self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
```

i.e. `(1, K, K)` with every row $=\pi$.

### Shapes

Both `get_Qt(beta_t, device)` and `get_Qt_bar(alpha_bar_t, device)`
return a `PlaceHolder` with

- `X: (B, d_X, d_X)`
- `E: (B, d_E, d_E)`
- `y: (B, d_y, d_y)`

### Non-obvious details

- Each row of $Q_t$ and $\bar{Q}_t$ sums to $1$; `apply_noise` asserts
  this at `src/diffusion_model_discrete.py:425-426`.
- `AbsorbingStateTransition.__init__` contains a bug: it assigns to
  `self.u_e[:, :, abs_state] = 1` instead of `self.u_y`
  (`src/diffusion/noise_schedule.py:202-203`). The class is not
  instantiated by any shipped config, so this never triggers.

---

## 6. Forward noising `apply_noise`

`src/diffusion_model_discrete.py:407-442`.

Procedure:

1. Sample $t_\text{int} \sim \operatorname{Uniform}\{0, \ldots, T\}$ in
   training, or $\{1, \ldots, T\}$ in evaluation (the $t = 0$ term is
   reserved for `reconstruction_logp`):

   ```python
   lowest_t = 0 if self.training else 1
   t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()
   s_int = t_int - 1
   ```

2. Normalise: `t_float = t_int / self.T`, `s_float = s_int / self.T`.
3. Look up $\beta_t$, $\bar{\alpha}_{t-1}$, $\bar{\alpha}_t$ from the
   precomputed buffers.
4. Build the per-sample cumulative transition:
   $\bar{Q}_t^{(X)} \in \mathbb{R}^{B \times d_X \times d_X}$,
   $\bar{Q}_t^{(E)} \in \mathbb{R}^{B \times d_E \times d_E}$.
5. Compute posterior over classes:

   ```python
   probX = X @ Qtb.X                # (B, n, d_X)
   probE = E @ Qtb.E.unsqueeze(1)   # (B, n, n, d_E)
   ```

6. Sample the corrupted state via `sample_discrete_features`
   (`src/diffusion/diffusion_utils.py:233-266`) which:
   - Replaces masked rows with a uniform probability,
   - Flattens and uses `multinomial(1)` independently per entry,
   - Symmetrises $E$ via `triu(·, diagonal=1)` plus its transpose, so
     only upper-triangular samples are drawn and mirrored.
7. One-hot the samples and apply `PlaceHolder.mask(node_mask)`.

Returned dict:

```python
{'t_int', 't', 'beta_t', 'alpha_s_bar', 'alpha_t_bar',
 'X_t', 'E_t', 'y_t', 'node_mask'}
```

### Shapes

| Key | Shape |
|-----|-------|
| `t_int`, `t`, `beta_t`, `alpha_s_bar`, `alpha_t_bar` | `(B, 1)` |
| `X_t` | `(B, n, d_X)` one-hot |
| `E_t` | `(B, n, n, d_E)` one-hot symmetric |
| `y_t` | `(B, d_y)` (copied from input `y`) |
| `node_mask` | `(B, n)` bool |

### Non-obvious details

- `y` is *not* diffused here; it is passed through unchanged.
- Each graph in the batch draws its own $t$.
- $E$ symmetry is enforced by sampling only the strict upper triangle,
  so the $(i, j)$ and $(j, i)$ classes are identical, and the diagonal
  is zero.
- When `self.training` is `False`, $t = 0$ is never sampled here; it
  is handled separately by `reconstruction_logp`.

---

## 7. Extra features

Implemented in `src/diffusion/extra_features.py` and
`src/diffusion/extra_features_molecular.py`. The umbrella class
`ExtraFeatures(extra_features_type, dataset_info)` dispatches by
`extra_features_type` $\in \{\text{null}, \text{cycles},
\text{eigenvalues}, \text{all}\}$.

### Normalised node count

```python
n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
```

appended to $y$ (so every mode carries it).

### Cycle features (`NodeCycleFeatures`, `src/diffusion/extra_features.py:57-71`)

From adjacency $A = \mathbf{1}[E_t$ class $\ge 1]$ (collapsed across
non-zero edge classes, `E_t[..., 1:].sum(-1)`), closed walks give cycle
counts. `KNodeCycles.calculate_kpowers` computes $A^k$ for $k = 1,
\ldots, 6$ via repeated multiplication. Per-node counts for $k \in \{3,
4, 5\}$ and graph-level counts for $k \in \{3, 4, 5, 6\}$ are
assembled:

- $c_3^{(i)} = (A^3)_{ii}/2$, graph total $\operatorname{tr}(A^3)/6$;
- $c_4^{(i)}$: `diag(A^4) - d*(d-1) - (A d)_i`;
- $c_5^{(i)}$: `diag(A^5) - 2 tri * d - (A tri)_i + tri` with
  `tri = diag(A^3)`;
- $c_6$ (graph-level only): the polynomial in `k6_cycle` at
  `src/diffusion/extra_features.py:239-255`.

Counts are divided by 10 and clamped at 1 before concatenation to
$X$ / $y$:

```python
x_cycles = x_cycles / 10
y_cycles = y_cycles / 10
x_cycles[x_cycles > 1] = 1
y_cycles[y_cycles > 1] = 1
```

Per-node shape: `(B, n, 3)`; per-graph shape: `(B, 4)`.

### Spectral features (`EigenFeatures`, `src/diffusion/extra_features.py:74-111`)

Combinatorial Laplacian $L = D - A$ is built per graph with a
diagonal-patch for padded positions:

```python
L = compute_laplacian(A, normalize=False)
mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag
```

(`compute_laplacian` at `src/diffusion/extra_features.py:114-136`
symmetrises via $(L + L^\top)/2$.)

For `mode == 'eigenvalues'`:

- Full spectrum via `torch.linalg.eigvalsh(L)`, divided by
  `sum(mask, dim=1)` (node count), so eigenvalues are expressed in
  units of $1/n$.
- `get_eigenvalues_features(eigenvalues, k=5)` at
  `src/diffusion/extra_features.py:139-155`:
  - `n_connected_components = (ev < 1e-5).sum(dim=-1)`;
  - the next $k = 5$ eigenvalues *after* the zero-eigenvalues are
    gathered; pads with value $2$ if there are fewer than $k$ non-zero
    eigenvalues.

For `mode == 'all'` (default via `extra_features: 'all'`):
- Both eigenvalues and eigenvectors via `torch.linalg.eigh(L)`.
- `get_eigenvectors_features(vectors, node_mask, n_connected, k=2)` at
  `src/diffusion/extra_features.py:158-185`:
  - Flags nodes not in the largest connected component using the mode
    of the rounded first eigenvector ("not-LCC indicator"), shape
    `(B, n, 1)`;
  - Gathers the first $k = 2$ eigenvectors associated with the
    smallest non-zero eigenvalues, shape `(B, n, 2)`.

### Final assembly (`ExtraFeatures.__call__`)

For `extra_features_type == 'all'`:

```python
return utils.PlaceHolder(
    X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
    E=extra_edge_attr,   # shape (B, n, n, 0)
    y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
```

Per-node extra width: $3 + 1 + 2 = 6$. Per-graph extra width: $1 + 4
+ 1 + 5 = 11$. Edge extra width: $0$.

The time $t$ (normalised, `(B, 1)`) is appended to $y$ by
`DiscreteDenoisingDiffusion.compute_extra_data` at
`src/diffusion_model_discrete.py:668-669`.

### Molecular features (`src/diffusion/extra_features_molecular.py`)

Only used by QM9 / MOSES / GuacaMol. `ExtraMolecularFeatures` appends
per-atom `(charge, valency)` to $X$ and graph-weight to $y$:

```python
return utils.PlaceHolder(X=torch.cat((charge, valency), dim=-1),
                         E=extra_edge_attr,
                         y=weight)
```

For non-molecular datasets this is replaced by `DummyExtraFeatures`
which returns empty tensors (`src/main.py:93`).

### Hyperparameters

| Name | Default | Config |
|------|---------|--------|
| `extra_features` | `'all'` | `configs/model/discrete.yaml:10` |

---

## 8. Model architecture — `GraphTransformer`

Defined in `src/models/transformer_model.py`.

### Top-level

```python
class GraphTransformer(nn.Module):
    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in, act_fn_out):
        ...
        self.mlp_in_X = nn.Sequential(Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        # analogous mlp_in_E, mlp_in_y
        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(dx=hidden_dims['dx'], de=hidden_dims['de'], dy=hidden_dims['dy'],
                                n_head=hidden_dims['n_head'],
                                dim_ffX=hidden_dims['dim_ffX'], dim_ffE=hidden_dims['dim_ffE'])
            for i in range(n_layers)])
        # analogous mlp_out_X, mlp_out_E, mlp_out_y
```

The forward pass:

1. Compute a diagonal mask on $E$ (`diag_mask`).
2. Slice the prefix `X[..., :out_dim_X]` (likewise `E`, `y`). These
   are the one-hot features of $X_t$, $E_t$, $y_t$ (before extras are
   concatenated) and will be added residually at the end.
3. Symmetrise embedded edges: `new_E = (new_E + new_E.transpose(1,
   2)) / 2`.
4. Run $n_\text{layers}$ `XEyTransformerLayer`s.
5. Project back through three output MLPs.
6. Residual add `X_to_out`, `E_to_out`, `y_to_out`; zero the diagonal
   of $E$; symmetrise $E$.

```python
E = (E + E_to_out) * diag_mask
...
E = 1/2 * (E + torch.transpose(E, 1, 2))
return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
```

### Transformer layer

`XEyTransformerLayer` (`src/models/transformer_model.py:16-92`) wraps a
`NodeEdgeBlock` in pre-norm + feedforward + residuals:

```python
newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

newX_d = self.dropoutX1(newX); X = self.normX1(X + newX_d)
newE_d = self.dropoutE1(newE); E = self.normE1(E + newE_d)
new_y_d = self.dropout_y1(new_y); y = self.norm_y1(y + new_y_d)

ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
X = self.normX2(X + self.dropoutX3(ff_outputX))
# analogous for E and y
```

`self.activation = F.relu`. Dropout is 0.1 (default).

### `NodeEdgeBlock`

(`src/models/transformer_model.py:95-213`). Per layer:

- $Q, K, V = W_Q X, W_K X, W_V X$, reshaped to `(B, n, n_head, df)`.
- Raw score $Y_{ij} = Q_i \odot K_j / \sqrt{d_f}$, shape `(B, n, n,
  n_head, df)`.
- **FiLM from E to X**: $Y \leftarrow Y \odot (E_1 + 1) + E_2$ where
  $E_1 = W_{e,\mathrm{mul}} E$ and $E_2 = W_{e,\mathrm{add}} E$ (both
  projected to $d_x$ then reshaped).
- **FiLM from y to E**: $\mathrm{new}E \leftarrow W_{y,e,\mathrm{add}}
  y + (W_{y,e,\mathrm{mul}} y + 1) \cdot \mathrm{new}E$.
- Edge output: `self.e_out(newE) * e_mask1 * e_mask2` to return
  `(B, n, n, d_e)`.
- Attention: `masked_softmax` over dimension $j = 2$ (keys along
  source), multiply by $V$, sum.
- **FiLM from y to X**: $\mathrm{new}X = W_{y,x,\mathrm{add}} y +
  (W_{y,x,\mathrm{mul}} y + 1) \cdot \mathrm{weighted}V$.
- Node output: `self.x_out(newX) * x_mask`.
- Graph-level $y$ update:

  ```python
  y = self.y_y(y); e_y = self.e_y(E); x_y = self.x_y(X)
  new_y = y + x_y + e_y
  new_y = self.y_out(new_y)      # 2-layer MLP with ReLU
  ```

The pooling layers are defined in `src/models/layers.py:5-38`: `Xtoy`
concatenates `[mean, min, max, std]` over the node dim and projects
via a linear layer, `Etoy` does the same over the two edge dimensions.

`masked_softmax` (`src/models/layers.py:41-46`) fills masked entries
with $-\infty$ before softmax.

### Shapes

Input to `GraphTransformer.forward`:
- `X: (B, n, d_X_in)`, where $d_X^{\text{in}} = d_X^{\text{out}} +
  \text{extra}_X$.
- `E: (B, n, n, d_E_in)`, where $d_E^{\text{in}} = d_E^{\text{out}} +
  \text{extra}_E$.
- `y: (B, d_y_in)`.
- `node_mask: (B, n)`.

Output: a `PlaceHolder` with `X: (B, n, d_X_out)`, `E: (B, n, n,
d_E_out)`, `y: (B, d_y_out)`.

### Hyperparameters

| Name | SBM default | Generic default | Config |
|------|-------------|-----------------|--------|
| `n_layers` | 8 | 5 | `configs/model/discrete.yaml:7`, `configs/experiment/sbm.yaml:22` |
| `hidden_dims.dx` | 256 | 256 | `configs/model/discrete.yaml:17` |
| `hidden_dims.de` | 64 | 64 | same |
| `hidden_dims.dy` | 64 | 64 | same |
| `hidden_dims.n_head` | 8 | 8 | same |
| `hidden_dims.dim_ffX` | 256 | 256 | same |
| `hidden_dims.dim_ffE` | 64 | 128 | `configs/experiment/sbm.yaml:31` vs. `configs/model/discrete.yaml:17` |
| `hidden_dims.dim_ffy` | 256 | 128 | same |
| `hidden_mlp_dims.X` | 128 | 256 | `configs/experiment/sbm.yaml:28` vs. `configs/model/discrete.yaml:14` |
| `hidden_mlp_dims.E` | 64 | 128 | same |
| `hidden_mlp_dims.y` | 128 | 128 | same |

---

## 9. Training objective

`TrainLossDiscrete` (`src/metrics/train_metrics.py:62-123`). The
objective is the unweighted-in-$t$ cross-entropy of the clean
data under the $x_0$-parameterised predictor:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\, \mathbf{z}_0,\, \mathbf{z}_t}\,\big[
  \operatorname{CE}(X_0,\, p_\theta(X_0 \mid \mathbf{z}_t))
  + \lambda_E \operatorname{CE}(E_0,\, p_\theta(E_0 \mid \mathbf{z}_t))
  + \lambda_y \operatorname{CE}(y_0,\, p_\theta(y_0 \mid \mathbf{z}_t))
\big],
$$

where cross-entropy uses `F.cross_entropy(preds, argmax(target),
reduction='sum')` (`src/metrics/abstract_metrics.py:89-105`), divided
by total sample count by the `CrossEntropyMetric` aggregator, and where
masked rows are skipped by the predicate `(true != 0.).any(-1)`:

```python
true_X = torch.reshape(true_X, (-1, true_X.size(-1)))
...
mask_X = (true_X != 0.).any(dim=-1)
mask_E = (true_E != 0.).any(dim=-1)
flat_true_X = true_X[mask_X, :]
flat_pred_X = masked_pred_X[mask_X, :]
...
loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0
return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
```

### Shapes

| Tensor | Flattened shape |
|--------|-----------------|
| `masked_pred_X` / `true_X` | `(B·n, d_X)` |
| `masked_pred_E` / `true_E` | `(B·n·n, d_E)` |
| `pred_y` / `true_y` | `(B, d_y)` |

Masking logic: padded node rows have all-zero one-hot $\Rightarrow$
`(true != 0.).any(-1) == False` and are dropped; likewise padded and
diagonal edge positions (because `encode_no_edge` zeroes the diagonal).

### Non-obvious details

- The mask predicate relies on `encode_no_edge` having already
  converted absent-but-valid edges to the class-0 one-hot, which *does*
  satisfy `(true != 0.).any(-1)` and is therefore included in the
  loss. Diagonal entries are zeroed and excluded.
- `lambda_train` is a 2-tuple `[lambda_E, lambda_y]`, both applied as
  scalar multipliers at the final sum. Node CE has implicit weight 1.
- At eval time `compute_val_loss` assembles the variational bound
  (§11) rather than this CE.

### Hyperparameters

| Name | Default | Config |
|------|---------|--------|
| `lambda_train` | `[5, 0]` | `configs/model/discrete.yaml:19` |

---

## 10. Reverse sampling chain

`DiscreteDenoisingDiffusion.sample_batch`
(`src/diffusion_model_discrete.py:491-595`).

1. Draw $n$ per sample from the empirical node-count categorical
   `self.node_dist` (`DistributionNodes` at
   `src/diffusion/distributions.py:4-31`). Build `node_mask`.
2. Sample $z_T$ from $\pi^{\otimes \cdot}$ using
   `sample_discrete_feature_noise(limit_dist, node_mask)`
   (`src/diffusion/diffusion_utils.py:366-394`): flatten $\pi$ across
   `(B, n)` and `(B, n, n)`, draw via `multinomial(1)`, one-hot,
   symmetrise $E$ by retaining only the strict upper triangle plus its
   transpose, mask.
3. Loop $s = T - 1, T - 2, \ldots, 0$ (i.e. `reversed(range(0,
   T))`), setting $t = s + 1$. In normalised time $s/T$ and $t/T$.
4. Per step, call `sample_p_zs_given_zt(s_norm, t_norm, X, E, y,
   node_mask)` (`src/diffusion_model_discrete.py:597-655`).

The posterior step computes

$$
p_\theta(z_s \mid z_t) = \sum_{x_0} q(z_s \mid z_t, x_0)\, p_\theta(x_0 \mid z_t),
$$

where $p_\theta(x_0 \mid z_t)$ is softmax of the network output and
$q(z_s \mid z_t, x_0)$ is derived from
`compute_batched_over0_posterior_distribution`:

$$
q(z_s = k \mid z_t = j, x_0 = i) \propto (z_t Q_t^\top)_k \cdot (x_0 \bar{Q}_s)_k \,/\, (x_0 \bar{Q}_t)_j,
$$

implemented at `src/diffusion/diffusion_utils.py:293-321`:

```python
Qt_T = Qt.transpose(-1, -2)                 # (B, d_t, d_{t-1})
left_term = X_t @ Qt_T                      # (B, N, d_{t-1})
left_term = left_term.unsqueeze(dim=2)      # (B, N, 1, d_{t-1})
right_term = Qsb.unsqueeze(1)               # (B, 1, d_0, d_{t-1})
numerator = left_term * right_term          # (B, N, d_0, d_{t-1})
prod = Qtb @ X_t_transposed                 # (B, d_0, N)
prod = prod.transpose(-1, -2)               # (B, N, d_0)
denominator = prod.unsqueeze(-1)            # (B, N, d_0, 1)
denominator[denominator == 0] = 1e-6
out = numerator / denominator
```

The final per-entry distribution over $z_s$ classes is

```python
weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X       # (B, n, d_0, d_{t-1})
unnormalized_prob_X = weighted_X.sum(dim=2)                   # (B, n, d_{t-1})
unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)
```

with the analogous edge computation (after flattening $E$ to
`(B, n·n, d_E)` and unflattening back). `sample_discrete_features`
draws the categorical samples and symmetrises $E$ to the strict upper
triangle. The result is converted back to one-hot, masked, and
asserted symmetric:

```python
assert (E_s == torch.transpose(E_s, 1, 2)).all()
```

5. After the loop, `sampled_s.mask(node_mask, collapse=True)` converts
   to integer class labels with $-1$ in padded positions, and the
   `molecule_list` is assembled.

### Chains

`keep_chain` graphs have their intermediate states recorded at
`number_chain_steps` evenly spaced snapshots:

```python
write_index = (s_int * number_chain_steps) // self.T
chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
```

The chain is reversed (`reverse_tensor`, time flows from $T$ down to
0 in storage; reversed for playback) and the last frame is repeated
ten times for visualisation.

### Non-obvious details

- The reverse-step posterior is computed in terms of
  $p_\theta(x_0 \mid z_t)$ (the network output), not in terms of a
  noise prediction.
- Every step samples independently per entry; `multinomial(1)` is used,
  not $\operatorname{argmax}$.
- $y$ is initialised to `torch.zeros(y_t.shape[0], 0)` in the discrete
  sampler (non-molecular case) and is not updated through the chain.
- `number_chain_steps < self.T` is asserted at
  `src/diffusion_model_discrete.py:519`.

### Hyperparameters

| Name | SBM default | Default | Config |
|------|------------|---------|--------|
| `samples_to_generate` (validation) | 40 | 512 | `configs/experiment/sbm.yaml:10`, `configs/general/general_default.yaml:14` |
| `samples_to_save` | 9 | 20 | same |
| `chains_to_save` | 1 | 1 | same |
| `number_chain_steps` | 50 | 50 | `configs/general/general_default.yaml:18` |
| `final_model_samples_to_generate` | 40 | 10000 | same |

---

## 11. Variational lower bound (VLB)

`DiscreteDenoisingDiffusion.compute_val_loss`
(`src/diffusion_model_discrete.py:444-483`). The batch-wise negative
log-likelihood estimator is

$$
-\log p_\theta(G) \le -\log p(N) + \operatorname{KL}_{\text{prior}} + T \cdot \mathbb{E}_{t \sim \mathrm{Unif}\{1..T\}} \big[ L_t \big] - L_0,
$$

where each term is:

### a. `log_pN` (node count prior)

```python
N = node_mask.sum(1).long()
log_pN = self.node_dist.log_prob(N)
```

(`DistributionNodes.log_prob`, `src/diffusion/distributions.py:25-31`:
`log(p[N] + 1e-30)`.)

### b. `kl_prior`

(`src/diffusion_model_discrete.py:303-337`.) Computes
$\operatorname{KL}(q(z_T \mid x) \,\|\, \pi)$ via `F.kl_div` after:

- looking up $\bar{\alpha}_T$ at `t_int = T`;
- computing $q(z_T \mid x_0) = x_0 \bar{Q}_T$;
- expanding `limit_dist` to the tensor shape;
- `mask_distributions` (`src/diffusion/diffusion_utils.py:324-356`)
  replaces padded rows with a degenerate one-hot, adds $10^{-7}$ for
  numerical stability, and renormalises.

```python
kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E)
```

### c. `compute_Lt` — the diffusion term

(`src/diffusion_model_discrete.py:339-366`.) For a single sampled $t$,

$$
L_t = \operatorname{KL}\!\big(q(z_{t-1} \mid z_t, x_0) \,\|\, p_\theta(z_{t-1} \mid z_t)\big).
$$

`posterior_distributions` (`src/diffusion/diffusion_utils.py:359-363`)
relies on `compute_posterior_distribution`:

$$
\hat{q}_{t-1}(k) \propto (z_t Q_t^\top)_k \cdot (x_0 \bar{Q}_{t-1})_k,
$$

and normalises by $x_0 \bar{Q}_t z_t^\top$
(`src/diffusion/diffusion_utils.py:269-290`):

```python
left_term = M_t @ Qt_M_T
right_term = M @ Qsb_M
product = left_term * right_term
denom = M @ Qtb_M
denom = (denom * M_t).sum(dim=-1)
prob = product / denom.unsqueeze(-1)
```

The KL divergence is taken using `SumExceptBatchKL`
(`src/metrics/abstract_metrics.py:75-86`). The return value is $T \cdot
(KL_X + KL_E)$. The factor $T$ reflects Monte-Carlo estimation of a
sum $\sum_{t=1}^{T} L_t$ using a single uniform sample.

### d. `reconstruction_logp` — the $L_0$ term

(`src/diffusion_model_discrete.py:368-405`.)

- Force $t = 0$, compute $Q_0$ from $\beta_0$.
- Compute $x_0 Q_0$ and *sample* $z_0$ via
  `sample_discrete_features`. One-hot.
- Run the network on $z_0$ with extras to obtain predicted
  distributions $p_\theta(\cdot \mid z_0)$.
- Replace masked and diagonal rows with all-ones (so the log of 1 is
  zero and they do not contribute when multiplied by one-hot targets).

```python
probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)
diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
probE0[diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)] = torch.ones(self.Edim_output).type_as(probE0)
```

### e. Combination

```python
loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())
nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
```

Note `loss_term_0` is $\sum X \log p_\theta + \sum E \log p_\theta$
(i.e. a positive log-likelihood), so subtracting it adds the
negative-log-likelihood term of $L_0$.

### Non-obvious details

- The validation NLL uses `lowest_t = 1` in `apply_noise`, so
  `compute_Lt` never sees $t = 0$. The reconstruction term is instead
  computed independently with $t = 0$.
- `compute_Lt` returns $T \cdot \mathrm{KL}$, then
  `on_validation_epoch_end` at
  `src/diffusion_model_discrete.py:169` logs
  `self.val_X_kl.compute() * self.T` separately for reporting, so the
  two $T$ factors do not compound in the reported NLL (the metric
  stored inside `val_X_kl` is the un-multiplied mean KL used for the
  log entry; the $T$ factor is applied inside `compute_Lt` for
  inclusion in `nlls`). Reading the code carefully: `SumExceptBatchKL`
  stores the KL summed-then-averaged over samples; `compute_Lt` calls
  it once on the batch and returns `self.T * (kl_x + kl_e)` where
  `kl_x` and `kl_e` are the batch averages returned by the metric;
  this is the per-batch term that ends up inside `nlls`. The
  standalone multiplier at
  `src/diffusion_model_discrete.py:169` is for separate logging.

---

## 12. Masking, symmetry, diagonal conventions

- **Diagonal of $E$**: zero after `encode_no_edge`
  (`src/utils.py:74-75`); the model zeros it again in the output
  (`src/models/transformer_model.py:279`). Loss excludes diagonal
  rows because `(true != 0).any(-1)` is false there.
- **Symmetry of $E$**: enforced at every stage:
  - After `sample_discrete_features`: `triu(E, diagonal=1) +
    triu(E, diagonal=1).T` (`src/diffusion/diffusion_utils.py:263-264`).
  - After sampling initial noise: same upper-triangular construction
    (`src/diffusion/diffusion_utils.py:385-390`).
  - Inside the model: `new_E = (new_E + new_E.transpose(1, 2)) / 2`
    before and after the transformer stack
    (`src/models/transformer_model.py:267, 282`).
  - Asserted on sampling: `assert (E == torch.transpose(E, 1, 2)).all()`
    at `src/diffusion_model_discrete.py:518`, and after each reverse
    step at `src/diffusion_model_discrete.py:649`.
  - Asserted inside `PlaceHolder.mask` when `collapse=False`:
    `assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))`
    (`src/utils.py:130`).
- **Padding**: `node_mask` is a bool of shape `(B, n_max)`;
  multiplication by `x_mask = node_mask.unsqueeze(-1)`,
  `e_mask1 = x_mask.unsqueeze(2)`, `e_mask2 = x_mask.unsqueeze(1)`
  zeroes all padded entries. Inside attention, masked keys are set to
  $-\infty$ before softmax (`src/models/layers.py:41-46`).
- **Sign convention**: all $\beta_t$ are in $[0, 1]$; $\bar{\alpha}_t$
  decays monotonically from near 1 at $t = 0$ to near 0 at $t = T$.

---

## 13. Evaluation metrics (sampling time)

Implemented for non-molecular datasets in
`src/analysis/spectre_utils.py`. `SpectreSamplingMetrics.__init__`
(`src/analysis/spectre_utils.py:735-745`) pre-converts all train / val
/ test graphs to NetworkX. `SpectreSamplingMetrics.forward`
(`src/analysis/spectre_utils.py:756-857`) dispatches by `metrics_list`:

| Metric | Function | Meaning |
|--------|----------|---------|
| `degree` | `degree_stats` | MMD over degree histograms |
| `clustering` | `clustering_stats` | MMD over clustering coefficients (100 bins) |
| `spectre` | `spectral_stats` | MMD over Laplacian spectra |
| `orbit` | `orbit_stats_all` | MMD over 4-node orbit counts (uses ORCA) |
| `motif` | `motif_stats` | MMD over 4-cycle counts |
| `sbm` | `eval_acc_sbm_graph` + `is_sbm_graph` | Wald-test on block-model fit |
| `planar` | `eval_acc_planar_graph` + `is_planar_graph` | planarity fraction |

The SBM variant (`SBMSamplingMetrics`, `src/analysis/spectre_utils.py:877-881`)
uses:

```python
metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'sbm']
```

and `compute_emd=False`.

`is_sbm_graph` (`src/analysis/spectre_utils.py:608-657`) fits a
stochastic block model with `graph_tool.minimize_blockmodel_dl`,
refines with 1000 `multiflip_mcmc_sweep(beta=inf)` steps, estimates
per-block intra/inter edge probabilities and computes a Wald statistic

$$
W = \frac{(\hat{p} - p)^2}{\hat{p}(1 - \hat{p}) + \varepsilon},
$$

converts via $\chi^2_1$ to a $p$-value, averages over blocks and
returns `True` iff $p > 0.9$ and the recovered partition has
between 2 and 5 blocks of size in $[20, 40]$. Reference
probabilities: $p_{\text{intra}} = 0.3$, $p_{\text{inter}} = 0.005$.

Additional counts reported together for non-molecular datasets
(`src/analysis/spectre_utils.py:843-852`):

- `sampling/frac_unique`: fraction of generated graphs not
  isomorphic to another generated graph.
- `sampling/frac_unique_non_iso`: further subtracts graphs that
  match any training graph.
- `sampling/frac_unic_non_iso_valid`: intersects with the
  dataset-specific validity predicate (`is_sbm_graph` /
  `is_planar_graph`).
- `sampling/frac_non_iso`: `1 - eval_fraction_isomorphic(...)`,
  i.e. fraction of generated graphs not matching any training graph.

---

## 14. Lightning plumbing

### Training step

`src/diffusion_model_discrete.py:103-120`:

```python
dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
dense_data = dense_data.mask(node_mask)
X, E = dense_data.X, dense_data.E
noisy_data = self.apply_noise(X, E, data.y, node_mask)
extra_data = self.compute_extra_data(noisy_data)
pred = self.forward(noisy_data, extra_data, node_mask)
loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                       true_X=X, true_E=E, true_y=data.y,
                       log=i % self.log_every_steps == 0)
```

### Optimiser

`configure_optimizers` (`src/diffusion_model_discrete.py:122-124`):

```python
return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                         weight_decay=self.cfg.train.weight_decay)
```

AdamW with `amsgrad=True` and no learning-rate scheduler. The
`optimizer` key in `configs/train/train_default.yaml:11` advertises
`adamw,nadamw,nadam` options, but the code only ever uses `AdamW`.

### Validation step

`src/diffusion_model_discrete.py:159-166`: runs `apply_noise` with
`lowest_t = 1`, then `compute_val_loss` (the VLB). The resulting NLL
is reported via `self.val_nll` and logged under `val/epoch_NLL`
(`src/diffusion_model_discrete.py:183`).

Generation is gated by

```python
self.val_counter += 1
if self.val_counter % self.cfg.general.sample_every_val == 0:
    ...
```

(`src/diffusion_model_discrete.py:189-217`). Each trigger samples
`samples_to_generate` graphs in batches of $2 \cdot
\text{batch\_size}$.

### Trainer

`src/main.py:190-200`:

```python
trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                  strategy="ddp_find_unused_parameters_true",
                  accelerator='gpu' if use_gpu else 'cpu',
                  devices=cfg.general.gpus if use_gpu else 1,
                  max_epochs=cfg.train.n_epochs,
                  check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                  fast_dev_run=cfg.general.name == 'debug',
                  enable_progress_bar=False,
                  callbacks=callbacks,
                  log_every_n_steps=50 if name != 'debug' else 1,
                  logger=[])
```

### Checkpointing

`src/main.py:171-179`:

```python
checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                      filename='{epoch}',
                                      monitor='val/epoch_NLL',
                                      save_top_k=5, mode='min', every_n_epochs=1)
last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                 filename='last', every_n_epochs=1)
```

Two callbacks: top-5 by `val/epoch_NLL` and a rolling `last.ckpt`.

### EMA

Declared in `configs/train/train_default.yaml:8` (`ema_decay: 0`)
and gated at `src/main.py:181-183`:

```python
if cfg.train.ema_decay > 0:
    ema_callback = utils.EMA(decay=cfg.train.ema_decay)
    callbacks.append(ema_callback)
```

The shipped YAML disables EMA; `utils.EMA` is not defined in the
repository as read, so enabling EMA via config requires adding that
class.

### Test step

`src/diffusion_model_discrete.py:229-300`: computes the VLB on the
test set, then generates
`final_model_samples_to_generate` graphs and writes them to
`generated_samples<N>.txt`.

---

## 15. Hyperparameter reference

The table below consolidates every shipped knob, its default, the
config file that sets it, and the spec section in which it appears.
"SBM override" lists a value overriding the default for
`configs/experiment/sbm.yaml`.

| Knob | Role | Default | SBM override | Config file | Spec § |
|------|------|---------|-------------|-------------|--------|
| `general.name` | run/experiment name | `graph-tf-model` | `sbm` | `configs/general/general_default.yaml:2`, `configs/experiment/sbm.yaml:3` | 14 |
| `general.wandb` | W&B mode | `online` | `online` | `configs/general/general_default.yaml:4`, `configs/experiment/sbm.yaml:5` | 14 |
| `general.gpus` | number of GPUs | 1 | 1 | `configs/general/general_default.yaml:5` | 14 |
| `general.check_val_every_n_epochs` | validation cadence | 5 | 100 | `configs/general/general_default.yaml:11`, `configs/experiment/sbm.yaml:8` | 14 |
| `general.sample_every_val` | generation cadence (in val calls) | 4 | 4 | same | 10, 14 |
| `general.samples_to_generate` | val-time graphs | 512 | 40 | `configs/general/general_default.yaml:14`, `configs/experiment/sbm.yaml:10` | 10 |
| `general.samples_to_save` | val-time graphs saved | 20 | 9 | same | 10 |
| `general.chains_to_save` | val-time chains saved | 1 | 1 | same | 10 |
| `general.log_every_steps` | train logging cadence | 50 | 50 | `configs/general/general_default.yaml:17` | 14 |
| `general.number_chain_steps` | frames per chain gif | 50 | 50 | `configs/general/general_default.yaml:18` | 10 |
| `general.final_model_samples_to_generate` | test-time graphs | 10000 | 40 | `configs/general/general_default.yaml:20`, `configs/experiment/sbm.yaml:13` | 14 |
| `general.final_model_samples_to_save` | test-time graphs saved | 30 | 30 | same | 14 |
| `general.final_model_chains_to_save` | test-time chains | 20 | 20 | same | 14 |
| `general.evaluate_all_checkpoints` | iterate over all ckpts | `False` | inherited | `configs/general/general_default.yaml:27` | 14 |
| `model.type` | continuous vs discrete | `discrete` | inherited | `configs/model/discrete.yaml:2` | 1, 9 |
| `model.transition` | limit distribution | `marginal` | inherited | `configs/model/discrete.yaml:3` | 4, 5 |
| `model.model` | architecture family | `graph_tf` | inherited | `configs/model/discrete.yaml:4` | 8 |
| `model.diffusion_steps` | $T$ | 500 | 1000 | `configs/model/discrete.yaml:5`, `configs/experiment/sbm.yaml:21` | 2, 3 |
| `model.diffusion_noise_schedule` | schedule name | `cosine` | inherited | `configs/model/discrete.yaml:6` | 3 |
| `model.n_layers` | transformer depth | 5 | 8 | `configs/model/discrete.yaml:7`, `configs/experiment/sbm.yaml:22` | 8 |
| `model.extra_features` | cycles / spectral / all | `all` | `all` | `configs/model/discrete.yaml:10`, `configs/experiment/sbm.yaml:24` | 7 |
| `model.hidden_mlp_dims.X/E/y` | input/output MLP widths | 256 / 128 / 128 | 128 / 64 / 128 | `configs/model/discrete.yaml:14`, `configs/experiment/sbm.yaml:28` | 8 |
| `model.hidden_dims.dx` | node embedding | 256 | 256 | same | 8 |
| `model.hidden_dims.de` | edge embedding | 64 | 64 | same | 8 |
| `model.hidden_dims.dy` | y embedding | 64 | 64 | same | 8 |
| `model.hidden_dims.n_head` | attention heads | 8 | 8 | same | 8 |
| `model.hidden_dims.dim_ffX` | node FFN width | 256 | 256 | same | 8 |
| `model.hidden_dims.dim_ffE` | edge FFN width | 128 | 64 | same | 8 |
| `model.hidden_dims.dim_ffy` | y FFN width | 128 | 256 | same | 8 |
| `model.lambda_train` | `[lambda_E, lambda_y]` | `[5, 0]` | `[5, 0]` | `configs/model/discrete.yaml:19` | 9 |
| `train.n_epochs` | max epochs | 1000 | 50000 | `configs/train/train_default.yaml:2`, `configs/experiment/sbm.yaml:17` | 14 |
| `train.batch_size` | graphs per batch | 512 | 12 | `configs/train/train_default.yaml:3`, `configs/experiment/sbm.yaml:18` | 14 |
| `train.lr` | AdamW lr | `2e-4` | inherited | `configs/train/train_default.yaml:4` | 14 |
| `train.clip_grad` | gradient clip value | `null` | inherited | `configs/train/train_default.yaml:5` | 14 |
| `train.save_model` | enable checkpoints | `True` | `True` | `configs/train/train_default.yaml:6`, `configs/experiment/sbm.yaml:19` | 14 |
| `train.num_workers` | DataLoader workers | 0 | inherited | `configs/train/train_default.yaml:7` | — |
| `train.ema_decay` | EMA (0 disables) | 0 | inherited | `configs/train/train_default.yaml:8` | 14 |
| `train.weight_decay` | AdamW weight decay | `1e-12` | inherited | `configs/train/train_default.yaml:10` | 14 |
| `train.optimizer` | optimiser name (unused; code hardcodes AdamW) | `adamw` | inherited | `configs/train/train_default.yaml:11` | 14 |
| `train.seed` | RNG seed | 0 | inherited | `configs/train/train_default.yaml:12` | — |
| `dataset.name` | dataset selector | `qm9` (default config) | `sbm` (via experiment) | `configs/dataset/*.yaml`, `configs/experiment/sbm.yaml:1` | 1 |
| `dataset.datadir` | data root | dataset-dependent | `data/sbm/` | `configs/dataset/sbm.yaml:3` | 1 |

For the planar experiment, `n_layers=10`, `batch_size=64`,
`n_epochs=100000` (`configs/experiment/planar.yaml:17-22`).
