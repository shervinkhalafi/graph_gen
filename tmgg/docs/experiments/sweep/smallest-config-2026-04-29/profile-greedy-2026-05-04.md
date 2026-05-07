# Profile report — Greedy (round-5 winner), 2026-05-04

Source artefacts (Modal volume `tmgg-outputs`):
- `profiles/greedy_20260504T093435/train/{summary.txt, trace.json}`
- `profiles/greedy_20260504T093435/eval/{summary.txt, trace.json}` (still
  running at write time; eval section appended on completion)

Config profiled: round-5 Greedy (n_layers=4, dx=16, de=8, dy=8,
dim_ffX=16, dim_ffE=8, dim_ffy=32, ExtraFeatures=`all`),
A100 (`GPU_TIER=fast`), bf16-mixed, 100 steps total
(5 warmup + 20 active steps profiled, then trail).

Headline: **GPU is starved by ~5×.** End-to-end wall time is ~99 s for
100 training steps (~990 ms/step). Inside the 20 profiled active steps,
Self CUDA total = 1.775 s ≈ 88 ms/step, Self CPU total = 9.502 s
≈ 475 ms/step. The transformer FLOPs are negligible (28 K params); the
bottleneck is on the host side, not in the matmul.

## Train hotspots ranked by changeability

| # | Hotspot | Time / step | Share of CUDA | Share of CPU wall | Lever |
|---|---------|-------------|---------------|-------------------|-------|
| 1 | `Optimizer.step#AdamW.step` | 251 ms CPU | — | **50 %** of run\_training\_batch | `fused=True`/`foreach=True`, drastically reduce param-group count, or `torch.compile` the optimizer step |
| 2 | `aten::_linalg_eigh` (ExtraFeatures spectral) | 26.4 ms CUDA | **30 %** | 6.9 % | Cache eigenfeatures across diffusion timesteps that share the underlying graph; or drop `extra_features_type=all` → `cycles` only |
| 3 | `aten::native_layer_norm_backward` | 22.4 ms CUDA over 22 norms / step | **25 %** | 0.45 % | Fuse with `torch.compile` (NVFuser fuses LN backward chains); also bf16 cast traffic |
| 4 | `aten::mm` (Linear backward) | 11.1 ms CUDA over 91 mm / step | **12.5 %** | 2.1 % | Bigger batch (currently the small per-call cost dominates; throughput-bound region) |
| 5 | `cutlass_80_wmma_tensorop_bf16` cluster | 8.7 ms CUDA / step | 9.8 % | — | Already the right kernel; throughput here scales with batch |
| 6 | `on_before_optimizer_step` callback | 15.9 ms CUDA / step | **17.9 %** | — | Audit which Lightning callback fires here (likely `OptimizerHealth` or grad-clip); make it cheaper or skip every Nth step |

Numbers come from
`profiles/greedy_20260504T093435/train/summary.txt`. CUDA shares are
relative to Self CUDA total = 1.775 s; CPU-wall shares are relative
to `run_training_batch` Self CPU = 4.789 s (one full optimisation
step group).

## Diagnosis

### #1 — Optimizer step is the wall-clock killer (~50 % of step)

`Optimizer.step#AdamW.step` reports **2.578 s self CPU over 19 calls
= 135 ms/step host-side**, plus 821 ms CUDA total / 19 calls
= 43.2 ms/step on the device — i.e. the AdamW kernel launches on the
GPU spend almost as long waiting on the host as actually running.

For a 28 K-parameter model this is absurd. Two known causes that fit
the pattern:

1. **Per-tensor unfused AdamW.** PyTorch's default-construction AdamW
   iterates parameter tensors one-by-one when `foreach=False` and
   `fused=False`. With many small tensors (per-head, per-block,
   biases) the launch overhead dominates the compute.
2. **Many param groups.** If the optimiser was constructed with
   distinct LR/WD groups per layer (common in ported DiGress code),
   each group walks the full state-dict step machinery separately.

Cheapest probe: print `len(optimizer.param_groups)` and
`optimizer.defaults.get("fused", False)` at fit-start. If the answer
is "1 param group, fused/foreach=False", a one-line `fused=True` flip
likely cuts 100+ ms/step.

### #2 — Spectral ExtraFeatures dominate the GPU

`aten::_linalg_eigh` is called 20 times (once per training step), each
launching ~628 `syevbj_batch_32x16` kernels — that is one eigendecomp
per (graph, t) pair in the batch. At 30 % of CUDA time it is the single
largest CUDA-side line item.

This is the ExtraFeatures `all` setting (spectral + cycles + identity).
The eigen-features only depend on the **clean** adjacency, not the
noised one — they are the *same* across all diffusion timesteps for a
given graph. Caching them per-batch (compute once, broadcast across t)
should cut the cost by `T = num_timesteps_in_batch` (≥ 10×). If
caching is more invasive, dropping `extra_features_type=all` →
`cycles` (no eigh) saves all 528 ms/20 steps = 26 ms/step on CUDA;
that's also the largest single Round-6 ablation candidate.

### #3 — LayerNorm backward is unfused

22 layer norms / step × 1.02 ms backward = 22 ms CUDA (25 % of
CUDA time). Each norm fires its own `GammaBetaBackward` kernel
(20.5 % CUDA on its own). Under bf16-mixed the norms also incur
extra cast traffic. `torch.compile()` with NVFuser fuses
`LayerNormBackward → GammaBetaBackward` into one kernel — the pickup
doc explicitly notes `torch.compile` is **not** currently used in
`tmgg.training`. Enabling it is the largest single non-architectural
speedup available (estimate: −15–20 ms CUDA / step on this stack).

### #4–5 — Linear / MM backward

Already on the right cutlass tensor-core kernel (bf16 wmma). 12.5 %
CUDA and 9.8 % CUDA on the cutlass cluster mean these are not the
limit — they will *get faster per token* with a larger batch. The
batch-size lever is therefore secondary to the AdamW one.

### #6 — `on_before_optimizer_step` 17.9 % CUDA

This is the LightningModule hook that fires between backward and
step. 318 ms CUDA over 20 active steps = 15.9 ms/step. In the tmgg
codebase the obvious candidate is the `OptimizerHealth` mixin
draining (parameter-norm, grad-norm, etc.). Worth checking whether
the per-step health metrics can be sampled every Nth step instead of
every step.

## Decision menu for round-6

Ranked by effort × expected wall-clock impact:

| Lever | Effort | Expected wall-clock cut | Notes |
|-------|--------|--------------------------|-------|
| **A. AdamW `fused=True`** (or reduce param-groups) | minutes | **−25–30 %** of step time | Highest leverage. One-line fix in optimiser construction. |
| **B. `torch.compile(model)`** | hours (mode/strict tweaks) | **−15–25 %** of step time | Fuses LN + matmul + activation; addresses #3 + indirectly #4. Risk: graph breaks on the diffusion loop. |
| **C. Cache spectral ExtraFeatures across t** | hours (refactor in `tmgg.models.digress.extra_features`) | −15 % CUDA / step | Largest CUDA win. Doesn't touch host overhead. |
| **D. Drop `extra_features_type=all` → `cycles`** | minutes (yaml flip) | −15 % CUDA / step, **but changes model expressivity** | Cleanest ablation; needs validation NLL re-check. Pair with a small Round-6 sweep. |
| **E. Sample `on_before_optimizer_step` health metrics every Nth step** | low | −5–10 % wall | Lightning callback audit. |
| **F. Bigger batch (×2)** | minutes | Per-step time ≈ flat, throughput ×2 | Only worth it after A is done; otherwise CPU stays the bottleneck. |

Recommendation: **A first** (cheap, decisive), then **B**. C/D should
be a deliberate science decision (do we still want spectral
features?), not a perf shortcut.

## Eval profile

Source: 12 GB chrome trace at
`local_profiles/greedy_20260504T093435/eval/trace.json`, aggregated
locally by `scripts/profiling/aggregate_chrome_trace.py` (the in-Modal
`key_averages().table()` aggregation hit Modal's 900 s heartbeat — the
profiler captured everything; only the post-profile aggregation step
timed out, so we re-aggregate on the host). Aggregation parsed
21,109,191 of an estimated ~22 M events before a single
`"correlation"` field caused a lexical error in a trailing
GPU-annotation block. The diffusion sample loop is highly repetitive
(32 graphs × 500 timesteps × uniform-shape forwards), so the
aggregation is representative.

Run shape: 32 samples, T=500 diffusion timesteps, A100, bf16.
Eval-cycle wall time end-to-end was ~5.4 min (modal log), of which:
- Self CUDA total: **48.9 s**
- Self CPU total: **397.4 s**

GPU is **idle ~88 % of the eval cycle.** Almost everything left is
either eigendecomposition or host-GPU synchronisation overhead.

### Top CUDA hotspots (eval)

| # | Kernel | CUDA total | Share | Calls | Notes |
|---|--------|------------|-------|-------|-------|
| 1 | `syevbj_batch_32x16` | **15.69 s** | **32.1 %** | 700 183 | Jacobi eigenvalue kernel — *the* spectral feature compute |
| 2 | `row_rotate_batch_32x16_phase1` | 5.22 s | 10.7 % | 700 183 | Same eigendecomp, Jacobi rotation pass |
| 3 | `column_rotate_batch_32x16` | 4.76 s | 9.7 % | 700 183 | Same eigendecomp |
| 4 | `row_rotate_batch_32x32_phase2` | 2.30 s | 4.7 % | 700 183 | Same eigendecomp |
| **Subtotal: eigendecomp kernels** | **27.97 s** | **57.2 %** | | | **Spectral ExtraFeatures alone consume more than half of eval CUDA** |
| 5 | `vectorized_layer_norm_kernel` | 6.27 s | 12.8 % | 8 976 | LayerNorm forward |
| 6 | `ampere_sgemm_32x128_tn` (matmul) | 5.27 s | 10.8 % | 30 366 | The actual model |
| 7 | `ampere_sgemm_64x64_tn` | 2.19 s | 4.5 % | 748 | Matmul (small) |
| 8 | softmax / elementwise | ~2.5 s | ~5 % | – | Attention internals |

Net: the **transformer** itself (matmuls + LN + softmax) accounts for
~28 % of CUDA. The **spectral ExtraFeatures** account for **57 %**.
Everything else is glue.

### Top CPU hotspots (eval)

| # | Op | CPU total | Share | Calls | Diagnosis |
|---|----|-----------|-------|-------|-----------|
| 1 | `aten::_linalg_eigh` | **99.1 s** | **24.95 %** | 1 039 | Each eigh call sits ~95 ms wall — GPU compute time + host wait. 1 039 calls × ~95 ms = 99 s. |
| 2 | `aten::linalg_eigh` (wrapper) | 99.1 s | 24.94 % | 1 038 | Same, just the public op |
| 3 | `aten::item` | **36.7 s** | **9.24 %** | **69 640** | GPU→CPU sync points (`tensor.item()`). Each one is a ~527 µs pipeline stall. |
| 4 | `aten::_local_scalar_dense` | 36.6 s | 9.20 % | 69 640 | Internal of `aten::item` — same events |
| 5 | `cudaMemcpyAsync` | 20.2 s | 5.07 % | 166 461 | Host↔device copies |
| 6 | `cudaLaunchKernel` | **18.1 s** | **4.55 %** | **3 492 050** | 3.5 M kernel launches. With 32 × 500 = 16 000 forwards, ≈ 218 launches per forward. Pure launch latency. |
| 7 | `cudaStreamSynchronize` | 13.2 s | 3.33 % | 126 069 | Explicit syncs; probably tied to the `aten::item` pattern |
| 8 | `aten::allclose` | 10.8 s | 2.71 % | 9 056 | Suspicious — possibly a debug consistency check left in the sample loop. ~1.2 ms each, 9 056 calls = 10.8 s pure overhead. |
| 9 | `aten::isclose` | 2.65 s | 0.67 % | 9 056 | Paired with `allclose` above |
| 10 | `aten::eye` | 2.19 s | 0.55 % | 30 192 | Identity-matrix construction per step — almost always cacheable |

The `(_)linalg_eigh` rows alone account for **50 % of all CPU wall
time on eval**. Combined with the 57 % of CUDA, this is the
overwhelming bottleneck.

### Diagnosis

#### #1 — Spectral ExtraFeatures dominate eval (57 % CUDA, 50 % CPU)

In training, the eigendecomp can be cached because the **clean** graph
is fixed (only the noised version changes per step). **In eval, the
graph changes every diffusion timestep**, so the eigenfeatures must be
recomputed for every (sample, t) pair. That explains why eval is
proportionally far worse than train for this op:
- Train: 528 ms over 20 active steps = 26 ms/step (29.7 % of CUDA)
- Eval: 27.97 s over the cycle = **57 % of CUDA**, 1 039 outer eigh
  calls (≈ 32 samples × 32 timestep-batches)

This is a **science-vs-cost** decision rather than a perf bug:
spectral features are part of the model's input, the original DiGress
paper relies on them. Three options:

1. **Drop spectral entirely** (`extra_features_type=cycles` or `none`).
   Saves ~57 % of eval CUDA, ~50 % of eval CPU. **Round-6 candidate
   ablation.** Validation MMD will move; the *direction* needs a
   sweep-style check, not a perf assumption.
2. **Subsample**: compute spectral features every Nth diffusion step
   instead of every step (e.g. every 5). Linear cost reduction, model
   sees stale eigenfeatures between refreshes. Untested in this code.
3. **Cheaper spectral approximation**: power-iteration top-k
   eigenpairs instead of full `eigh`. Significant code change.

#### #2 — `aten::item()` 9.24 % CPU = 36.7 s of pipeline stalls

69 640 calls to `.item()` over the eval cycle. Each one forces the GPU
queue to drain so the host can read a scalar. Likely culprits in the
sample loop:
- argmax → int conversion for choosing categorical edge/node states
- "is t > 0" branching guards
- Validity checks (planarity, node count thresholds)

Audit `tmgg.diffusion.noise_process` and the discrete-diffusion sample
loop for `.item()` calls; replace with vectorised tensor ops where
possible. Even halving these saves ~18 s per eval cycle (≈ 6 % of
eval wall time).

#### #3 — `aten::allclose` × 9 056 in the sample loop

This is almost certainly a debug check that should not run in
production sampling. 1.2 ms × 9056 = **10.8 s of pure consistency-
checking overhead per eval cycle.** Find it (`rg "allclose|isclose"
src/tmgg/diffusion src/tmgg/training`), gate it behind a debug flag.
Cheap win.

#### #4 — `cudaLaunchKernel` 3.5 M calls

218 kernel launches per forward × 16 000 forwards. Most are tiny
(elementwise, rotations). `torch.compile()` would fuse most of these
into a few large launches, cutting the 18 s of launch latency
substantially. Same lever already called out for train.

#### #5 — `aten::eye` 30 192 calls

Identity matrix constructed and torn down 30 K times per eval cycle.
Free win: hoist a per-shape cache in the spectral feature module
(`tmgg.models.digress.extra_features`). Saves 2.2 s wall.

### Decision menu — eval

| Lever | Effort | Expected wall-clock cut | Notes |
|-------|--------|--------------------------|-------|
| **A. Reduce diffusion T (500 → 250 or 100)** | minutes (yaml) | **−50 % to −80 %** of eval cycle | Most direct lever. 100 steps is paper-default for some DiGress configs. |
| **B. Drop spectral ExtraFeatures** | minutes (yaml flip) + science check | **−40 %** of eval cycle | Need to compare validation MMD with vs without. Pair with B-train change for consistency. |
| **C. Remove debug `allclose`/`isclose` from sample loop** | low (one or two grep+gate) | −3 % | Cheap, probably worth landing regardless. |
| **D. `torch.compile(model)`** | hours | −10–20 % | Same lever as train. Watch for graph breaks in the diffusion loop. |
| **E. Audit `.item()` calls in sample loop** | medium | −5–10 % | Vectorise scalar checks. |
| **F. Cache `aten::eye` and other constants** | low | −1 % | Sweet but tiny. |

Eval recommendation: **A + B together**. Halving T *and* dropping
spectral takes a 5.4 min cycle to roughly 1 min, which matters for
mid-run validation cadence and for sweep iteration speed. Both are
science decisions (do we need T=500? do we need spectral?), so they
should land as a deliberate Round-6 ablation, not a silent perf
change.

## Combined train + eval picture

The two profiles agree on the structural diagnoses:

| Issue | Train impact | Eval impact | Same root cause? |
|-------|-------------|-------------|------------------|
| Spectral eigendecomp | 30 % CUDA | **57 %** CUDA / **50 %** CPU | Yes — same `ExtraFeatures` module |
| Per-tensor optimiser launches | **50 %** CPU wall | n/a (no optimiser on eval) | Train-only |
| Many small kernel launches | implicit (LN, rotations) | 3.5 M launches, 4.5 % CPU | Yes — `torch.compile` helps both |
| Host-GPU sync stalls | not dominant | **9.2 %** CPU (`.item()`) | Eval-only signature |

**Top three round-6 levers, by combined wall-clock impact:**

1. **`fused=True` AdamW** (train-only, but train wall time is the limit
   for sweep throughput). One-line change.
2. **Diffusion T cut + spectral ablation** (eval-side; halves the
   per-checkpoint eval cost; deliberate science choice). YAML-only.
3. **`torch.compile(model)`** (helps both; biggest non-architectural
   speedup; needs care around graph breaks). Hours of work.

