# G15: Pyright Suppression Cleanup

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce 182 inline `# pyright: ignore` comments to the minimum necessary, using per-file directives for unavoidable library-stub issues and removing redundant/resolvable suppressions, while maintaining 0 pyright errors.

**Architecture:** Three-tier approach: (1) delete redundant suppressions already covered by global config, (2) consolidate unavoidable library-stub suppressions into per-file directives, (3) resolve suppressions that can be fixed with better typing. Each tier is independently committable and verifiable.

**Tech Stack:** basedpyright, PyTorch Lightning, NetworkX stubs

---

## Tier 1: Remove Redundant Suppressions (11 lines deleted)

These suppressions suppress `reportConstantRedefinition`, which is already globally disabled in `pyproject.toml` line 154. They do nothing.

### Task 1.1: Remove redundant suppressions from `noise.py`

**Files:**
- Modify: `src/tmgg/data/noising/noise.py` (lines 56, 61, 62, 68, 69, 120, 183, 275)

- [ ] **Step 1: Remove all 8 `reportConstantRedefinition` suppression comments**

Lines 56, 61, 62, 68, 69, 120, 183, 275 — remove `# pyright: ignore[reportConstantRedefinition]  # math notation` from each, keeping the code.

- [ ] **Step 2: Verify 0 pyright errors**

```bash
uv run basedpyright src/tmgg/data/noising/noise.py
```
Expected: 0 errors (warnings acceptable)

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -k "noise" -v
```

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/data/noising/noise.py
git commit -m "chore(G15): remove 8 redundant pyright suppressions from noise.py

reportConstantRedefinition is already globally disabled in pyproject.toml."
```

### Task 1.2: Remove redundant suppressions from `synthetic_graphs.py`

**Files:**
- Modify: `src/tmgg/data/datasets/synthetic_graphs.py` (lines 172, 192, 395)

- [ ] **Step 1: Remove 3 `reportConstantRedefinition` suppression comments**

Lines 172, 192, 395 — remove `# pyright: ignore[reportConstantRedefinition]` from each.

- [ ] **Step 2: Verify 0 pyright errors**

```bash
uv run basedpyright src/tmgg/data/datasets/synthetic_graphs.py
```

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/synthetic_graphs.py
git commit -m "chore(G15): remove 3 redundant pyright suppressions from synthetic_graphs.py"
```

**Tier 1 total: -11 suppressions**

---

## Tier 2: Consolidate Unavoidable Suppressions via Per-File Directives

These suppressions exist because third-party type stubs (PyTorch, Lightning, NetworkX, Modal) are incomplete. The code is correct. Replace N inline comments with 1 per-file directive per error category.

**Basedpyright per-file directive syntax:** Place `# pyright: reportFoo=false` as a comment at module top level (after docstring, before imports). This disables the rule for the entire file.

### Task 2.1: `transitions.py` — consolidate `reportUninitializedInstanceVariable` (8 -> 1)

**Files:**
- Modify: `src/tmgg/diffusion/transitions.py`

- [ ] **Step 1: Add per-file directive after docstring**

```python
# pyright: reportUninitializedInstanceVariable=false
# PyTorch register_buffer() initialises these attributes at runtime;
# pyright cannot track this pattern.
```

- [ ] **Step 2: Remove all 8 inline `# pyright: ignore[reportUninitializedInstanceVariable]` comments**

Lines 35-37 (DiscreteUniformTransition: `u_x`, `u_e`, `u_y`) and lines 169-173 (MarginalTransition: `u_x`, `u_e`, `u_y`, `x_marginals`, `e_marginals`).

- [ ] **Step 3: Verify 0 errors**

```bash
uv run basedpyright src/tmgg/diffusion/transitions.py
```

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/diffusion/transitions.py
git commit -m "chore(G15): consolidate 8 pyright suppressions in transitions.py to per-file directive"
```

### Task 2.2: `schedule.py` — consolidate `reportUninitializedInstanceVariable` (3 -> 1)

**Files:**
- Modify: `src/tmgg/diffusion/schedule.py`

- [ ] **Step 1: Add per-file directive and remove 3 inline suppressions**

Same pattern as Task 2.1. Lines 58-60 (`betas`, `alphas`, `alpha_bar`).

- [ ] **Step 2: Verify and commit**

```bash
uv run basedpyright src/tmgg/diffusion/schedule.py
git add src/tmgg/diffusion/schedule.py
git commit -m "chore(G15): consolidate 3 pyright suppressions in schedule.py to per-file directive"
```

### Task 2.3: `synthetic_graphs.py` — consolidate `reportArgumentType` (13 -> 1)

**Files:**
- Modify: `src/tmgg/data/datasets/synthetic_graphs.py`

- [ ] **Step 1: Add per-file directive**

```python
# pyright: reportArgumentType=false
# NetworkX's to_numpy_array() dtype parameter has incomplete type stubs.
```

- [ ] **Step 2: Remove all 13 inline `reportArgumentType` suppressions**

Lines 60, 102, 200, 238, 292, 331, 398, 435, 467, 496, 525, 559, 591.

- [ ] **Step 3: Verify and commit**

```bash
uv run basedpyright src/tmgg/data/datasets/synthetic_graphs.py
git add src/tmgg/data/datasets/synthetic_graphs.py
git commit -m "chore(G15): consolidate 13 pyright suppressions in synthetic_graphs.py to per-file directive"
```

### Task 2.4: `modal/_functions.py` — consolidate `reportArgumentType` (7 -> 1)

**Files:**
- Modify: `src/tmgg/modal/_functions.py`

- [ ] **Step 1: Add per-file directive and remove inline suppressions**

```python
# pyright: reportArgumentType=false
# Modal's type stubs are incomplete for volume mount parameters.
```

Remove 7 inline `reportArgumentType` suppressions + the 1 `reportAssignmentType` (check if this is also covered or needs its own directive).

- [ ] **Step 2: Verify and commit**

```bash
uv run basedpyright src/tmgg/modal/_functions.py
git add src/tmgg/modal/_functions.py
git commit -m "chore(G15): consolidate 8 pyright suppressions in _functions.py to per-file directive"
```

### Task 2.5: `base_data_module.py` — consolidate `reportExplicitAny` (7 -> 1)

**Files:**
- Modify: `src/tmgg/data/data_modules/base_data_module.py`

- [ ] **Step 1: Add per-file directive**

```python
# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.
```

- [ ] **Step 2: Remove 7 inline suppressions and verify**

```bash
uv run basedpyright src/tmgg/data/data_modules/base_data_module.py
```

- [ ] **Step 3: Commit**

### Task 2.6: Remaining data_module files (7 -> per-file directives)

**Files:**
- Modify: `src/tmgg/data/data_modules/multigraph_data_module.py` (4 suppressions)
- Modify: `src/tmgg/data/data_modules/data_module.py` (2 suppressions)
- Modify: `src/tmgg/data/data_modules/single_graph_data_module.py` (1 suppression)

- [ ] **Step 1: For each file, check if all suppressions are the same error code. If so, add per-file directive and remove inline comments. If mixed, handle individually.**

- [ ] **Step 2: Verify each file has 0 errors, then commit as a batch**

```bash
uv run basedpyright src/tmgg/data/data_modules/
git commit -m "chore(G15): consolidate pyright suppressions across data_module files"
```

**Tier 2 total: ~49 inline suppressions replaced by ~7 per-file directives**

---

## Tier 3: The Lightning Modules — Systematic Consolidation

These three files account for 85 of 182 suppressions (47%). They suppress a mix of error codes driven by PyTorch Lightning's dynamic typing. Strategy: one per-file directive block per file, consolidating all error codes that appear 3+ times in that file.

### Task 3.1: Audit and categorise each file's suppressions

**Files:**
- `src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py` (43)
- `src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py` (29)
- `src/tmgg/experiments/_shared_utils/lightning_modules/base_graph_module.py` (13)

- [ ] **Step 1: For each file, tally error codes and decide per-file vs keep-inline**

Rules:
- If an error code appears **3+ times** in a file and is caused by the same library-stub limitation: per-file directive.
- If an error code appears **1-2 times** and documents an intentional design choice (e.g. `reportIncompatibleMethodOverride` on a deliberate override): keep inline with a clarifying comment.
- If a suppression can be resolved by adding a type annotation or cast: resolve it.

### Task 3.2: `base_graph_module.py` (13 -> directive block)

- [ ] **Step 1: Add per-file directive block**

Likely candidates based on audit: `reportExplicitAny`, `reportUnknownMemberType`. Keep `reportAttributeAccessIssue` inline if it documents a specific design choice.

- [ ] **Step 2: Remove consolidated inline suppressions, verify 0 errors**

- [ ] **Step 3: Commit**

### Task 3.3: `diffusion_module.py` (43 -> directive block + minimal inline)

- [ ] **Step 1: Add per-file directive block**

Likely: `reportUnknownMemberType=false`, `reportExplicitAny=false`, `reportUnknownVariableType=false`, `reportUnknownArgumentType=false`.

These cover ~38 of 43 suppressions. The remaining ~5 (`reportAttributeAccessIssue`, `reportUnnecessaryComparison`, `reportIncompatibleMethodOverride`) stay inline with comments.

- [ ] **Step 2: Remove consolidated inline suppressions**

- [ ] **Step 3: Verify 0 errors**

```bash
uv run basedpyright src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py
```

- [ ] **Step 4: Commit**

### Task 3.4: `denoising_module.py` (29 -> directive block + minimal inline)

- [ ] **Step 1: Add per-file directive block**

Same pattern as diffusion_module.py. Keep `reportIncompatibleMethodOverride` (3 occurrences) inline since they document intentional override decisions.

- [ ] **Step 2: Remove consolidated inline suppressions, verify, commit**

**Tier 3 total: ~75 inline suppressions replaced by 3 directive blocks + ~8 intentional inline suppressions**

---

## Tier 4: Remaining Files (scattered, 1-3 suppressions each)

~36 suppressions across ~20 files with 1-3 each. Handle file-by-file.

### Task 4.1: Batch process remaining files

- [ ] **Step 1: For each remaining file with pyright suppressions, categorise:**

Files to process (from the audit):
- `diffusion/diffusion_sampling.py` (2)
- `diffusion/sampler.py` (2)
- `diffusion/noise_process.py` (2)
- `diffusion/diffusion_math.py` (1)
- `models/layers/topk_eigen.py` (2)
- `models/factory.py` (3)
- `models/spectral_denoisers/base_spectral.py` (3)
- `models/spectral_denoisers/bilinear.py` (3)
- `models/hybrid/hybrid.py` (1)
- `models/digress/extra_features.py` (1)
- `models/digress/transformer_model.py` (2)
- `experiments/discrete_diffusion_generative/datamodule.py` (5)
- `experiments/_shared_utils/evaluation_metrics/graph_evaluator.py` (1)
- `experiments/_shared_utils/evaluation_metrics/mmd_metrics.py` (1)
- `experiments/_shared_utils/logging.py` (3)
- `experiments/_shared_utils/lightning_modules/digress_checkpoint_compat.py` (1)
- `experiments/eigenstructure_study/collector.py` (2)
- `experiments/grid_search_runner.py` (1)
- `experiments/embedding_study/embeddings/base.py` (1)
- `data/datasets/graph_dataset.py` (1)
- `data/datasets/pyg_datasets.py` (2)

Decision per file:
- **1 suppression:** Keep inline (adding a per-file directive for 1 line is overkill).
- **2-3 same error code:** Per-file directive if stub-caused; keep inline if intentional.
- **2-3 mixed codes:** Keep inline.

- [ ] **Step 2: Process each file, verify 0 errors after each**

- [ ] **Step 3: Commit in batches by directory**

```bash
git commit -m "chore(G15): clean up scattered pyright suppressions across diffusion/"
git commit -m "chore(G15): clean up scattered pyright suppressions across models/"
git commit -m "chore(G15): clean up scattered pyright suppressions across experiments/"
git commit -m "chore(G15): clean up scattered pyright suppressions across data/"
```

---

## Verification

After all tiers, run the full verification:

```bash
# Full pyright check — must be 0 errors
uv run basedpyright src/tmgg/

# Count remaining inline suppressions (target: <30, down from 182)
rg -c "pyright: ignore" src/tmgg/ | awk -F: '{s+=$2} END {print "Total inline suppressions:", s}'

# Count per-file directives added
rg -c "^# pyright: report" src/tmgg/ | awk -F: '{s+=$2} END {print "Per-file directives:", s}'

# Full test suite
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```

**Target outcome:**
- 0 pyright errors
- ~15-25 inline suppressions remaining (intentional overrides, single-occurrence stubs)
- ~12-15 per-file directives (each with a comment explaining why)
- 182 -> ~20 inline suppressions = ~89% reduction

---

## Post-cleanup: Mark G15 FIXED

Update `docs/reports/2026-03-12-tmgg-review/SUMMARY.md`: both G15 rows -> FIXED.
