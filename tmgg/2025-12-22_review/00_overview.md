# TMGG Stage Experiments Review

**Date**: 2025-12-22
**Scope**: Code and config sharing analysis across all experimental stages

## Executive Summary

The TMGG project implements 7 experimental stages for validating spectral positional encoding architectures in graph denoising tasks. The codebase demonstrates strong code sharing through inheritance (a single `DenoisingLightningModule` base class handles 80% of training logic), but exhibits significant config duplication that hinders comparability between stages.

### Key Metrics

| Metric | Value |
|--------|-------|
| Total stages | 7 (stage1_poc through stage5_full) |
| Config files | 72 YAML files in `exp_configs/` |
| Code duplication | Low (~5%) due to inheritance |
| Config duplication | High (~60% in single_graph data configs) |
| Identified issues | 6 bugs/inconsistencies |

### Critical Findings

1. **Optimizer inconsistency** between stage1 (adam, no weight decay) and stage1_5/2 (adamw, weight_decay=1e-12) makes cross-stage comparisons unreliable
2. **Nine single_graph data configs** repeat identical boilerplate for batch_size, num_workers, and noise settings
3. **stage1_5 and stage2 are near-duplicates**, differing only by `same_graph_all_splits` and `val_test_seed`

## Navigation

| Document | Description |
|----------|-------------|
| [01_architecture.md](./01_architecture.md) | Code sharing analysis with inheritance diagrams |
| [02_config_sharing.md](./02_config_sharing.md) | Config composition and duplication analysis |
| [03_stage_comparison.md](./03_stage_comparison.md) | Stage-by-stage comparison matrix |
| [04_bugs_and_issues.md](./04_bugs_and_issues.md) | Identified bugs and inconsistencies |
| [05_recommendations.md](./05_recommendations.md) | Improvement recommendations |

## Codebase References

Key files examined during this review:

- [`src/tmgg/experiments/stages/runner.py`](../src/tmgg/experiments/stages/runner.py) — Stage CLI entry points
- [`src/tmgg/experiment_utils/base_lightningmodule.py`](../src/tmgg/experiment_utils/base_lightningmodule.py) — Shared training logic
- [`src/tmgg/exp_configs/stage/`](../src/tmgg/exp_configs/stage/) — Stage configuration files
- [`src/tmgg/exp_configs/data/`](../src/tmgg/exp_configs/data/) — Dataset configuration files
