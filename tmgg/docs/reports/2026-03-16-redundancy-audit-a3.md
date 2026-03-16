# A3 Decision Record: TMGG Codebase Redundancy & Math Audit

**Date:** 2026-03-16
**Branch:** `audit/codebase-redundancy-review`
**Process:** NGT → Pugh → Red Team (3 independent auditors + 1 red team)

---

## Decision

**The codebase is in good shape.** After the cleanup-2 and G40 refactors, only one actionable item remains. The math is correct throughout. Most structural proposals were challenged by the Red Team and found to be either premature or actively harmful.

## Alternatives Considered (8 findings) + Red Team Verdict

| # | Finding | Proposed Action | Red Team Verdict | Rationale |
|---|---------|-----------------|-----------------|-----------|
| F1 | Spectral metrics duplicated in 2 files | Consolidate | **SKIP** | Only 6 lines of trivial arithmetic; consolidating would pollute the tach-protected utils layer |
| F2 | experiment_setup.py thin utility file | Merge into run_experiment.py | **DEFER** | Low payoff, orchestration layer still under active refactoring |
| F3 | NoiseGenerator vs NoiseProcess naming | Rename NoiseGenerator | **SKIP** | Would create NoiseFunction/NoiseProcess pair that's equally confusing; add docstring instead |
| F4 | SyntheticCategoricalDataModule in experiments/ | Move to data/ | **SKIP** | Marginal computation is diffusion-specific; only one consumer; keep experiment-local |
| F5 | Two graph_types.py files | Rename diffusion version | **SKIP** | `containers.py` is vaguer; collision exists in one file only; a comment resolves it |
| F6 | diffusion_sampling.py redundant name | Rename to sampling.py | **SKIP** | Would create sampling.py / sampler.py pair — worse than current state |
| F7 | sanity_check.py never enabled (564 lines) | Remove or document | **PROCEED (document)** | Add usage example; removal loses debugging capability with high reconstruction cost |
| F8 | training/ contains evaluation_metrics | Accept or rename | **DEFER** | Recent rename from _shared_utils; further churn not justified now; track as future intent |

## Action Items

### Do Now (1 item)

**F7: Document sanity_check.py activation.** Add a comment showing `sanity_check: true` usage to the module docstring and `_base_infra.yaml`. Cost: 5 minutes.

### Track for Future (2 items)

- **F2:** Merge experiment_setup.py into run_experiment.py when decomposing the G60 monolith
- **F8:** Consider promoting evaluation_metrics to a sibling `tmgg.evaluation/` package

### Skip (5 items)

F1, F3, F4, F5, F6 — the Red Team successfully argued these would create equal or worse confusion. The current state is intentional, not accidental.

## Key Tradeoffs

The main tension is **clean namespace purity vs stability of a recently-refactored codebase**. The cleanup-2 branch touched 183 files; the G40 branch touched 21 more. Further renames right now would compound git history churn with no functional benefit. The architectural decisions (two noise hierarchies, experiment-local datamodule, monomial basis) are documented as intentional design choices, not oversights.

## Math Verification: All Clear

All 9 mathematical components verified against published literature:
- DDPM posterior (Ho et al. 2020) — correct
- Categorical posterior (Vignac et al. 2023) — correct
- Eigengap, TopK selection, spectral polynomials — correct with documented deviations from standard terminology
- Laplacian, KL, noise schedules — correct

No mathematical issues remain after the cleanup-2 fixes.
