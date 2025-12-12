"""Stage-specific experiment runners.

Provides CLI entry points for each experimental stage:
- Stage 1: Proof of Concept
- Stage 2: Core Validation
- Stage 3: Dataset Diversity (future)
- Stage 4: Real Benchmarks (future)
- Stage 5: Full Validation (future)
"""

from tmgg.experiments.stages.runner import (
    stage1,
    stage2,
    stage3,
    stage4,
    stage5,
)

__all__ = ["stage1", "stage2", "stage3", "stage4", "stage5"]
