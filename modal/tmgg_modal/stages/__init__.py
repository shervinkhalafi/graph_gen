"""Stage-specific Modal functions.

Provides Modal entry points for each experimental stage.
"""

from tmgg_modal.stages.stage1 import run_stage1  # pyright: ignore[reportImplicitRelativeImport]
from tmgg_modal.stages.stage2 import run_stage2  # pyright: ignore[reportImplicitRelativeImport]

__all__ = ["run_stage1", "run_stage2"]
