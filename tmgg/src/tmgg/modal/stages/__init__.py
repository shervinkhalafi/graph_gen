"""Stage-specific Modal functions.

Provides Modal entry points for each experimental stage.
"""

from tmgg.modal.stages.stage1 import run_stage1
from tmgg.modal.stages.stage2 import run_stage2

__all__ = ["run_stage1", "run_stage2"]
