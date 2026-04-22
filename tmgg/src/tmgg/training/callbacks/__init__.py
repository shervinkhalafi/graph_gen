"""PyTorch Lightning callbacks owned by tmgg.training.

Callbacks defined here keep the orchestration layer slim by isolating
event-driven behaviour (e.g. EMA shadow weights) behind well-named
classes that ``run_experiment`` can register conditionally based on
config.
"""

from .ema import EMACallback

__all__ = ["EMACallback"]
