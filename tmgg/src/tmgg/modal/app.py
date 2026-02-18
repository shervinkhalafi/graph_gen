"""Modal app constants and GPU configurations.

Pure constants — no ``import modal`` at module level. The actual
``modal.App`` instance lives in ``_functions.py``, the sole deployment
entry-point.
"""

# App name shared between deployment (_functions.py) and runtime
# lookups (runner.py uses modal.Function.from_name with this name).
MODAL_APP_NAME = "tmgg-spectral"

# GPU tier configurations
# Maps logical names to Modal GPU specifications (string format)
GPU_CONFIGS = {
    "debug": "T4",
    "standard": "A10G",
    "fast": "A100-40GB",
    "multi": "A100-40GB:2",
    "h100": "H100",
}

# Default timeouts per GPU tier (in seconds)
# Set to 24 hours to avoid premature termination during long experiments
DEFAULT_TIMEOUTS = {
    "debug": 86400,
    "standard": 86400,
    "fast": 86400,
    "multi": 86400,
    "h100": 86400,
}

# How long containers stay warm after completing a task (in seconds)
# Higher values improve container reuse during sweeps but cost more
DEFAULT_SCALEDOWN_WINDOW = 60

# Memory configurations (for non-GPU containers)
MEMORY_CONFIGS = {
    "small": 2048,  # 2 GB
    "medium": 8192,  # 8 GB
    "large": 32768,  # 32 GB
}
