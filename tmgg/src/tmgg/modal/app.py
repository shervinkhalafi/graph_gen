"""Modal App definition and GPU configurations.

Provides the central Modal App instance and GPU tier mappings
used across all stage runners.
"""

import modal

# Modal App for TMGG spectral denoising experiments
app = modal.App("tmgg-spectral")

# GPU tier configurations
# Maps logical names to Modal GPU specifications (string format)
GPU_CONFIGS = {
    # Development/testing tier (cheapest)
    "debug": "T4",
    # Standard tier for most experiments
    "standard": "A10G",
    # Fast tier for larger models or time-sensitive runs
    "fast": "A100-40GB",
    # Multi-GPU for DiGress or very large experiments
    "multi": "A100-40GB:2",
    # H100 tier for maximum performance (if available)
    "h100": "H100",
}

# Default timeouts per GPU tier (in seconds)
# Set to 24 hours to avoid premature termination during long experiments
DEFAULT_TIMEOUTS = {
    "debug": 86400,  # 24 hours
    "standard": 86400,  # 24 hours
    "fast": 86400,  # 24 hours
    "multi": 86400,  # 24 hours
    "h100": 86400,  # 24 hours
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
