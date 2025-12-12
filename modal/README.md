# TMGG Modal Cloud Execution

Modal cloud wrappers for running TMGG spectral denoising experiments on cloud GPUs with Tigris S3 storage.

## Setup

### 1. Install dependencies

```bash
cd modal
uv sync
```

### 2. Configure Modal

```bash
# Set Modal credentials (get from modal.com)
modal token set --token-id $TMGG_MODAL_TOKEN_ID --token-secret $TMGG_MODAL_TOKEN_SECRET
```

### 3. Configure Tigris Storage

Create a Modal secret named `tigris-credentials` with:

```bash
modal secret create tigris-credentials \
  TMGG_TIGRIS_BUCKET=your-bucket-name \
  TMGG_TIGRIS_ENDPOINT=https://fly.storage.tigris.dev \
  TMGG_TIGRIS_ACCESS_KEY=your-access-key \
  TMGG_TIGRIS_SECRET_KEY=your-secret-key
```

### 4. Configure W&B (optional)

```bash
modal secret create wandb-credentials \
  WANDB_API_KEY=your-wandb-api-key
```

## Usage

### Stage 1: Proof of Concept (4.4 GPU-hours)

```bash
# Run full stage
uv run python scripts/run_stage1.py

# Dry run to see configurations
uv run python scripts/run_stage1.py --dry-run

# Custom parallelism and GPU
uv run python scripts/run_stage1.py --parallelism 8 --gpu fast
```

### Stage 2: Core Validation (166.5 GPU-hours)

```bash
# Run with Stage 1 best config (default)
uv run python scripts/run_stage2.py

# Run without Stage 1 results
uv run python scripts/run_stage2.py --no-stage1-best

# With fast GPUs for DiGress
uv run python scripts/run_stage2.py --gpu fast
```

### Direct Modal Execution

```bash
# Stage 1
modal run tmgg_modal/stages/stage1.py --parallelism 4

# Stage 2
modal run tmgg_modal/stages/stage2.py --parallelism 4 --gpu fast
```

## GPU Tiers

| Tier | GPU | Use Case |
|------|-----|----------|
| `debug` | T4 | Testing, small experiments |
| `standard` | A10G | Most experiments |
| `fast` | A100 (40GB) | DiGress, larger models |
| `multi` | 2× A100 | Very large experiments |
| `h100` | H100 | Maximum performance |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TMGG_TIGRIS_BUCKET` | Tigris bucket name |
| `TMGG_TIGRIS_ENDPOINT` | Tigris endpoint (default: https://fly.storage.tigris.dev) |
| `TMGG_TIGRIS_ACCESS_KEY` | Tigris access key |
| `TMGG_TIGRIS_SECRET_KEY` | Tigris secret key |
| `WANDB_API_KEY` | Weights & Biases API key |

## Project Structure

```
modal/
├── pyproject.toml          # Modal project dependencies
├── tmgg_modal/
│   ├── __init__.py
│   ├── app.py              # Modal App and GPU configs
│   ├── image.py            # Docker image builder
│   ├── runner.py           # ModalRunner implementation
│   ├── storage.py          # Tigris S3 integration
│   ├── volumes.py          # Modal volume management
│   └── stages/
│       ├── stage1.py       # Stage 1 Modal functions
│       └── stage2.py       # Stage 2 Modal functions
└── scripts/
    ├── run_stage1.py       # Stage 1 CLI
    └── run_stage2.py       # Stage 2 CLI
```
