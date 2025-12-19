# TMGG Documentation

TMGG is a research framework for graph denoising with attention, GNN, and hybrid architectures. It uses Hydra for reproducible experiments, supports multiple noise models, and can run locally or on the cloud.

## Quick start

```bash
uv sync
uv run tmgg-attention
```

Need a different model or a custom configuration? Start with the **Get started** guide and the **Configuration** reference.

## What’s in this documentation

- [Get started](get-started.md) — install, run a first experiment, and learn the basics
- [Configuration](configuration.md) — Hydra config hierarchy and override patterns
- [Experiments](experiments.md) — stage-based workflows, outputs, and logging
- [Data](data.md) — datasets, noise types, and data modules
- [Models](models.md) — available architectures and parameters
- [Cloud execution](cloud.md) — local vs Modal runners and storage backends
- [Architecture](architecture.md) — system design and module layout
- [Extending](extending.md) — add new models, datasets, or backends

## When to use which entry point

- `tmgg-attention` — attention-based denoising
- `tmgg-gnn` — GNN-based denoising
- `tmgg-hybrid` — hybrid GNN + transformer model
- `tmgg-digress` — DiGress transformer
- `tmgg-spectral` — spectral positional encoding models

If you’re unsure, start with `tmgg-attention` and explore overrides in the Configuration guide.
