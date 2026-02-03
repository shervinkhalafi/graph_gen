# W&B Tools

Standalone uv scripts for W&B data export, aggregation, and analysis. Each script is self-contained with inline dependencies and can be run directly with `uv run`.

## Prerequisites

### Authentication

The tools support multiple authentication methods (checked in order):

1. **Command line**: `--api-key YOUR_KEY` (export_runs.py only)
2. **Environment variable**: `WANDB_API_KEY`
3. **Project .env file**: `GRAPH_DENOISE_TEAM_SERVICE` key
4. **Default credentials**: `~/.netrc` (from `wandb login`)

For team access (`graph_denoise_team`), create a `.env` file in the project root:

```bash
# .env
GRAPH_DENOISE_TEAM_SERVICE=wandb_v1_xxxxx...
```

The tools will automatically load this key via python-dotenv.

## Tools

### list_entities.py

List accessible W&B teams/entities and their projects with run statistics.

```bash
# List all projects for default entities
uv run wandb-tools/list_entities.py

# List projects for a specific entity
uv run wandb-tools/list_entities.py --entity graph_denoise_team

# Output as JSON
uv run wandb-tools/list_entities.py --json > entities.json

# Discover all accessible entities
uv run wandb-tools/list_entities.py --discover
```

### export_runs.py

Export W&B runs with full configs and metrics to parquet files. Each run record includes:
- Core metadata (id, name, state, timestamps, etc.)
- Flattened config columns (prefixed with `config_`)
- Flattened metric columns (prefixed with `metric_`)
- Raw JSON columns (`_config_json`, `_summary_json`) for lossless preservation

```bash
# Export a specific project
uv run wandb-tools/export_runs.py --entity graph_denoise_team --project spectral_denoising

# Export all projects from an entity
uv run wandb-tools/export_runs.py -e graph_denoise_team -p "*" -o wandb_export/

# Export with date filter
uv run wandb-tools/export_runs.py -e igorkraw --since 7d

# Include training history (slower, larger files)
uv run wandb-tools/export_runs.py -e graph_denoise_team -p spectral_denoising --include-history
```

**Output structure:**
```
output_dir/
├── {entity}_{project}_runs.parquet     # Main runs data
├── {entity}_{project}_history.parquet  # Training history (if --include-history)
└── {entity}_{project}_metadata.json    # Export metadata
```

### aggregate_runs.py

Merge multiple parquet exports into a unified analysis-ready dataframe with derived columns.

```bash
# Aggregate all exports in a directory
uv run wandb-tools/aggregate_runs.py wandb_export/ -o analysis/unified.parquet

# Aggregate specific files with filtering
uv run wandb-tools/aggregate_runs.py wandb_export/*.parquet -o analysis/unified.parquet --state finished

# Filter by name pattern
uv run wandb-tools/aggregate_runs.py wandb_export/ --filter "stage2c" -o analysis/stage2c.parquet

# Skip enrichment (no parsed columns)
uv run wandb-tools/aggregate_runs.py wandb_export/ -o raw.parquet --no-enrich
```

**Added columns:**
- `protocol`: Training protocol (distribution or single_graph)
- `stage`: Parsed from run name (stage1, stage2c, etc.)
- `arch`: Architecture type (gnn_all, digress_default, etc.)
- `model_type`: High-level model category (spectral, digress, gnn)
- `k`, `lr_parsed`, `wd_parsed`, `seed_parsed`: Values parsed from run names

**Protocol filtering** (default: distribution only):
```bash
# Default: only distribution-based experiments (train/test on different graphs)
uv run wandb-tools/aggregate_runs.py wandb_export/ -o out.parquet

# Include single-graph experiments (train/test on same graph)
uv run wandb-tools/aggregate_runs.py wandb_export/ -o out.parquet --protocol all
```

### analyze_runs.py

Quick CLI analysis with automatic architecture/dataset parsing, pivot tables, and top-N display.

```bash
# Use latest export automatically (from wandb_export/)
uv run wandb-tools/analyze_runs.py

# Analyze PyG distribution experiments with pivot table
uv run wandb-tools/analyze_runs.py --filter "stage3_pyg_dist" --metric metric_test_accuracy --descending --pivot

# View top runs by metric
uv run wandb-tools/analyze_runs.py --top 20 --metric metric_test_mse

# Group by parsed dimensions
uv run wandb-tools/analyze_runs.py --group-by architecture,dataset

# Filter and export
uv run wandb-tools/analyze_runs.py --filter "stage2c" -o results.csv

# Show schema info
uv run wandb-tools/analyze_runs.py --schema
```

**Key features:**
- Auto-detects latest parquet export from `wandb_export/`
- Parses `architecture` and `dataset` from run names automatically
- `--pivot` shows metric breakdown by architecture × dataset with sample counts
- `--descending` for accuracy metrics (vs default ascending for loss/error)

## Typical Workflow

```bash
# 1. Check what's available
uv run wandb-tools/list_entities.py --entity graph_denoise_team

# 2. Export runs from target projects
uv run wandb-tools/export_runs.py -e graph_denoise_team -p spectral_denoising -o wandb_export/

# 3. Quick analysis (uses latest export automatically)
uv run wandb-tools/analyze_runs.py --filter "stage3_pyg_dist" --pivot --descending

# 4. For custom filtering, aggregate first
uv run wandb-tools/aggregate_runs.py wandb_export/ -o analysis/unified.parquet --state finished
uv run wandb-tools/analyze_runs.py analysis/unified.parquet --group-by architecture,dataset

# 5. Load in polars for custom analysis
python -c "import polars as pl; df = pl.read_parquet('wandb_export/graph_denoise_team_spectral_denoising_runs.parquet'); print(df.schema)"
```

## Data Preservation

The export process stores data in two forms:

1. **Flattened columns** (`config_*`, `metric_*`): Convenient for querying and filtering
2. **Raw JSON columns** (`_config_json`, `_summary_json`): Lossless preservation of original W&B data

To recover original nested config from a parquet file:

```python
import json
import polars as pl

df = pl.read_parquet("analysis/unified.parquet")
config = json.loads(df["_config_json"][0])
```

## Column Naming Conventions

| Prefix | Source | Description |
|--------|--------|-------------|
| `config_` | W&B run.config | Flattened configuration parameters |
| `metric_` | W&B run.summary | Numeric metrics from training |
| `_` | Internal | Metadata columns (source file, raw JSON) |
| (none) | Derived | Parsed/enriched columns (stage, arch, etc.) |

## Export Behavior

The export tool writes parquet files per-project as soon as each project completes (streaming output). Progress is logged every 100 runs during large exports. This avoids memory issues and provides intermediate results for long-running exports.

For large projects (2000+ runs), exports may run for several minutes. The tool prints progress like:
```
... fetched 100 runs
... fetched 200 runs
...
Saved 2163 runs to wandb_export/graph_denoise_team_spectral_denoising_runs.parquet
```

## Current Data (graph_denoise_team)

As of 2026-01-18, the following projects were exported:

| Project | Runs | Finished | Description |
|---------|------|----------|-------------|
| `spectral_denoising` | 2163 | 2153 | Main experiments |
| `00_initial_experiment_widening` | 269 | - | Early experiments |
| `tmgg-stage2_validation` | 215 | - | Stage 2 validation |

**Key factors of variation** in the experiments:
- `config_model_type`: self_attention, filter_bank, linear_pe, digress_* variants (12 types)
- `config_data_graph_type`: ring_of_cliques, pyg_enzymes, pyg_proteins, sbm, tree, regular (10 types)
- `config_model_k`: 8, 16, 32, 50 (eigenvalue counts)
- `stage`: stage1, stage1c, stage1d, stage1f, stage2, stage2b, stage2c, stage3

**Best performing configurations** (by MSE at eps=0.1):
1. `filter_bank` architecture achieves lowest MSE across most datasets (0.098-0.26 on tree/regular)
2. `self_attention` is second-best (2.9-4.3 on real datasets)
3. k=8 or k=16 generally outperforms k=32 or k=50
4. DiGress variants show higher MSE but were tested on different objectives
