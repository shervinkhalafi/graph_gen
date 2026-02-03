# W&B Tools Skill

Use these tools to export, aggregate, and analyze W&B experiment data.

## When to Use

- User asks about W&B runs, experiments, or metrics
- Need to fetch experiment data for analysis
- Comparing hyperparameters or results across runs
- Creating reports on experiment outcomes

## Tool Selection

| Task | Tool | Example |
|------|------|---------|
| See available projects | `list_entities.py` | "What projects do I have in W&B?" |
| Get experiment data | `export_runs.py` | "Export my spectral_denoising runs" |
| Combine multiple exports | `aggregate_runs.py` | "Merge all exports into one file" |
| Analyze results | `analyze_runs.py` | "Show best runs by test_mse" |

## Workflow

### 1. Discovery
```bash
uv run wandb-tools/list_entities.py --entity graph_denoise_team
```

### 2. Export
```bash
uv run wandb-tools/export_runs.py -e graph_denoise_team -p spectral_denoising -o wandb_export/
```

### 3. Aggregate (if multiple sources)
```bash
uv run wandb-tools/aggregate_runs.py wandb_export/ -o analysis/unified.parquet --state finished
```

### 4. Analyze
```bash
uv run wandb-tools/analyze_runs.py analysis/unified.parquet --group-by stage --top 10
```

## Common Options

**export_runs.py:**
- `-e, --entity`: W&B entity (user/team)
- `-p, --project`: Project name or "*" for all
- `--since 7d`: Only recent runs
- `--include-history`: Fetch training curves

**aggregate_runs.py:**
- `--state finished`: Only completed runs
- `--filter "stage2c"`: Name pattern filter
- `--no-enrich`: Skip derived columns

**analyze_runs.py:**
- `--group-by stage,arch`: Aggregate dimensions
- `--top 20`: Show top N runs
- `--metric metric_test_mse`: Target metric
- `--output results.csv`: Export to CSV

## Output Format

All tools output parquet files with:
- `config_*` columns: Flattened W&B config
- `metric_*` columns: Numeric metrics
- `_config_json`, `_summary_json`: Raw JSON for lossless access
- Derived columns: `stage`, `arch`, `model_type` (after aggregation)

## Error Handling

- If export fails for a project, it continues with others
- Aggregation warns about schema mismatches but proceeds
- Missing columns become NaN in unified output

## Loading Results in Code

```python
import polars as pl

# Load unified data
df = pl.read_parquet("analysis/unified.parquet")

# Filter by stage
stage2c = df.filter(pl.col("stage") == "stage2c")

# Access original config
import json
config = json.loads(df["_config_json"][0])
```
