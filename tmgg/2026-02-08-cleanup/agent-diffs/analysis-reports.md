# Change Manifest: Steps 1.5 and 1.6 -- Analysis Report Framework and W&B Consumer Updates

## Files Created

### `src/tmgg/analysis/report_base.py` (~270 lines)

Abstract `ReportGenerator` base class using `abc.ABC` with three abstract hooks (`load_data`, `compute_tables`, `generate_figures`) and a concrete `generate` method that orchestrates the full pipeline: load -> compute tables -> generate figures -> write markdown summary. Includes `REPORT_REGISTRY` (module-level dict) and `@register_report(name)` decorator for pluggable report discovery. The `_render_summary` helper produces a markdown file linking all artefacts.

Config accepts `OmegaConf.DictConfig` or plain dict. Output dir is `pathlib.Path`.

### `src/tmgg/analysis/cli.py` (~100 lines)

Click CLI entry point with three commands:
- `generate` -- run a single named report (`--report-name`, `--output-dir`, optional `--config-path`)
- `generate-all` -- run all registered reports under a shared output directory
- `list` -- print available report names from `REPORT_REGISTRY`

### `src/tmgg/analysis/figures.py` (~350 lines)

Shared figure utilities:
- `setup_style()` -- matplotlib/seaborn publication defaults (Agg backend, whitegrid, paper context)
- `box_plot`, `violin_plot`, `heatmap`, `grouped_bar_chart`, `scatter_with_annotations` -- all accept optional `ax` parameter via `_ensure_ax` helper
- `save_figure(fig, path, dpi)` -- saves and closes figure, creates parent dirs

All plot functions forward `**kwargs` to the underlying seaborn calls for flexibility.

## Files Modified

### `src/tmgg/analysis/__init__.py`

Added imports and exports for:
- `ReportGenerator`, `REPORT_REGISTRY`, `register_report` from `report_base`
- `main` from `cli`

Updated module docstring to document the new submodules.

### `wandb-tools/aggregate_runs.py`

Removed 6 local function definitions that duplicated `tmgg.analysis.parsing`:
- `parse_stage`
- `parse_architecture`
- `parse_model_type`
- `parse_run_name_fields`
- `parse_protocol`
- `enrich_dataframe`

Added `from tmgg.analysis.parsing import enrich_dataframe, parse_protocol`. The `filter_dataframe` function still uses `parse_protocol` directly for pre-enrichment protocol filtering. The JSON column logging that was inside the old local `enrich_dataframe` is preserved as a `console.print` call in the `main()` function before calling the canonical `enrich_dataframe`.

Also added `"tmgg"` to the script dependencies header.

Removed unused `import re`.

### `wandb-tools/analyze_runs.py`

Removed 3 local function definitions:
- `parse_architecture` (different pattern set from canonical -- see discrepancies below)
- `parse_dataset` (different interface: took `str`, canonical takes `pd.Series`)
- `enrich_with_parsed_columns`

Added `from tmgg.analysis.parsing import enrich_dataframe`. The `enrich_with_parsed_columns(df)` call is replaced by `enrich_dataframe(df)` with a `display_name -> name` column copy guard for backward compatibility.

Updated column references: `"architecture"` -> `"arch"` in `print_top_runs` display columns and `print_pivot_table` default parameter. Updated help text and docstring usage examples to use `arch` instead of `architecture`.

Added `"tmgg"` to the script dependencies header.

## Discrepancies from Spec

### `analyze_runs.py` -- Pattern Coverage Gap

The original local `parse_architecture` in `analyze_runs.py` detected patterns not present in the canonical `tmgg.analysis.parsing.parse_architecture`:
- `digress_transformer_gnn_qk` (canonical has `gnn_qk` separately)
- `filter_bank`
- `linear_pe`
- `mlp`

The canonical version detects patterns the local version did not:
- `gnn_all`, `gnn_v`, `digress_default`, `asymmetric`, `spectral_linear`, `spectral`

Runs matching the old-only patterns will now map to `"other"`. This is documented in the module docstring of the modified `analyze_runs.py`. If the old patterns are needed, they should be added to `tmgg.analysis.parsing.parse_architecture`.

### `analyze_runs.py` -- Default Dataset Label

The old local `parse_dataset` returned `"synthetic"` as its fallback; the canonical version returns `"unknown"`. This is documented in the module docstring.

### `analyze_runs.py` -- Column Name Change

The old code produced an `"architecture"` column; the canonical `enrich_dataframe` produces `"arch"`. All internal references have been updated, but external consumers that depend on the `"architecture"` column name from `analyze_runs.py` output will need updating.

### `analyze_runs.py` -- Richer Enrichment

The canonical `enrich_dataframe` adds more columns than the old `enrich_with_parsed_columns` did: `protocol`, `stage`, `model_type`, `k`, `lr_parsed`, `wd_parsed`, `seed_parsed`, `eps_parsed`, `asymmetric_flag`. This is strictly additive and should not break existing workflows.

## Verification Results

```
$ uv run basedpyright src/tmgg/analysis/report_base.py src/tmgg/analysis/cli.py src/tmgg/analysis/figures.py
0 errors, 142 warnings, 0 notes
```

All warnings are from matplotlib/seaborn partial type stubs (expected for `**kwargs: Any` forwarding, `Unknown` member types on matplotlib objects, etc.). No actionable errors.

```
$ uv run python -c "from tmgg.analysis.report_base import ReportGenerator, REPORT_REGISTRY; from tmgg.analysis.cli import main; from tmgg.analysis.figures import setup_style; print('imports OK')"
imports OK
```

```
$ uv run python wandb-tools/aggregate_runs.py --help
[works, shows updated help]
```

```
$ uv run python wandb-tools/analyze_runs.py --help
[works, shows updated help with 'arch' references]
```
