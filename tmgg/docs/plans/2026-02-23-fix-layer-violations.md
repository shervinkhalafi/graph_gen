# Fix Remaining Layer Violations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Eliminate the two remaining cross-layer dependency violations so that `analysis/` and `modal/` no longer import from `experiments/`.

**Architecture:** Two independent fixes. (1) `modal/evaluate.py` currently imports `load_model_from_checkpoint` from `experiments/gaussian_diffusion_generative/`. The training path already dispatches via CLI subprocess — evaluation should follow the same pattern: the Modal function calls the `evaluate_checkpoint` CLI as a subprocess rather than reimplementing evaluation logic inline. (2) `analysis/reports/eigenstructure.py` imports `SpectralAnalyzer` from `experiments/eigenstructure_study/` to re-analyze raw safetensors. The eigenstructure study already writes `analysis.json` with all the fields the report needs, so the report should read that JSON directly.

**Tech Stack:** Python, Modal SDK, subprocess, JSON, pytest

---

### Task 1: Make eigenstructure report read JSON instead of importing SpectralAnalyzer (#23b)

**Files:**
- Modify: `src/tmgg/analysis/reports/eigenstructure.py:119-170`
- Modify: `tests/analysis/test_eigenstructure_report.py`

**Context:** `EigenstructureReport.load_data()` currently does:
```python
from tmgg.experiments.eigenstructure_study import SpectralAnalyzer
analyzer = SpectralAnalyzer(ds_dir)
result = analyzer.analyze()
row = asdict(result)
```
This imports `SpectralAnalyzer` which depends on the eigenstructure study's storage format (safetensors batches, manifests). The eigenstructure study's `analyze` command already writes `analysis.json` with every field in `SpectralAnalysisResult` as a plain dict. The report should read that JSON file directly.

**Step 1: Write a test for the new JSON-based load_data**

Add to `tests/analysis/test_eigenstructure_report.py`:

```python
class TestLoadDataFromJson:
    """load_data should read analysis.json files from disk, not import SpectralAnalyzer.

    Rationale: the report is in the analysis layer, which must not
    import from experiments/. The eigenstructure study already writes
    analysis.json containing all SpectralAnalysisResult fields as a
    flat dict. Reading JSON eliminates the layer violation.
    """

    def test_load_data_reads_analysis_json(self, tmp_path: Path) -> None:
        """Given a results directory with per-dataset subdirectories each
        containing analysis.json, load_data should return a DataFrame
        with one row per dataset and all _METRIC_COLUMNS present."""
        import json

        for ds_name in ["sbm", "er"]:
            ds_dir = tmp_path / ds_name
            ds_dir.mkdir()
            data = {col: 1.0 for col in _METRIC_COLUMNS}
            data["dataset_name"] = ds_name
            data["num_graphs"] = 100
            (ds_dir / "analysis.json").write_text(json.dumps(data))

        config = {
            "report": {
                "data": {
                    "results_dir": str(tmp_path),
                    "datasets": ["sbm", "er"],
                }
            }
        }
        report = EigenstructureReport(name="eigenstructure")
        df = report.load_data(config)
        assert len(df) == 2
        assert set(df["dataset"]) == {"sbm", "er"}

    def test_load_data_missing_analysis_json_raises(self, tmp_path: Path) -> None:
        """If a dataset subdirectory exists but has no analysis.json,
        load_data should raise FileNotFoundError."""
        (tmp_path / "sbm").mkdir()
        config = {
            "report": {
                "data": {
                    "results_dir": str(tmp_path),
                    "datasets": ["sbm"],
                }
            }
        }
        report = EigenstructureReport(name="eigenstructure")
        with pytest.raises(FileNotFoundError, match="analysis.json"):
            report.load_data(config)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/analysis/test_eigenstructure_report.py::TestLoadDataFromJson -v`
Expected: FAIL — current code imports SpectralAnalyzer, not JSON

**Step 3: Rewrite `load_data` to read JSON**

Replace the `load_data` method in `src/tmgg/analysis/reports/eigenstructure.py` (lines 129-170):

```python
    def load_data(self, config: DictConfig | dict[str, Any]) -> pd.DataFrame:
        """Load spectral analysis results from pre-computed JSON files.

        Each dataset subdirectory under ``results_dir`` must contain an
        ``analysis.json`` file written by the eigenstructure study's
        ``analyze`` command (via ``SpectralAnalyzer.save_results()``).

        Parameters
        ----------
        config : DictConfig or dict
            Must contain ``report.data.results_dir`` and
            ``report.data.datasets`` (see ``eigenstructure.yaml``).

        Returns
        -------
        pd.DataFrame
            One row per dataset with spectral metric columns.
        """
        import json

        datasets = _resolve_datasets(config)
        results_dir = _resolve_results_dir(config)

        rows: list[dict[str, Any]] = []
        for ds_name in datasets:
            ds_dir = results_dir / ds_name
            if not ds_dir.is_dir():
                raise FileNotFoundError(
                    f"Dataset directory '{ds_dir}' does not exist. "
                    f"Run the eigenstructure study for dataset '{ds_name}' first."
                )
            json_path = ds_dir / "analysis.json"
            if not json_path.is_file():
                raise FileNotFoundError(
                    f"No analysis.json in '{ds_dir}'. "
                    f"Run 'tmgg-eigenstructure analyze' for dataset '{ds_name}' first."
                )
            with open(json_path) as f:
                row = json.load(f)
            # Normalize key: SpectralAnalysisResult uses dataset_name,
            # but the report DataFrame uses dataset.
            if "dataset_name" in row and "dataset" not in row:
                row["dataset"] = row.pop("dataset_name")
            rows.append(row)

        if not rows:
            raise FileNotFoundError(
                f"No dataset results found in {results_dir}. "
                f"Expected subdirectories for: {datasets}."
            )

        return pd.DataFrame(rows, columns=_METRIC_COLUMNS)  # pyright: ignore[reportArgumentType]
```

Also remove the `from dataclasses import asdict` import at the top of the file (no longer needed). Update the module docstring to remove references to importing `SpectralAnalyzer`.

**Step 4: Run all eigenstructure report tests**

Run: `uv run pytest tests/analysis/test_eigenstructure_report.py -v`
Expected: all pass

**Step 5: Verify no analysis → experiments imports remain**

Run: `rg 'from tmgg\.experiments' src/tmgg/analysis/ --type py`
Expected: zero hits

**Step 6: Commit**

```bash
git add src/tmgg/analysis/reports/eigenstructure.py tests/analysis/test_eigenstructure_report.py
git commit -m "refactor: eigenstructure report reads JSON instead of importing SpectralAnalyzer (#23b)"
```

---

### Task 2: Make modal evaluation dispatch via CLI subprocess (#23a)

**Files:**
- Rewrite: `src/tmgg/modal/evaluate.py`
- Modify: `tests/modal/test_eigenstructure_modal.py` (if it tests evaluate.py internals)

**Context:** The training path works like this: `_functions.py` → `modal_run_cli()` → writes config YAML → calls `subprocess.run([cmd, ...])`. The evaluation path currently bypasses this: `_functions.py` → `modal_evaluate_mmd()` → `run_mmd_evaluation()` → directly imports `load_model_from_checkpoint` from `experiments/`. The fix: rewrite `run_mmd_evaluation()` to call the existing `evaluate_checkpoint` CLI as a subprocess, mirroring the training pattern.

The existing CLI at `tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint` already handles everything: loading checkpoints, generating graphs, computing MMD, writing JSON results. It accepts `--checkpoint`, `--dataset`, `--num-samples`, `--num-nodes`, `--num-steps`, `--mmd-kernel`, `--mmd-sigma`, `--output`, `--device`, `--seed`.

**Step 1: Write a test for the subprocess-based evaluation**

Create or modify test in `tests/modal/test_evaluate.py`:

```python
"""Tests for modal/evaluate.py subprocess dispatch.

Test rationale: modal/evaluate.py must dispatch evaluation via CLI
subprocess (same pattern as training), not import experiment code
directly. These tests verify that run_mmd_evaluation builds the
correct CLI arguments and interprets subprocess results.
"""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from tmgg.modal.evaluate import run_mmd_evaluation, EvaluationInput, EvaluationOutput


class TestRunMmdEvaluation:
    """run_mmd_evaluation dispatches via subprocess, mirroring training."""

    def test_no_experiments_import(self) -> None:
        """modal/evaluate.py must not import from tmgg.experiments."""
        import importlib
        import tmgg.modal.evaluate as mod

        source = importlib.util.find_spec("tmgg.modal.evaluate")
        assert source is not None and source.origin is not None
        text = Path(source.origin).read_text()
        assert "from tmgg.experiments" not in text
        assert "import tmgg.experiments" not in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modal/test_evaluate.py::TestRunMmdEvaluation::test_no_experiments_import -v`
Expected: FAIL — current code has `from tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint import load_model_from_checkpoint`

**Step 3: Rewrite `run_mmd_evaluation` to use subprocess**

Replace `run_mmd_evaluation()` and delete `_reconstruct_data_module()` in `src/tmgg/modal/evaluate.py`. The function should:

1. Build CLI args from `EvaluationInput` fields
2. Call `subprocess.run(["python", "-m", "tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint", ...])` — this is the same pattern as training, where the subprocess import boundary keeps `modal/` clean
3. Read the JSON output file and convert to `EvaluationOutput`

```python
def run_mmd_evaluation(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics via CLI subprocess.

    Mirrors the training dispatch pattern: writes evaluation parameters
    to CLI arguments and calls the evaluate_checkpoint CLI as a subprocess.
    The subprocess handles model loading, data reconstruction, sampling,
    and MMD computation. Results are written to a JSON file on the
    shared volume, then read back and returned.

    Parameters
    ----------
    task_dict
        Serialized EvaluationInput as a dictionary.

    Returns
    -------
    dict
        EvaluationOutput as a dictionary.
    """
    import json
    import subprocess
    import tempfile

    task = EvaluationInput(**task_dict)
    run_id = task.run_id

    output_dir = Path(OUTPUTS_MOUNT) / run_id
    config_path = output_dir / "config.yaml"

    if task.checkpoint_path:
        checkpoint_path = Path(task.checkpoint_path)
    else:
        checkpoint_path = output_dir / "checkpoints" / "last.ckpt"

    checkpoint_name = checkpoint_path.stem

    # Fail fast on missing files
    if not config_path.exists():
        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="failed",
                error_message=f"Config not found: {config_path}",
                timestamp=datetime.now().isoformat(),
            )
        )
    if not checkpoint_path.exists():
        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="failed",
                error_message=f"Checkpoint not found: {checkpoint_path}",
                timestamp=datetime.now().isoformat(),
            )
        )

    # Extract dataset type from saved config
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    dataset_type = config.get("data", {}).get("graph_type", "sbm")

    # Run evaluation for each split
    all_results: dict[str, dict[str, float]] = {}
    device = "cuda" if _cuda_available() else "cpu"

    for split in task.splits:
        result_file = tempfile.NamedTemporaryFile(
            suffix=".json", prefix=f"mmd_{split}_", delete=False
        )
        result_path = Path(result_file.name)
        result_file.close()

        cli_args = [
            "python", "-m",
            "tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint",
            "--checkpoint", str(checkpoint_path),
            "--dataset", dataset_type,
            "--num-samples", str(task.num_samples),
            "--num-nodes", str(config.get("data", {}).get("num_nodes", 20)),
            "--num-steps", str(task.num_steps),
            "--mmd-kernel", task.mmd_kernel,
            "--mmd-sigma", str(task.mmd_sigma),
            "--seed", str(task.seed),
            "--device", device,
            "--output", str(result_path),
        ]

        logger.info(f"[{split}] Running: {' '.join(cli_args)}")
        proc = subprocess.run(cli_args, capture_output=True, text=True)

        if proc.returncode != 0:
            error_tail = proc.stderr[-500:] if proc.stderr else "no stderr"
            logger.error(f"[{split}] CLI failed (exit {proc.returncode}): {error_tail}")
            continue

        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            all_results[split] = result_data.get("mmd_results", {})
            result_path.unlink()
        else:
            logger.warning(f"[{split}] No result file produced")

    # Write combined results to volume
    eval_output_path = output_dir / f"mmd_evaluation_{checkpoint_name}.json"
    eval_data = {
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.now().isoformat(),
        "params": {
            "num_samples": task.num_samples,
            "num_steps": task.num_steps,
            "mmd_kernel": task.mmd_kernel,
            "mmd_sigma": task.mmd_sigma,
            "seed": task.seed,
        },
        "results": all_results,
    }
    with open(eval_output_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    status = "completed" if all_results else "failed"
    error_msg = "No splits produced results" if not all_results else None

    output = EvaluationOutput(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        status=status,
        results=all_results,
        error_message=error_msg,
        evaluation_params=eval_data["params"],
        timestamp=datetime.now().isoformat(),
    )
    return asdict(output)


def _cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

Remove:
- `_reconstruct_data_module()` (lines 346-381) — no longer needed
- The `from tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint import load_model_from_checkpoint` import inside `run_mmd_evaluation` — replaced by subprocess call
- The `from tmgg.experiment_utils.mmd_metrics import ...` import inside `run_mmd_evaluation` — no longer needed (CLI handles it)

Keep:
- `EvaluationInput`, `EvaluationOutput` dataclasses (transport contract)
- `list_checkpoints_for_run()` (no experiment imports, uses only `Path` + `OUTPUTS_MOUNT`)

**Step 4: Run tests**

Run: `uv run pytest tests/modal/test_evaluate.py tests/modal/test_gpu_tier.py -v`
Expected: all pass

**Step 5: Verify no modal → experiments imports remain**

Run: `rg 'from tmgg\.experiments' src/tmgg/modal/ --type py`
Expected: zero hits

**Step 6: Commit**

```bash
git add src/tmgg/modal/evaluate.py tests/modal/test_evaluate.py
git commit -m "refactor: modal evaluation dispatches via CLI subprocess (#23a)"
```

---

## Verification

After both tasks, verify the dependency structure is clean:

```bash
# No analysis → experiments
rg 'from tmgg\.experiments' src/tmgg/analysis/ --type py
# Expected: zero hits

# No modal → experiments
rg 'from tmgg\.experiments' src/tmgg/modal/ --type py
# Expected: zero hits

# Run full test suite
uv run pytest tests/analysis/ tests/modal/ -v
```

The resulting dependency graph should show:
- `analysis/` → (none) — fully isolated
- `modal/` → `experiment_utils` only — subprocess boundary keeps experiment imports in the subprocess
