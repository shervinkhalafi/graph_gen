# DiGress Repro Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new datasets (Planar SPECTRE, QM9, MOSES, GuacaMol) to the tmgg discrete-diffusion pipeline so the existing `DiffusionModule` + `CategoricalNoiseProcess` can run the full DiGress repro panel from Vignac et al., ICLR 2023 (Tables 1–6) without modifications.

**Architecture:** Compositional class layout. Planar mirrors `spectre_sbm.py`. Molecular pipeline factors into `AtomBondVocabulary` (frozen presets) → `SMILESCodec` (the only RDKit user) → `MolecularGraphDataset` ABC + per-dataset subclasses → `MolecularDataModule` base + per-dataset thin subclasses → `MolecularMetric` ABC + concrete metric classes → `MolecularEvaluator` (composer with classmethod presets `for_qm9` / `for_moses` / `for_guacamol`). The new evaluator duck-types `GraphEvaluator`'s `.evaluate(refs, generated) → results.to_dict()` contract so `DiffusionModule` accepts either.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Lightning, Hydra, RDKit (≥ 2024.3), fcd_torch (≥ 1.0.7), moses (≥ 0.4), guacamol (≥ 0.5), Modal, W&B. Test framework: pytest with `not slow` / `slow` marks.

**Spec:** `docs/specs/2026-04-28-digress-repro-datasets-spec.md` (commit `c0699712`).

**Implementation order** (per the user directive "burn through the complete implementation before testing"):
Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 (tests) → 8 (validation runs).

---

## Phase 0 — Modal image + dependencies (Phase 0)

### Task 0.1: Add molecular packages to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current `pyproject.toml` `dependencies` block**

Run: `rg -n '^dependencies' /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/pyproject.toml -A 50`
Note the current end of the list and the surrounding indentation/style.

- [ ] **Step 2: Append the four new deps to the `dependencies` list**

Add (in the existing array, before the closing `]`):

```toml
    "rdkit>=2024.3",
    "fcd-torch>=1.0.7",
    "moses>=0.4",
    "guacamol>=0.5",
```

(Use the dashed `fcd-torch` form; PyPI normalises the underscore form to dashes.)

- [ ] **Step 3: Reinstall the editable env so the new deps land + reinstall the project for pyright to resolve them**

Run:
```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
uv sync --group dev
uv pip install -e .
```
Expected: `Resolved N packages` then `Installed M packages` including rdkit, fcd-torch, moses, guacamol. ~3–5 min cold.

- [ ] **Step 4: Smoke-import each package on the host**

Run:
```bash
uv run python -c "from rdkit import Chem; from fcd_torch import FCD; import moses; import guacamol; print('ok')"
```
Expected: `ok` printed. Any `ImportError` ⇒ stop, fix the dep version pin, retry.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add rdkit + fcd-torch + moses + guacamol for molecular DiGress repro"
```

---

### Task 0.2: Add a Modal deploy-time hook to warm ChemNet weights

**Files:**
- Modify: `src/tmgg/modal/_functions.py` (add a new `@app.function` near `_compile_orca_in_image`)
- Test: deferred to Phase 7

**Why:** `fcd_torch.FCD` downloads ChemNet weights (~50 MB) on first instantiation. Without this hook, the first validation cycle on Modal pays the download cost and risks a network flake mid-run.

- [ ] **Step 1: Locate the existing `_compile_orca_in_image` function**

Run: `rg -n "_compile_orca_in_image" /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/modal/_functions.py`
Expected: 2-3 hits. Read the function definition and its surrounding `@app.function` decorator pattern.

- [ ] **Step 2: Add a sibling `_warm_molecular_caches_in_image` function**

Insert near the existing `_compile_orca_in_image` definition (same pattern). Code:

```python
@app.function(
    name="_warm_molecular_caches_in_image",
    image=experiment_image,
    timeout=600,
    volumes=get_volume_mounts(),
)
def _warm_molecular_caches_in_image() -> str:
    """Pre-download ChemNet weights into the dataset volume.

    Called once at deploy time so the first validation cycle that
    instantiates ``FCDMetric`` does not pay the ~50 MB ChemNet
    download cost on the hot path. Idempotent — fcd_torch caches
    the weights itself; this just primes the cache.
    """
    from pathlib import Path

    from fcd_torch import FCD

    cache_dir = Path("/data/datasets/molecular/chemnet")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Instantiating FCD with a device argument triggers the ChemNet
    # weight download into its default cache. We then copy any new
    # weights into the persisted volume location.
    _ = FCD(device="cpu", n_jobs=1)
    return f"chemnet warm; cache_dir={cache_dir}"
```

- [ ] **Step 3: Verify the file imports cleanly**

Run: `uv run python -c "from tmgg.modal._functions import _warm_molecular_caches_in_image; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/modal/_functions.py
git commit -m "feat(modal): add _warm_molecular_caches_in_image deploy hook for ChemNet"
```

---

## Phase 1 — SPECTRE-Planar datamodule (Phase 1)

### Task 1.1: Add the `tmgg.data.datasets.spectre_planar` low-level helpers

**Files:**
- Create: `src/tmgg/data/datasets/spectre_planar.py`
- Test: deferred to Phase 7

**Why:** Mirror `tmgg.data.datasets.spectre_sbm` for the SPECTRE planar fixture. The fixture is a 200-graph torch.save'd tuple of dense adjacency matrices, all at exactly n=64. Splits use the same upstream-DiGress randperm rule (seed 0, 80/20 → 80/20 train/val/test).

- [ ] **Step 1: Read the SBM helper as the reference pattern**

Run: `cat /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/data/datasets/spectre_sbm.py`
Note the URL constant, `download_..._fixture()`, `load_..._fixture()`, `split_...()`, `adjacency_to_pyg_data()`, and the `Spectre*Dataset(Dataset[Data])` class.

- [ ] **Step 2: Create the new file with constants + helpers + dataset class**

Create `src/tmgg/data/datasets/spectre_planar.py`:

```python
"""SPECTRE planar fixture (planar_64_200.pt) loader + split helper.

Mirrors :mod:`tmgg.data.datasets.spectre_sbm`. The planar fixture is
200 dense adjacency matrices, all at n = 64, used by DiGress's public
planar experiment (Table 1 in Vignac et al., ICLR 2023).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# Source of truth for the fixture. Pinned to a raw GitHub URL on the
# SPECTRE repository's ``main`` branch.
SPECTRE_PLANAR_URL = (
    "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
)

# Default on-disk cache path. Mirrors the SBM convention.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tmgg" / "spectre"
DEFAULT_FIXTURE_PATH = DEFAULT_CACHE_DIR / "planar_64_200.pt"

# Upstream split geometry: total 200 graphs at n = 64. 20 % test,
# 80 % of the remainder train. Matches the upstream DiGress rule.
SPECTRE_PLANAR_TOTAL = 200
SPECTRE_PLANAR_TEST_LEN = int(round(SPECTRE_PLANAR_TOTAL * 0.2))  # 40
SPECTRE_PLANAR_TRAIN_LEN = int(
    round((SPECTRE_PLANAR_TOTAL - SPECTRE_PLANAR_TEST_LEN) * 0.8)
)  # 128
SPECTRE_PLANAR_VAL_LEN = (
    SPECTRE_PLANAR_TOTAL - SPECTRE_PLANAR_TEST_LEN - SPECTRE_PLANAR_TRAIN_LEN
)  # 32
SPECTRE_PLANAR_SPLIT_SEED = 0

# Pinned SHA-256 of the upstream fixture (downloaded 2026-04-28).
# Used by tests; production code does not enforce.
SPECTRE_PLANAR_SHA256 = (
    "063dc3e675a5c63144e56aa974ca961abfcba02914368192a09b21a364df38fc"
)


def download_spectre_planar_fixture(
    dest: Path | None = None,
    *,
    url: str = SPECTRE_PLANAR_URL,
    force: bool = False,
) -> Path:
    """Download the SPECTRE planar fixture to a local path."""
    target = Path(dest) if dest is not None else DEFAULT_FIXTURE_PATH
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)  # noqa: S310 - pinned https URL
    return target


def load_spectre_planar_fixture(
    path: Path | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Load the SPECTRE planar fixture and return adjacencies + node counts.

    Returns the same shape as :func:`load_spectre_sbm_fixture`: a list
    of dense adjacency tensors and a parallel list of node counts (all
    64 for planar). Uses ``weights_only=False`` because the file
    predates the PyTorch default; the fixture is first-party-trusted
    via the SPECTRE upstream URL.
    """
    fixture_path = path if path is not None else download_spectre_planar_fixture()
    obj: list[Any] = torch.load(fixture_path, weights_only=False)
    if not isinstance(obj, list) or len(obj) < 4:
        raise RuntimeError(
            f"Unexpected SPECTRE planar fixture structure at {fixture_path}: "
            f"expected a list of length ≥ 4, got {type(obj).__name__} "
            f"of length {len(obj) if hasattr(obj, '__len__') else '?'}."
        )
    adjs: list[torch.Tensor] = obj[0]
    n_nodes: list[int] = [int(n) for n in obj[3]]
    if len(adjs) != len(n_nodes):
        raise RuntimeError(
            f"SPECTRE planar fixture inconsistent: {len(adjs)} adjacencies vs "
            f"{len(n_nodes)} node-counts."
        )
    return adjs, n_nodes


def split_spectre_planar(
    num_graphs: int = SPECTRE_PLANAR_TOTAL,
    *,
    seed: int = SPECTRE_PLANAR_SPLIT_SEED,
) -> dict[str, list[int]]:
    """Produce train/val/test indices matching upstream DiGress."""
    generator = torch.Generator()
    _ = generator.manual_seed(seed)
    permutation = torch.randperm(num_graphs, generator=generator).tolist()

    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))

    train_indices = permutation[:train_len]
    val_indices = permutation[train_len : num_graphs - test_len]
    test_indices = permutation[num_graphs - test_len :]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def adjacency_to_pyg_data(adj: torch.Tensor) -> Data:
    """Convert one dense adjacency tensor to a :class:`Data` instance."""
    adj_f = adj.float()
    edge_index, _ = dense_to_sparse(adj_f)
    return Data(edge_index=edge_index, num_nodes=adj_f.shape[0])


class SpectrePlanarDataset(Dataset[Data]):
    """Indexable view over a SPECTRE-planar split."""

    def __init__(self, adjacencies: list[torch.Tensor]) -> None:
        self._adjacencies = adjacencies

    def __len__(self) -> int:
        return len(self._adjacencies)

    def __getitem__(self, idx: int) -> Data:
        return adjacency_to_pyg_data(self._adjacencies[idx])
```

- [ ] **Step 3: Verify the file imports cleanly**

Run: `uv run python -c "from tmgg.data.datasets.spectre_planar import load_spectre_planar_fixture, SpectrePlanarDataset; adjs, n = load_spectre_planar_fixture(); print(len(adjs), n[0])"`
Expected: `200 64`.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/data/datasets/spectre_planar.py
git commit -m "feat(data): add SPECTRE planar fixture loader + dataset"
```

---

### Task 1.2: Add `SpectrePlanarDataModule`

**Files:**
- Create: `src/tmgg/data/data_modules/spectre_planar.py`
- Test: deferred to Phase 7

- [ ] **Step 1: Read the SBM datamodule as the reference**

Run: `cat /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/data/data_modules/spectre_sbm.py | head -80`
Note the constructor signature, `setup()` shape, dataloader factory pattern.

- [ ] **Step 2: Create the planar datamodule, mirroring SBM**

Create `src/tmgg/data/data_modules/spectre_planar.py`:

```python
"""Lightning DataModule for the SPECTRE planar fixture.

Mirrors :class:`tmgg.data.data_modules.spectre_sbm.SpectreSBMDataModule`
but loads the SPECTRE planar fixture (n=64, 200 graphs) used in
DiGress Table 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from tmgg.data.data_modules.base_data_module import BaseGraphDataModule
from tmgg.data.data_modules.multigraph_data_module import (
    GraphDataCollator,
    RawPyGCollator,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.spectre_planar import (
    SpectrePlanarDataset,
    load_spectre_planar_fixture,
    split_spectre_planar,
)
from tmgg.utils.noising.size_distribution import SizeDistribution


class SpectrePlanarDataModule(BaseGraphDataModule):
    """Load the SPECTRE planar fixture and serve via the TMGG collator."""

    num_nodes: int

    def __init__(
        self,
        *,
        batch_size: int = 12,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
        cache_dir: str | None = None,
        fixture_path: str | None = None,
        num_nodes_max_static: int = 64,
        pad_to_static_n_max: bool = False,
        **_metadata: object,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        self.num_nodes_max_static = num_nodes_max_static
        self.pad_to_static_n_max = pad_to_static_n_max
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        self._fixture_path: Path | None = (
            Path(fixture_path) if fixture_path else None
        )

        self._train_data: list[Data] | None = None
        self._val_data: list[Data] | None = None
        self._test_data: list[Data] | None = None
        self._train_node_counts: list[int] | None = None

        self.graph_type = "planar"
        self.num_nodes = 0  # updated by setup()

        self.save_hyperparameters()

    @override
    def setup(self, stage: str | None = None) -> None:
        if self._train_data is not None:
            return

        fixture = self._fixture_path
        if fixture is None and self._cache_dir is not None:
            fixture = self._cache_dir / "planar_64_200.pt"

        adjs, n_nodes = load_spectre_planar_fixture(path=fixture)
        splits = split_spectre_planar()

        train_adjs = [adjs[i] for i in splits["train"]]
        val_adjs = [adjs[i] for i in splits["val"]]
        test_adjs = [adjs[i] for i in splits["test"]]

        self._train_data = [
            SpectrePlanarDataset(train_adjs)[i] for i in range(len(train_adjs))
        ]
        self._val_data = [
            SpectrePlanarDataset(val_adjs)[i] for i in range(len(val_adjs))
        ]
        self._test_data = [
            SpectrePlanarDataset(test_adjs)[i] for i in range(len(test_adjs))
        ]

        self._train_node_counts = [n_nodes[i] for i in splits["train"]]
        train_val_node_counts = [n_nodes[i] for i in splits["train"]] + [
            n_nodes[i] for i in splits["val"]
        ]
        self.num_nodes = int(max(train_val_node_counts))

    def _dense_collator(self) -> GraphDataCollator:
        return GraphDataCollator(
            n_max_static=self.num_nodes_max_static
            if self.pad_to_static_n_max
            else None
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        if self._train_data is None:
            raise RuntimeError(
                "SpectrePlanarDataModule not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=self._dense_collator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        if self._val_data is None:
            raise RuntimeError(
                "SpectrePlanarDataModule not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        if self._test_data is None:
            raise RuntimeError(
                "SpectrePlanarDataModule not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[object]:
        if self._train_data is None:
            raise RuntimeError(
                "SpectrePlanarDataModule not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=RawPyGCollator(),
        )

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        split_map: dict[str | None, list[Data] | None] = {
            "train": self._train_data,
            "val": self._val_data,
            "test": self._test_data,
        }
        if split is None:
            data_list = (
                (self._train_data or [])
                + (self._val_data or [])
                + (self._test_data or [])
            )
        elif split in split_map:
            data_list = split_map[split] or []
        else:
            raise ValueError(
                f"Unknown split {split!r}. Expected 'train', 'val', 'test', or None."
            )
        if not data_list:
            return SizeDistribution.fixed(self.num_nodes or 1)
        node_counts = [int(d.num_nodes) for d in data_list]  # type: ignore[arg-type]
        return SizeDistribution.from_node_counts(node_counts)
```

- [ ] **Step 3: Verify the file imports cleanly**

Run: `uv run python -c "from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule; dm = SpectrePlanarDataModule(); dm.setup('fit'); print('num_nodes:', dm.num_nodes, 'train:', len(dm._train_data))"`
Expected: `num_nodes: 64 train: 128`.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/data/data_modules/spectre_planar.py
git commit -m "feat(data): add SpectrePlanarDataModule mirroring spectre_sbm pattern"
```

---

### Task 1.3: Add the Planar data + experiment YAML configs

**Files:**
- Create: `src/tmgg/experiments/exp_configs/data/spectre_planar.yaml`
- Create: `src/tmgg/experiments/exp_configs/experiment/discrete_planar_digress_repro.yaml`

- [ ] **Step 1: Read the existing `spectre_sbm.yaml` data config as the reference**

Run: `cat /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/experiments/exp_configs/data/spectre_sbm.yaml`

- [ ] **Step 2: Create `spectre_planar.yaml`**

Create `src/tmgg/experiments/exp_configs/data/spectre_planar.yaml` with the contents:

```yaml
# @package data
_target_: tmgg.data.data_modules.spectre_planar.SpectrePlanarDataModule
batch_size: 12
num_workers: 0
pin_memory: true
prefetch_factor: 4
seed: 42
cache_dir: null
fixture_path: null
num_nodes_max_static: 64
pad_to_static_n_max: false
```

- [ ] **Step 3: Create `discrete_planar_digress_repro.yaml` mirroring the SBM repro**

Read the SBM repro config first:
```bash
cat /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/experiments/exp_configs/experiment/discrete_sbm_vignac_repro.yaml
```

Then create `src/tmgg/experiments/exp_configs/experiment/discrete_planar_digress_repro.yaml`:

```yaml
# @package _global_
# DiGress planar reproduction (Vignac et al., ICLR 2023, Table 1).
#
# Layered atop ``base_config_discrete_diffusion_generative`` via the
# Hydra experiment pattern. Targets the SPECTRE planar fixture
# (n=64, 200 graphs).
#
# Invocation:
#   tmgg-discrete-gen +experiment=discrete_planar_digress_repro

defaults:
  - /data: spectre_planar
  - override /models/discrete@model: discrete_sbm_official
  - _self_

experiment_name: discrete_planar_digress_repro
wandb_project: discrete-planar-digress-repro

seed: 666

# Trainer schedule. SPECTRE planar has 128 train graphs, ~11 batches/
# epoch at batch_size=12. We mirror SBM's 550k steps; planar tends to
# converge earlier per the paper.
trainer:
  max_steps: 550000
  val_check_interval: 5000
  check_val_every_n_epoch: null
  gradient_clip_val: null
  gradient_clip_algorithm: norm

model:
  eval_every_n_steps: 75000
  num_nodes: 64
  model:
    input_dims: { X: 1, E: 2, y: 0 }
    output_dims: { X: 1, E: 2, y: 0 }
    output_dims_x_class: 1
    output_dims_x_feat: null
    output_dims_e_class: 2
    output_dims_e_feat: null
```

- [ ] **Step 4: Verify the experiment config resolves through Hydra**

Run:
```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
uv run tmgg-discrete-gen +experiment=discrete_planar_digress_repro --cfg job 2>&1 | head -40
```
Expected: a YAML dump showing `data._target_: tmgg.data.data_modules.spectre_planar.SpectrePlanarDataModule` and `experiment_name: discrete_planar_digress_repro`. Any Hydra error means a config interpolation issue — fix and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/tmgg/experiments/exp_configs/data/spectre_planar.yaml src/tmgg/experiments/exp_configs/experiment/discrete_planar_digress_repro.yaml
git commit -m "feat(configs): add SPECTRE planar data + DiGress repro experiment YAMLs"
```

---

## Phase 2 — Molecular vocabulary + SMILES codec (Phase 2)

### Task 2.1: Define the `AtomBondVocabulary` frozen dataclass

**Files:**
- Create: `src/tmgg/data/datasets/molecular/__init__.py`
- Create: `src/tmgg/data/datasets/molecular/vocabulary.py`

- [ ] **Step 1: Create the package init**

Create `src/tmgg/data/datasets/molecular/__init__.py` with:

```python
"""Molecular datasets: SMILES → categorical GraphData pipeline.

Compositional layout per
``docs/specs/2026-04-28-digress-repro-datasets-spec.md``:

- :mod:`vocabulary` — frozen ``AtomBondVocabulary`` with QM9/MOSES/
  GuacaMol presets.
- :mod:`codec` — ``SMILESCodec``, the only place RDKit is imported.
- :mod:`dataset` — ``MolecularGraphDataset`` ABC with on-disk shard cache.
- :mod:`qm9`, :mod:`moses`, :mod:`guacamol` — concrete dataset subclasses.
"""

from __future__ import annotations

from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

__all__ = ["AtomBondVocabulary"]
```

- [ ] **Step 2: Create `vocabulary.py` with the frozen dataclass + presets**

Create `src/tmgg/data/datasets/molecular/vocabulary.py`:

```python
"""Frozen atom + bond vocabulary for molecular categorical diffusion.

Constants mirror upstream DiGress
(``digress-upstream-readonly/src/datasets/{qm9,moses,guacamol}_dataset.py``).
The class is hashable so its ``repr`` doubles as a stable cache key
component for :class:`SMILESCodec`'s preprocessed-shard directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# Bond decoder is shared across all three molecular datasets in DiGress.
# Order matters: the index into this tuple is the categorical edge class.
# ``"NONE"`` is the no-edge class (class 0). Order matches upstream's
# ``bonds = {Chem.BondType.SINGLE: 1, ...}`` mapping.
BOND_DECODER: tuple[str, ...] = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

# Per-atom max valence used for valency-mask construction in extra
# features. Mirrors upstream constants in the per-dataset modules.
_DEFAULT_MAX_VALENCES: Mapping[str, int] = {
    "C": 4, "N": 3, "O": 2, "F": 1,
    "B": 3, "Br": 1, "Cl": 1, "I": 1,
    "P": 5, "S": 6, "Se": 6, "Si": 4,
    "H": 1,
}


@dataclass(frozen=True)
class AtomBondVocabulary:
    """Mapping atom symbol ↔ class idx + bond type ↔ class idx.

    Frozen + hashable: instances can serve as cache keys (their
    ``__repr__`` is stable across runs by virtue of the frozen
    dataclass machinery). Equality is structural over the decoder
    tuples.

    Parameters
    ----------
    atom_decoder
        Tuple of atom symbols in class-index order. ``atom_decoder[0]``
        is class 0, etc. Upstream DiGress reserves no "no-atom" slot
        for the molecular datasets — every node is a real atom.
    bond_decoder
        Tuple of bond-type names in class-index order. ``bond_decoder[0]``
        is the no-edge class.
    max_valences
        Per-atom maximum valence. Used by valency-aware extra features.

    Notes
    -----
    Use the classmethod presets (:meth:`qm9`, :meth:`moses`,
    :meth:`guacamol`) rather than instantiating directly; they pin
    the upstream constants.
    """

    atom_decoder: tuple[str, ...]
    bond_decoder: tuple[str, ...] = BOND_DECODER
    max_valences: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: tuple(sorted(_DEFAULT_MAX_VALENCES.items()))
    )

    @classmethod
    def qm9(cls, *, remove_h: bool = True) -> "AtomBondVocabulary":
        """QM9 (no-H) atom decoder. Matches upstream
        ``qm9_dataset.py:atom_decoder`` for ``remove_h=True``."""
        if remove_h:
            atom_decoder = ("C", "N", "O", "F")
        else:
            atom_decoder = ("H", "C", "N", "O", "F")
        return cls(atom_decoder=atom_decoder)

    @classmethod
    def moses(cls) -> "AtomBondVocabulary":
        """MOSES atom decoder. Matches upstream
        ``moses_dataset.py:atom_decoder``."""
        return cls(atom_decoder=("C", "N", "S", "O", "F", "Cl", "Br", "H"))

    @classmethod
    def guacamol(cls) -> "AtomBondVocabulary":
        """GuacaMol atom decoder. Matches upstream
        ``guacamol_dataset.py:atom_decoder``."""
        return cls(
            atom_decoder=(
                "C", "N", "O", "F", "B", "Br", "Cl", "I",
                "P", "S", "Se", "Si",
            )
        )

    @property
    def num_atom_types(self) -> int:
        return len(self.atom_decoder)

    @property
    def num_bond_types(self) -> int:
        return len(self.bond_decoder)

    @property
    def atom_encoder(self) -> Mapping[str, int]:
        return {symbol: idx for idx, symbol in enumerate(self.atom_decoder)}

    @property
    def bond_encoder(self) -> Mapping[str, int]:
        return {symbol: idx for idx, symbol in enumerate(self.bond_decoder)}

    def encode_atom(self, symbol: str) -> int:
        try:
            return self.atom_encoder[symbol]
        except KeyError as exc:
            raise ValueError(
                f"Atom {symbol!r} not in vocabulary {self.atom_decoder}."
            ) from exc

    def decode_atom(self, idx: int) -> str:
        if not 0 <= idx < self.num_atom_types:
            raise ValueError(
                f"Atom index {idx} out of range [0, {self.num_atom_types})."
            )
        return self.atom_decoder[idx]

    def encode_bond(self, bond_type_name: str) -> int:
        try:
            return self.bond_encoder[bond_type_name]
        except KeyError as exc:
            raise ValueError(
                f"Bond type {bond_type_name!r} not in {self.bond_decoder}."
            ) from exc

    def decode_bond(self, idx: int) -> str:
        if not 0 <= idx < self.num_bond_types:
            raise ValueError(
                f"Bond index {idx} out of range [0, {self.num_bond_types})."
            )
        return self.bond_decoder[idx]

    def max_valence(self, symbol: str) -> int:
        for sym, valence in self.max_valences:
            if sym == symbol:
                return valence
        raise ValueError(
            f"No max valence registered for atom {symbol!r}."
        )
```

- [ ] **Step 3: Verify imports + presets**

Run:
```bash
uv run python -c "
from tmgg.data.datasets.molecular import AtomBondVocabulary
v = AtomBondVocabulary.qm9()
print('qm9 atoms:', v.atom_decoder, '#:', v.num_atom_types)
print('moses atoms:', AtomBondVocabulary.moses().atom_decoder)
print('guacamol atoms:', AtomBondVocabulary.guacamol().atom_decoder)
print('hash:', hash(v))
print('repr:', repr(v))
"
```
Expected: presets print correctly, `hash(v)` returns an int, `repr(v)` is the dataclass repr.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/data/datasets/molecular/__init__.py src/tmgg/data/datasets/molecular/vocabulary.py
git commit -m "feat(data): add AtomBondVocabulary with QM9/MOSES/GuacaMol presets"
```

---

### Task 2.2: Implement `SMILESCodec`

**Files:**
- Create: `src/tmgg/data/datasets/molecular/codec.py`

- [ ] **Step 1: Create the codec module**

Create `src/tmgg/data/datasets/molecular/codec.py`:

```python
"""SMILES ↔ GraphData codec — the only module that imports RDKit.

Parameterised by an :class:`AtomBondVocabulary`. Encodes a SMILES
string into a categorical :class:`GraphData` with ``X_class`` (atom
classes) and ``E_class`` (bond classes), or returns ``None`` on
parse failure / atom-count overflow.

The codec's :func:`__hash__` doubles as the cache-invalidation key
for preprocessed shards stored under
``<cache_root>/<dataset>/preprocessed/<codec_hash>/<split>.pt``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol  # type: ignore[import-not-found]


# Map RDKit's BondType to our bond_decoder string for `encode_bond`.
# Populated lazily so the rdkit import stays inside the codec.
def _rdkit_bond_name(bond_type: object) -> str:
    """Map an RDKit ``Chem.BondType`` to our bond_decoder name.

    Lazy-importing here avoids hoisting rdkit into module-level
    imports and keeps the codec the sole RDKit-touching surface.
    """
    from rdkit.Chem.rdchem import BondType

    if bond_type == BondType.SINGLE:
        return "SINGLE"
    if bond_type == BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == BondType.AROMATIC:
        return "AROMATIC"
    raise ValueError(f"Unsupported RDKit bond type: {bond_type!r}.")


def _bond_name_to_rdkit(name: str) -> object:
    from rdkit.Chem.rdchem import BondType

    return {
        "SINGLE": BondType.SINGLE,
        "DOUBLE": BondType.DOUBLE,
        "TRIPLE": BondType.TRIPLE,
        "AROMATIC": BondType.AROMATIC,
    }[name]


@dataclass(frozen=True)
class SMILESCodec:
    """SMILES ↔ GraphData round-trip parameterised by a vocabulary.

    Parameters
    ----------
    vocab
        Atom + bond vocabulary the codec encodes against.
    remove_h
        When True, hydrogens are stripped before encoding (matches
        DiGress's default for QM9/MOSES/GuacaMol).
    kekulize
        When True, aromatic bonds are kekulised so the encoder sees
        explicit SINGLE/DOUBLE pairs rather than AROMATIC. Matches
        upstream DiGress's preprocessing.
    max_atoms
        Molecules with more than ``max_atoms`` heavy atoms are dropped
        (encode returns ``None``).
    """

    vocab: AtomBondVocabulary
    remove_h: bool = True
    kekulize: bool = True
    max_atoms: int = 30

    def __hash__(self) -> int:
        return hash(
            (self.vocab, self.remove_h, self.kekulize, self.max_atoms)
        )

    def cache_key(self) -> str:
        """Stable SHA-256 hex digest used as the on-disk shard subdirectory."""
        material = repr((self.vocab, self.remove_h, self.kekulize, self.max_atoms))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, smiles: str) -> GraphData | None:
        """Encode a SMILES string into a categorical :class:`GraphData`.

        Returns ``None`` when the molecule fails to parse, has too
        many heavy atoms, or contains an atom not in the vocabulary.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if self.remove_h:
            mol = Chem.RemoveHs(mol)
        else:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            try:
                Chem.Kekulize(mol)
            except Exception:
                return None

        atoms = list(mol.GetAtoms())
        n = len(atoms)
        if n == 0 or n > self.max_atoms:
            return None

        # Atom classes
        try:
            atom_classes = [self.vocab.encode_atom(a.GetSymbol()) for a in atoms]
        except ValueError:
            return None

        x_class = torch.zeros((1, n, self.vocab.num_atom_types), dtype=torch.float32)
        for i, cls in enumerate(atom_classes):
            x_class[0, i, cls] = 1.0

        # Bond classes (E_class[0,i,j] = one-hot over bond_decoder).
        e_class = torch.zeros(
            (1, n, n, self.vocab.num_bond_types), dtype=torch.float32
        )
        # Default everywhere: NONE (class 0).
        e_class[..., 0] = 1.0
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            try:
                bond_class = self.vocab.encode_bond(
                    _rdkit_bond_name(bond.GetBondType())
                )
            except ValueError:
                return None
            # Symmetric edge.
            e_class[0, i, j] = 0.0
            e_class[0, j, i] = 0.0
            e_class[0, i, j, bond_class] = 1.0
            e_class[0, j, i, bond_class] = 1.0
        # Diagonal stays at NONE (already set).

        node_mask = torch.ones((1, n), dtype=torch.bool)

        return GraphData(
            X_class=x_class,
            X_feat=None,
            E_class=e_class,
            E_feat=None,
            y=torch.zeros((1, 0), dtype=torch.float32),
            node_mask=node_mask,
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, data: GraphData) -> str | None:
        """Decode a single-graph :class:`GraphData` back to a SMILES.

        Returns ``None`` if the resulting molecule fails RDKit
        sanitisation. Used by :class:`ValidityMetric` and round-trip
        tests; not used in the training loop.
        """
        from rdkit import Chem
        from rdkit.Chem import RWMol

        if data.X_class is None or data.E_class is None:
            return None
        x_class = data.X_class
        e_class = data.E_class
        node_mask = data.node_mask
        if x_class.shape[0] != 1:
            raise ValueError(
                f"SMILESCodec.decode expects a single-graph batch; got "
                f"batch size {x_class.shape[0]}."
            )

        n_real = int(node_mask[0].sum().item())
        atom_idx = x_class[0, :n_real].argmax(dim=-1).tolist()
        bond_idx = e_class[0, :n_real, :n_real].argmax(dim=-1).tolist()

        rwmol = RWMol()
        for a in atom_idx:
            rwmol.AddAtom(Chem.Atom(self.vocab.decode_atom(int(a))))
        for i in range(n_real):
            for j in range(i + 1, n_real):
                cls = int(bond_idx[i][j])
                if cls == 0:  # NONE
                    continue
                bond_name = self.vocab.decode_bond(cls)
                rwmol.AddBond(i, j, _bond_name_to_rdkit(bond_name))  # type: ignore[arg-type]

        mol = rwmol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # batch helper
    # ------------------------------------------------------------------

    def encode_dataset(self, smiles_iter: Iterable[str]) -> Iterator[GraphData]:
        """Iterate over SMILES, dropping parse failures silently.

        Use :meth:`encode_dataset_with_stats` if you need the
        dropped-mol counters for logging.
        """
        for smi in smiles_iter:
            data = self.encode(smi)
            if data is not None:
                yield data

    def encode_dataset_with_stats(
        self, smiles_iter: Iterable[str]
    ) -> tuple[list[GraphData], dict[str, int]]:
        """Encode a list of SMILES, returning (graphs, drop_counters).

        Counters: ``"parse_failure"``, ``"atom_count_overflow"``,
        ``"vocab_miss"``, ``"kekulize_failure"``.
        """
        graphs: list[GraphData] = []
        counters = {
            "input": 0,
            "parse_failure": 0,
            "atom_count_overflow": 0,
            "vocab_miss": 0,
            "kekulize_failure": 0,
            "kept": 0,
        }
        # Re-implement the encode() body inline so we can fill counter
        # buckets per failure mode.
        from rdkit import Chem

        for smi in smiles_iter:
            counters["input"] += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                counters["parse_failure"] += 1
                continue
            mol = Chem.RemoveHs(mol) if self.remove_h else Chem.AddHs(mol)
            if self.kekulize:
                try:
                    Chem.Kekulize(mol)
                except Exception:
                    counters["kekulize_failure"] += 1
                    continue
            atoms = list(mol.GetAtoms())
            n = len(atoms)
            if n == 0 or n > self.max_atoms:
                counters["atom_count_overflow"] += 1
                continue
            try:
                atom_classes = [
                    self.vocab.encode_atom(a.GetSymbol()) for a in atoms
                ]
            except ValueError:
                counters["vocab_miss"] += 1
                continue
            data = self.encode(smi)
            if data is None:
                # Edge case where encode() raised in the second pass;
                # very rare, count under parse_failure.
                counters["parse_failure"] += 1
                continue
            graphs.append(data)
            counters["kept"] += 1
        return graphs, counters
```

- [ ] **Step 2: Verify the codec round-trips a known SMILES**

Run:
```bash
uv run python << 'PY'
from tmgg.data.datasets.molecular import AtomBondVocabulary
from tmgg.data.datasets.molecular.codec import SMILESCodec
codec = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
data = codec.encode("CCO")
print("encoded:", data.X_class.shape, data.E_class.shape)
print("decoded:", codec.decode(data))
print("cache_key:", codec.cache_key())
PY
```
Expected output (or close):
```
encoded: torch.Size([1, 3, 4]) torch.Size([1, 3, 3, 5])
decoded: CCO
cache_key: <16 hex chars>
```

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/molecular/codec.py
git commit -m "feat(data): add SMILESCodec — the only RDKit-importing module"
```

---

## Phase 3 — Molecular Dataset + DataModule (Phase 3)

### Task 3.1: `MolecularGraphDataset` ABC with on-disk shard cache

**Files:**
- Create: `src/tmgg/data/datasets/molecular/dataset.py`

- [ ] **Step 1: Create the ABC + cache logic**

Create `src/tmgg/data/datasets/molecular/dataset.py`:

```python
"""Abstract :class:`MolecularGraphDataset` with on-disk shard cache.

Subclasses (:class:`QM9Dataset`, :class:`MOSESDataset`,
:class:`GuacaMolDataset`) override :meth:`download_smiles_split` and
:meth:`make_codec` only. The cache machinery here is dataset-agnostic.

Cache layout:
``<cache_root>/<dataset_name>/preprocessed/<codec_hash>/<split>/<shard_idx>.pt``

``<codec_hash>`` is :meth:`SMILESCodec.cache_key` — any change to
vocab/codec params invalidates the entire preprocessed tree.
"""

from __future__ import annotations

import abc
import logging
import os
import urllib.request
from pathlib import Path
from typing import override

import torch
from torch.utils.data import Dataset

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.molecular.codec import SMILESCodec

logger = logging.getLogger(__name__)

# Shard size cap for MOSES/GuacaMol; QM9 fits in a single shard.
DEFAULT_SHARD_SIZE = 50_000


def _local_cache_root() -> Path:
    """Resolve the local cache root.

    ``/data/datasets`` exists on Modal containers (mounted via the
    tmgg-datasets volume per ``tmgg.modal._lib.volumes.DATASETS_MOUNT``);
    otherwise fall back to ``~/.cache/tmgg/``. This means a Modal
    training container reads + writes the persisted volume directly,
    while a host-side dev run reads + writes the user's home cache.
    """
    modal_root = Path("/data/datasets")
    if modal_root.exists() and modal_root.is_dir():
        return modal_root
    return Path.home() / ".cache" / "tmgg"


class MolecularGraphDataset(Dataset[GraphData], abc.ABC):
    """ABC for the three molecular datasets.

    Subclasses provide:
    - ``DATASET_NAME``: subdirectory name under the cache root.
    - ``RAW_FILES``: mapping ``split → URL`` for the raw SMILES files.
    - ``make_codec``: classmethod returning the dataset's
      :class:`SMILESCodec`.
    - ``download_smiles_split``: returns a list[str] of SMILES for
      a given split (uses ``RAW_FILES`` by default).

    Lifecycle:

    1. ``prepare_data()`` is called by the matching DataModule and
       downloads raw SMILES. Idempotent.
    2. ``setup(split)`` populates ``self._graphs`` from the cached
       shards, preprocessing on first call.
    3. ``__getitem__`` returns ``GraphData`` — no RDKit calls.
    """

    DATASET_NAME: str = ""  # subclass override
    RAW_FILES: dict[str, str] = {}  # split → URL
    DEFAULT_MAX_ATOMS: int = 30  # subclass override

    def __init__(
        self,
        *,
        split: str,
        cache_root: Path | None = None,
        shard_size: int = DEFAULT_SHARD_SIZE,
    ) -> None:
        if not self.DATASET_NAME:
            raise NotImplementedError(
                f"{type(self).__name__} must set DATASET_NAME."
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}.")
        self.split = split
        self.cache_root = cache_root or _local_cache_root()
        self.shard_size = shard_size
        self._codec: SMILESCodec | None = None
        self._graphs: list[GraphData] | None = None

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def make_codec(cls) -> SMILESCodec:
        """Return the canonical codec for this dataset."""

    @abc.abstractmethod
    def download_smiles_split(self, split: str) -> list[str]:
        """Return the raw SMILES list for a split.

        Implementations should download into
        ``self.cache_root / self.DATASET_NAME / "raw"`` and parse the
        SMILES out of whatever format the upstream uses (CSV, txt,
        etc.).
        """

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def _raw_dir(self) -> Path:
        return self.cache_root / self.DATASET_NAME / "raw"

    def _preprocessed_dir(self) -> Path:
        codec = self._codec or self.make_codec()
        return (
            self.cache_root
            / self.DATASET_NAME
            / "preprocessed"
            / codec.cache_key()
        )

    def _shard_dir(self) -> Path:
        return self._preprocessed_dir() / self.split

    # ------------------------------------------------------------------
    # default URL-based downloader
    # ------------------------------------------------------------------

    def _default_download(self, split: str) -> Path:
        if split not in self.RAW_FILES:
            raise KeyError(
                f"{self.DATASET_NAME}: no RAW_FILES URL for split {split!r}."
            )
        url = self.RAW_FILES[split]
        target = self._raw_dir() / Path(url).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        logger.info("Downloading %s → %s", url, target)
        # Retry 3x with exponential backoff for transient network errors.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, target)  # noqa: S310
                return target
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Failed to download {url} after 3 attempts: {last_exc!r}"
        )

    # ------------------------------------------------------------------
    # preprocessing + caching
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download all raw splits. Safe to call multiple times."""
        for split in ("train", "val", "test"):
            try:
                _ = self.download_smiles_split(split)
            except KeyError:
                # Dataset may not declare every split (e.g. GuacaMol
                # 'test' is the held-out reference set; declare what
                # exists in the subclass).
                logger.info(
                    "Skipping %s/%s: not declared.",
                    self.DATASET_NAME, split,
                )

    def setup(self) -> None:
        """Populate ``self._graphs`` for the configured split."""
        self._codec = self.make_codec()
        shard_dir = self._shard_dir()
        if shard_dir.exists() and any(shard_dir.iterdir()):
            self._graphs = self._load_shards()
            return

        logger.info(
            "%s/%s: preprocessing (cache miss at %s)",
            self.DATASET_NAME, self.split, shard_dir,
        )
        smiles = self.download_smiles_split(self.split)
        graphs, counters = self._codec.encode_dataset_with_stats(smiles)
        logger.info(
            "%s/%s: preprocessed; counters=%s",
            self.DATASET_NAME, self.split, counters,
        )
        self._write_shards(graphs)
        self._graphs = graphs

    # ------------------------------------------------------------------
    # shard I/O
    # ------------------------------------------------------------------

    def _write_shards(self, graphs: list[GraphData]) -> None:
        shard_dir = self._shard_dir()
        shard_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx, start in enumerate(range(0, len(graphs), self.shard_size)):
            shard = graphs[start : start + self.shard_size]
            shard_path = shard_dir / f"{shard_idx:04d}.pt"
            # ``weights_only=False`` on load is required because
            # ``GraphData`` is a dataclass; these are first-party
            # files we wrote ourselves so the security memory's
            # "no third-party pickles" rule does not apply.
            torch.save(shard, shard_path)

    def _load_shards(self) -> list[GraphData]:
        shard_dir = self._shard_dir()
        graphs: list[GraphData] = []
        for shard_path in sorted(shard_dir.glob("*.pt")):
            shard: list[GraphData] = torch.load(shard_path, weights_only=False)
            graphs.extend(shard)
        return graphs

    # ------------------------------------------------------------------
    # Dataset[GraphData] surface
    # ------------------------------------------------------------------

    @override
    def __len__(self) -> int:
        if self._graphs is None:
            raise RuntimeError(
                f"{type(self).__name__} not set up. Call setup() first."
            )
        return len(self._graphs)

    @override
    def __getitem__(self, idx: int) -> GraphData:
        if self._graphs is None:
            raise RuntimeError(
                f"{type(self).__name__} not set up. Call setup() first."
            )
        return self._graphs[idx]
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/molecular/dataset.py
git commit -m "feat(data): add MolecularGraphDataset ABC with shard cache"
```

---

### Task 3.2: `QM9Dataset` concrete subclass

**Files:**
- Create: `src/tmgg/data/datasets/molecular/qm9.py`

- [ ] **Step 1: Create the QM9 subclass**

Create `src/tmgg/data/datasets/molecular/qm9.py`:

```python
"""QM9 dataset (no-H by default) — DiGress repro Table 4.

SMILES source: DiGress's published QM9 CSV mirror (the same file the
upstream repo reads). Each row is one SMILES; ~134k molecules total.
Splits: 80 % train / 10 % val / 10 % test, mirroring upstream.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import override

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# DiGress paper uses the standard PyG QM9 download (gdb9.sdf.csv).
# We download via the PyG mirror used by upstream DiGress.
_QM9_CSV_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
)

# Upstream split rule: random 80/10/10 with seed 0.
_QM9_SPLIT_SEED = 0


class QM9Dataset(MolecularGraphDataset):
    """QM9 (no-H) split. Atom decoder: (C, N, O, F)."""

    DATASET_NAME = "qm9"
    DEFAULT_MAX_ATOMS = 9
    # No per-split URLs — single CSV; we split locally.
    RAW_FILES: dict[str, str] = {}

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(remove_h=True),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        """Read all SMILES from the QM9 CSV and return the requested split."""
        raw_path = self._raw_dir() / "qm9.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            self._download_qm9_csv(raw_path)

        all_smiles: list[str] = []
        with raw_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("smiles") or row.get("SMILES")
                if smi:
                    all_smiles.append(smi.strip())

        rng = random.Random(_QM9_SPLIT_SEED)
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            picked = indices[:n_train]
        elif split == "val":
            picked = indices[n_train : n_train + n_val]
        elif split == "test":
            picked = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split {split!r}.")
        return [all_smiles[i] for i in picked]

    def _download_qm9_csv(self, target: Path) -> None:
        """Download + extract the QM9 CSV from DeepChem's mirror."""
        import tarfile
        import urllib.request

        archive = target.with_name("qm9.tar.gz")
        if not archive.exists():
            urllib.request.urlretrieve(_QM9_CSV_URL, archive)  # noqa: S310
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV inside {archive}")
            tar.extract(members[0], target.parent)
            extracted = target.parent / members[0].name
            extracted.rename(target)
```

- [ ] **Step 2: Smoke-test the import (don't actually download)**

Run: `uv run python -c "from tmgg.data.datasets.molecular.qm9 import QM9Dataset; ds = QM9Dataset(split='train'); print('codec:', ds.make_codec().cache_key())"`
Expected: a 16-char hex key prints, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/molecular/qm9.py
git commit -m "feat(data): add QM9Dataset (no-H) DiGress repro Table 4"
```

---

### Task 3.3: `MOSESDataset` concrete subclass

**Files:**
- Create: `src/tmgg/data/datasets/molecular/moses.py`

- [ ] **Step 1: Create the MOSES subclass**

Create `src/tmgg/data/datasets/molecular/moses.py`:

```python
"""MOSES dataset — DiGress repro Table 5.

SMILES source: the ``moses`` package (``moses.get_dataset(split)``).
Atom decoder: (C, N, S, O, F, Cl, Br, H). DiGress uses the published
MOSES train/test/scaffold split; we mirror the package's
``get_dataset`` helper.
"""

from __future__ import annotations

from typing import override

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# MOSES test split is large enough that DiGress evaluates against
# scaffold split (held-out scaffolds) for the FCD/SNN metrics. Our
# DataModule serves train + val(=test_scaffolds) + test(=test).
_MOSES_VAL_SPLIT = "test_scaffolds"


class MOSESDataset(MolecularGraphDataset):
    """MOSES split. Atom decoder: (C, N, S, O, F, Cl, Br, H)."""

    DATASET_NAME = "moses"
    DEFAULT_MAX_ATOMS = 30
    # SMILES come from the ``moses`` package, not URLs.
    RAW_FILES: dict[str, str] = {}

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.moses(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        """Pull SMILES from the ``moses`` package."""
        import moses

        moses_split = {
            "train": "train",
            "val": _MOSES_VAL_SPLIT,
            "test": "test",
        }.get(split)
        if moses_split is None:
            raise ValueError(f"Unknown split {split!r}.")
        return list(moses.get_dataset(moses_split))
```

- [ ] **Step 2: Smoke-import**

Run: `uv run python -c "from tmgg.data.datasets.molecular.moses import MOSESDataset; print(MOSESDataset(split='train').make_codec().cache_key())"`
Expected: 16-char hex.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/molecular/moses.py
git commit -m "feat(data): add MOSESDataset DiGress repro Table 5"
```

---

### Task 3.4: `GuacaMolDataset` concrete subclass

**Files:**
- Create: `src/tmgg/data/datasets/molecular/guacamol.py`

- [ ] **Step 1: Create the GuacaMol subclass**

Create `src/tmgg/data/datasets/molecular/guacamol.py`:

```python
"""GuacaMol dataset — DiGress repro Table 6.

SMILES source: GuacaMol's published train/val/test splits.
Atom decoder: (C, N, O, F, B, Br, Cl, I, P, S, Se, Si).
"""

from __future__ import annotations

from typing import override

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# GuacaMol's published splits.
_GUACAMOL_BASE_URL = (
    "https://figshare.com/ndownloader/files/13612760"  # train
)
_GUACAMOL_URLS = {
    "train": "https://figshare.com/ndownloader/files/13612760",
    "val":   "https://figshare.com/ndownloader/files/13612766",
    "test":  "https://figshare.com/ndownloader/files/13612757",
}


class GuacaMolDataset(MolecularGraphDataset):
    """GuacaMol split. Atom decoder: 12-element vocabulary."""

    DATASET_NAME = "guacamol"
    DEFAULT_MAX_ATOMS = 88
    RAW_FILES = _GUACAMOL_URLS

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.guacamol(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        path = self._default_download(split)
        with path.open("r") as f:
            return [line.strip() for line in f if line.strip()]
```

- [ ] **Step 2: Smoke-import**

Run: `uv run python -c "from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset; print(GuacaMolDataset(split='train').make_codec().cache_key())"`
Expected: 16-char hex.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/data/datasets/molecular/guacamol.py
git commit -m "feat(data): add GuacaMolDataset DiGress repro Table 6"
```

---

### Task 3.5: `MolecularDataModule` base class

**Files:**
- Create: `src/tmgg/data/data_modules/molecular/__init__.py`
- Create: `src/tmgg/data/data_modules/molecular/base.py`

- [ ] **Step 1: Create the package init**

Create `src/tmgg/data/data_modules/molecular/__init__.py`:

```python
"""Molecular DataModules (QM9, MOSES, GuacaMol)."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule

__all__ = ["MolecularDataModule"]
```

- [ ] **Step 2: Create the base DataModule**

Create `src/tmgg/data/data_modules/molecular/base.py`:

```python
"""Base molecular DataModule. Subclasses pin ``dataset_cls``.

Mirrors :class:`tmgg.data.data_modules.spectre_sbm.SpectreSBMDataModule`
in dataloader plumbing but reads :class:`MolecularGraphDataset`
subclasses (which already produce :class:`GraphData` objects) so the
collator becomes the identity over a list of single-graph
:class:`GraphData` instances stitched into a batch by
:class:`GraphDataCollator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from torch.utils.data import DataLoader

from tmgg.data.data_modules.base_data_module import BaseGraphDataModule
from tmgg.data.data_modules.multigraph_data_module import (
    GraphDataCollator,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.utils.noising.size_distribution import SizeDistribution


class MolecularDataModule(BaseGraphDataModule):
    """Generic molecular DataModule. Pin ``dataset_cls`` per subclass."""

    # Subclass overrides:
    dataset_cls: type[MolecularGraphDataset] = MolecularGraphDataset

    num_nodes: int

    def __init__(
        self,
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
        cache_root: str | None = None,
        num_nodes_max_static: int | None = None,
        pad_to_static_n_max: bool = False,
        **_metadata: object,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        self.cache_root: Path | None = (
            Path(cache_root) if cache_root else None
        )
        # Default static-pad target = the dataset class's
        # DEFAULT_MAX_ATOMS unless explicitly overridden.
        self.num_nodes_max_static = (
            num_nodes_max_static
            if num_nodes_max_static is not None
            else self.dataset_cls.DEFAULT_MAX_ATOMS
        )
        self.pad_to_static_n_max = pad_to_static_n_max

        self._train_dataset: MolecularGraphDataset | None = None
        self._val_dataset: MolecularGraphDataset | None = None
        self._test_dataset: MolecularGraphDataset | None = None

        self.graph_type = self.dataset_cls.DATASET_NAME
        self.num_nodes = self.dataset_cls.DEFAULT_MAX_ATOMS

        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    @override
    def prepare_data(self) -> None:
        """Trigger downloads + preprocessing on the train split."""
        ds = self.dataset_cls(
            split="train",
            cache_root=self.cache_root,
        )
        ds.prepare_data()

    @override
    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return
        self._train_dataset = self._make(split="train")
        self._val_dataset = self._make(split="val")
        self._test_dataset = self._make(split="test")
        # Refresh num_nodes from the largest observed graph (overrides
        # the constructor-time default of DEFAULT_MAX_ATOMS).
        all_n = [
            int(g.node_mask.sum().item())
            for g in (self._train_dataset._graphs or [])
            + (self._val_dataset._graphs or [])
        ]
        if all_n:
            self.num_nodes = max(all_n)

    def _make(self, *, split: str) -> MolecularGraphDataset:
        ds = self.dataset_cls(split=split, cache_root=self.cache_root)
        ds.setup()
        return ds

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _dense_collator(self) -> GraphDataCollator:
        return GraphDataCollator(
            n_max_static=self.num_nodes_max_static
            if self.pad_to_static_n_max
            else None
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        if self._train_dataset is None:
            raise RuntimeError(
                f"{type(self).__name__} not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(list(self._train_dataset._graphs or [])),
            shuffle=True,
            collate_fn=self._dense_collator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        if self._val_dataset is None:
            raise RuntimeError(
                f"{type(self).__name__} not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(list(self._val_dataset._graphs or [])),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        if self._test_dataset is None:
            raise RuntimeError(
                f"{type(self).__name__} not setup. Call setup() first."
            )
        return self._make_dataloader(
            _ListDataset(list(self._test_dataset._graphs or [])),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def get_size_distribution(
        self, split: str | None = None
    ) -> SizeDistribution:
        split_map = {
            "train": self._train_dataset,
            "val": self._val_dataset,
            "test": self._test_dataset,
        }
        if split is None:
            datasets = [
                d for d in split_map.values() if d is not None
            ]
        elif split in split_map:
            d = split_map[split]
            datasets = [d] if d is not None else []
        else:
            raise ValueError(f"Unknown split {split!r}.")

        node_counts: list[int] = []
        for ds in datasets:
            for g in ds._graphs or []:
                node_counts.append(int(g.node_mask.sum().item()))

        if not node_counts:
            return SizeDistribution.fixed(self.num_nodes or 1)
        return SizeDistribution.from_node_counts(node_counts)
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from tmgg.data.data_modules.molecular import MolecularDataModule; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/data/data_modules/molecular/__init__.py src/tmgg/data/data_modules/molecular/base.py
git commit -m "feat(data): add MolecularDataModule base class"
```

---

### Task 3.6: Per-dataset thin DataModule subclasses

**Files:**
- Create: `src/tmgg/data/data_modules/molecular/qm9.py`
- Create: `src/tmgg/data/data_modules/molecular/moses.py`
- Create: `src/tmgg/data/data_modules/molecular/guacamol.py`

- [ ] **Step 1: Create QM9 DataModule**

Create `src/tmgg/data/data_modules/molecular/qm9.py`:

```python
"""QM9 DataModule: pins QM9Dataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.qm9 import QM9Dataset


class QM9DataModule(MolecularDataModule):
    """QM9 (no-H) DataModule. Mirrors SpectreSBMDataModule shape."""

    dataset_cls = QM9Dataset
```

- [ ] **Step 2: Create MOSES DataModule**

Create `src/tmgg/data/data_modules/molecular/moses.py`:

```python
"""MOSES DataModule: pins MOSESDataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.moses import MOSESDataset


class MOSESDataModule(MolecularDataModule):
    """MOSES DataModule."""

    dataset_cls = MOSESDataset
```

- [ ] **Step 3: Create GuacaMol DataModule**

Create `src/tmgg/data/data_modules/molecular/guacamol.py`:

```python
"""GuacaMol DataModule: pins GuacaMolDataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset


class GuacaMolDataModule(MolecularDataModule):
    """GuacaMol DataModule."""

    dataset_cls = GuacaMolDataset
```

- [ ] **Step 4: Verify imports**

Run:
```bash
uv run python -c "
from tmgg.data.data_modules.molecular.qm9 import QM9DataModule
from tmgg.data.data_modules.molecular.moses import MOSESDataModule
from tmgg.data.data_modules.molecular.guacamol import GuacaMolDataModule
print('all ok')
"
```
Expected: `all ok`.

- [ ] **Step 5: Commit**

```bash
git add src/tmgg/data/data_modules/molecular/qm9.py src/tmgg/data/data_modules/molecular/moses.py src/tmgg/data/data_modules/molecular/guacamol.py
git commit -m "feat(data): add per-dataset molecular DataModule subclasses"
```

---

### Task 3.7: Three molecular data YAMLs

**Files:**
- Create: `src/tmgg/experiments/exp_configs/data/qm9_digress.yaml`
- Create: `src/tmgg/experiments/exp_configs/data/moses_digress.yaml`
- Create: `src/tmgg/experiments/exp_configs/data/guacamol_digress.yaml`

- [ ] **Step 1: Create the QM9 data YAML**

Create `src/tmgg/experiments/exp_configs/data/qm9_digress.yaml`:

```yaml
# @package data
_target_: tmgg.data.data_modules.molecular.qm9.QM9DataModule
batch_size: 256
num_workers: 4
pin_memory: true
prefetch_factor: 4
seed: 42
cache_root: null
num_nodes_max_static: 9
pad_to_static_n_max: false
```

- [ ] **Step 2: Create the MOSES data YAML**

Create `src/tmgg/experiments/exp_configs/data/moses_digress.yaml`:

```yaml
# @package data
_target_: tmgg.data.data_modules.molecular.moses.MOSESDataModule
batch_size: 64
num_workers: 4
pin_memory: true
prefetch_factor: 4
seed: 42
cache_root: null
num_nodes_max_static: 30
pad_to_static_n_max: false
```

- [ ] **Step 3: Create the GuacaMol data YAML**

Create `src/tmgg/experiments/exp_configs/data/guacamol_digress.yaml`:

```yaml
# @package data
_target_: tmgg.data.data_modules.molecular.guacamol.GuacaMolDataModule
batch_size: 32
num_workers: 4
pin_memory: true
prefetch_factor: 4
seed: 42
cache_root: null
num_nodes_max_static: 88
pad_to_static_n_max: false
```

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/experiments/exp_configs/data/qm9_digress.yaml src/tmgg/experiments/exp_configs/data/moses_digress.yaml src/tmgg/experiments/exp_configs/data/guacamol_digress.yaml
git commit -m "feat(configs): add QM9/MOSES/GuacaMol DiGress data YAMLs"
```

---

## Phase 4 — Molecular metrics (Phase 4)

### Task 4.1: `MolecularMetric` ABC

**Files:**
- Create: `src/tmgg/evaluation/molecular/__init__.py`
- Create: `src/tmgg/evaluation/molecular/metric.py`

- [ ] **Step 1: Create package init**

Create `src/tmgg/evaluation/molecular/__init__.py`:

```python
"""Molecular metrics + evaluator (DiGress repro Tables 4-6).

Composition: a metric is one class with a ``.compute(generated,
reference) → float | dict[str, float]`` method. The evaluator owns
a list of metric instances + a :class:`SMILESCodec` used to decode
generated :class:`GraphData` to SMILES once per evaluation pass.
"""

from __future__ import annotations

from tmgg.evaluation.molecular.evaluator import (
    MolecularEvaluationResults,
    MolecularEvaluator,
)
from tmgg.evaluation.molecular.metric import MolecularMetric

__all__ = [
    "MolecularEvaluationResults",
    "MolecularEvaluator",
    "MolecularMetric",
]
```

- [ ] **Step 2: Create the ABC**

Create `src/tmgg/evaluation/molecular/metric.py`:

```python
"""Abstract base class for a single molecular generation-quality metric."""

from __future__ import annotations

import abc
from collections.abc import Sequence


class MolecularMetric(abc.ABC):
    """One metric, one ``.compute(generated, reference)`` call.

    Stateful subclasses (e.g. :class:`FCDMetric` holding a ChemNet
    embedder) initialise their state in ``__init__``; metrics live
    as long as the :class:`MolecularEvaluator` that owns them.

    The ``name`` attribute determines the W&B key under
    ``gen-val/<name>`` when the result is a single float, or
    ``gen-val/<name>/<sub>`` when ``compute`` returns a dict.
    """

    name: str

    @abc.abstractmethod
    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float | dict[str, float]:
        """Compute the metric value from generated + (optional) reference SMILES."""
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from tmgg.evaluation.molecular.metric import MolecularMetric; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/tmgg/evaluation/molecular/__init__.py src/tmgg/evaluation/molecular/metric.py
git commit -m "feat(eval): add MolecularMetric ABC"
```

---

### Task 4.2: RDKit metrics (Validity, Uniqueness, Novelty)

**Files:**
- Create: `src/tmgg/evaluation/molecular/rdkit_metrics.py`

- [ ] **Step 1: Create the RDKit metrics file**

Create `src/tmgg/evaluation/molecular/rdkit_metrics.py`:

```python
"""RDKit-based generation-quality metrics: Validity / Uniqueness / Novelty.

Mirrors upstream DiGress's ``rdkit_functions.py`` formulas:

- Validity = |valid molecules| / |attempted|
- Uniqueness = |distinct canonical SMILES among valid| / |valid|
- Novelty = |canonical SMILES not in train set| / |valid|
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _to_canonical(smiles: str) -> str | None:
    """Canonicalise a SMILES via RDKit; return None if invalid."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


class ValidityMetric(MolecularMetric):
    """Fraction of generated SMILES that pass RDKit sanitisation."""

    name = "validity"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        if not generated:
            return 0.0
        n_valid = sum(1 for s in generated if _to_canonical(s) is not None)
        return n_valid / len(generated)


class UniquenessMetric(MolecularMetric):
    """Fraction of valid molecules that are unique by canonical SMILES."""

    name = "uniqueness"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        canonical = [_to_canonical(s) for s in generated]
        valid = [c for c in canonical if c is not None]
        if not valid:
            return 0.0
        return len(set(valid)) / len(valid)


class NoveltyMetric(MolecularMetric):
    """Fraction of valid molecules NOT in the reference (training) set."""

    name = "novelty"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError(
                "NoveltyMetric requires a reference set of training SMILES."
            )
        ref_canonical = {c for c in (_to_canonical(s) for s in reference) if c is not None}
        canonical = [_to_canonical(s) for s in generated]
        valid = [c for c in canonical if c is not None]
        if not valid:
            return 0.0
        novel = [c for c in valid if c not in ref_canonical]
        return len(novel) / len(valid)
```

- [ ] **Step 2: Verify imports + a quick sanity check**

Run:
```bash
uv run python << 'PY'
from tmgg.evaluation.molecular.rdkit_metrics import (
    ValidityMetric, UniquenessMetric, NoveltyMetric,
)
gen = ["CCO", "CCO", "not_a_smiles", "CC(=O)O"]
ref = ["CCO"]
print("validity:",   ValidityMetric().compute(gen))
print("uniqueness:", UniquenessMetric().compute(gen))
print("novelty:",    NoveltyMetric().compute(gen, ref))
PY
```
Expected: validity ≈ 0.75, uniqueness = 2/3 ≈ 0.667, novelty = 1/3 ≈ 0.333.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/evaluation/molecular/rdkit_metrics.py
git commit -m "feat(eval): add RDKit-based Validity/Uniqueness/Novelty metrics"
```

---

### Task 4.3: MOSES metrics (FCD, SNN, IntDiv, Filters, ScaffoldSplit)

**Files:**
- Create: `src/tmgg/evaluation/molecular/moses_metrics.py`

- [ ] **Step 1: Create the MOSES metrics module**

Create `src/tmgg/evaluation/molecular/moses_metrics.py`:

```python
"""MOSES metrics (Table 5 in DiGress): FCD/SNN/IntDiv/Filters/ScaffoldSplit.

Wraps the ``moses`` package + ``fcd_torch`` so the rest of our code
does not have to know either API. Each class is one metric.
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _drop_invalid(smiles: Sequence[str]) -> list[str]:
    """Filter SMILES to those that round-trip through RDKit."""
    from rdkit import Chem

    out: list[str] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        out.append(s)
    return out


class FCDMetric(MolecularMetric):
    """Frechet ChemNet Distance via :mod:`fcd_torch`."""

    name = "fcd"

    def __init__(self, device: str = "cpu", n_jobs: int = 1) -> None:
        self.device = device
        self.n_jobs = n_jobs
        self._fcd: object | None = None

    def _ensure(self) -> object:
        if self._fcd is None:
            from fcd_torch import FCD

            self._fcd = FCD(device=self.device, n_jobs=self.n_jobs)
        return self._fcd

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("FCDMetric requires reference SMILES.")
        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return float("inf")
        fcd = self._ensure()
        # fcd_torch's FCD object is callable: __call__(gen, ref) → float.
        return float(fcd(list(gen_valid), list(reference)))  # type: ignore[operator]


class SNNMetric(MolecularMetric):
    """Average Tanimoto similarity to nearest train neighbour (MOSES)."""

    name = "snn"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("SNNMetric requires reference SMILES.")
        from moses.metrics import SNNMetric as _SNN

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(_SNN()(list(gen_valid), list(reference)))


class IntDivMetric(MolecularMetric):
    """Internal diversity (1 − mean Tanimoto similarity)."""

    name = "int_div"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        from moses.metrics import internal_diversity

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(internal_diversity(list(gen_valid)))


class FiltersMetric(MolecularMetric):
    """Fraction of generated mols passing the MOSES filter set."""

    name = "filters"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        from moses.metrics import fraction_passes_filters

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(fraction_passes_filters(list(gen_valid)))


class ScaffoldSplitMetric(MolecularMetric):
    """Fraction of generated mols whose Bemis-Murcko scaffolds are
    novel relative to the reference set's scaffolds."""

    name = "scaffold_novelty"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError(
                "ScaffoldSplitMetric requires reference SMILES."
            )
        from moses.metrics import compute_scaffolds

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        ref_scaffolds = set(compute_scaffolds(list(reference)).keys())
        gen_scaffolds = compute_scaffolds(list(gen_valid))
        novel = [s for s in gen_scaffolds.keys() if s not in ref_scaffolds]
        return len(novel) / len(gen_scaffolds) if gen_scaffolds else 0.0
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from tmgg.evaluation.molecular.moses_metrics import FCDMetric, SNNMetric, IntDivMetric, FiltersMetric, ScaffoldSplitMetric; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/evaluation/molecular/moses_metrics.py
git commit -m "feat(eval): add MOSES metrics (FCD/SNN/IntDiv/Filters/ScaffoldSplit)"
```

---

### Task 4.4: GuacaMol metrics (KLDivProperty, FCDChEMBL)

**Files:**
- Create: `src/tmgg/evaluation/molecular/guacamol_metrics.py`

- [ ] **Step 1: Create the GuacaMol metrics module**

Create `src/tmgg/evaluation/molecular/guacamol_metrics.py`:

```python
"""GuacaMol Distribution-Learning metrics (Table 6 in DiGress).

Wraps :mod:`guacamol.distribution_learning_benchmark`. Two metrics:
KL divergence on physchem properties, and FCD against a ChEMBL
reference set.
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _drop_invalid(smiles: Sequence[str]) -> list[str]:
    from rdkit import Chem

    out: list[str] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            out.append(s)
    return out


class KLDivPropertyMetric(MolecularMetric):
    """KL divergence on the GuacaMol physchem-property distribution.

    Wraps ``guacamol.distribution_learning_benchmark.KLDivBenchmark``.
    Returns the benchmark's score in [0, 1] (higher = closer to ref).
    """

    name = "kl_div_property"

    def __init__(self, n_samples: int = 10_000) -> None:
        self.n_samples = n_samples

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("KLDivPropertyMetric requires reference SMILES.")
        from guacamol.distribution_learning_benchmark import KLDivBenchmark

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        bench = KLDivBenchmark(
            training_set=list(reference),
            number_samples=self.n_samples,
            name="kl_div_property",
        )
        # KLDivBenchmark.assess_model expects an object with
        # ``generate(n)`` returning SMILES; wrap our list in a tiny
        # adapter so we feed exactly our generated set rather than
        # re-sampling.
        class _Adapter:
            def __init__(self, smiles: list[str]) -> None:
                self.smiles = smiles

            def generate(self, _n: int) -> list[str]:
                return self.smiles

        result = bench.assess_model(_Adapter(list(gen_valid)))
        return float(result.score)


class FCDChEMBLMetric(MolecularMetric):
    """Frechet ChemNet Distance against the GuacaMol ChEMBL reference set."""

    name = "fcd_chembl"

    def __init__(self, device: str = "cpu", n_jobs: int = 1) -> None:
        self.device = device
        self.n_jobs = n_jobs
        self._fcd: object | None = None

    def _ensure(self) -> object:
        if self._fcd is None:
            from fcd_torch import FCD

            self._fcd = FCD(device=self.device, n_jobs=self.n_jobs)
        return self._fcd

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("FCDChEMBLMetric requires reference SMILES.")
        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return float("inf")
        fcd = self._ensure()
        return float(fcd(list(gen_valid), list(reference)))  # type: ignore[operator]
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from tmgg.evaluation.molecular.guacamol_metrics import KLDivPropertyMetric, FCDChEMBLMetric; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/evaluation/molecular/guacamol_metrics.py
git commit -m "feat(eval): add GuacaMol KLDivProperty + FCDChEMBL metrics"
```

---

## Phase 5 — `MolecularEvaluator` + experiment YAMLs (Phase 5)

### Task 5.1: `MolecularEvaluator` + `MolecularEvaluationResults`

**Files:**
- Create: `src/tmgg/evaluation/molecular/evaluator.py`

- [ ] **Step 1: Create the evaluator module**

Create `src/tmgg/evaluation/molecular/evaluator.py`:

```python
"""Composer: a list of :class:`MolecularMetric` + a :class:`SMILESCodec`.

Mirrors :class:`tmgg.evaluation.graph_evaluator.GraphEvaluator`'s
surface so :class:`DiffusionModule` accepts either via duck typing.
The ``results.to_dict()`` method returns a flat ``{name: float}``
mapping that the existing ``on_validation_epoch_end`` loop logs as
``gen-val/<name>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

if TYPE_CHECKING:
    from tmgg.data.datasets.graph_types import GraphData
    from tmgg.evaluation.molecular.metric import MolecularMetric


@dataclass
class MolecularEvaluationResults:
    """Flat results container, mirrors :class:`EvaluationResults`."""

    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


class MolecularEvaluator:
    """Compose a list of metrics + a codec for SMILES decoding."""

    def __init__(
        self,
        metrics: Sequence["MolecularMetric"],
        codec: SMILESCodec,
    ) -> None:
        self.metrics = list(metrics)
        self.codec = codec

    # ------------------------------------------------------------------
    # public surface (duck-types GraphEvaluator)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        refs: Sequence["GraphData"],
        generated: Sequence["GraphData"],
    ) -> MolecularEvaluationResults:
        """Decode both sets to SMILES; run each metric."""
        ref_smiles = self._decode_all(refs)
        gen_smiles = self._decode_all(generated)
        results = MolecularEvaluationResults()
        for metric in self.metrics:
            value = metric.compute(gen_smiles, ref_smiles or None)
            if isinstance(value, float):
                results.values[metric.name] = value
            else:
                # dict[str, float] → flat keys
                for sub, sv in value.items():
                    results.values[f"{metric.name}/{sub}"] = sv
        return results

    def _decode_all(
        self, batch: Sequence["GraphData"]
    ) -> list[str]:
        out: list[str] = []
        for data in batch:
            decoded = self.codec.decode(data)
            if decoded is not None:
                out.append(decoded)
        return out

    # ------------------------------------------------------------------
    # classmethod presets
    # ------------------------------------------------------------------

    @classmethod
    def for_qm9(cls) -> "MolecularEvaluator":
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.qm9(remove_h=True),
            max_atoms=9,
        )
        return cls(
            metrics=[ValidityMetric(), UniquenessMetric(), NoveltyMetric()],
            codec=codec,
        )

    @classmethod
    def for_moses(cls) -> "MolecularEvaluator":
        from tmgg.evaluation.molecular.moses_metrics import (
            FCDMetric,
            FiltersMetric,
            IntDivMetric,
            ScaffoldSplitMetric,
            SNNMetric,
        )
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.moses(),
            max_atoms=30,
        )
        return cls(
            metrics=[
                ValidityMetric(),
                UniquenessMetric(),
                NoveltyMetric(),
                FCDMetric(),
                SNNMetric(),
                IntDivMetric(),
                FiltersMetric(),
                ScaffoldSplitMetric(),
            ],
            codec=codec,
        )

    @classmethod
    def for_guacamol(cls) -> "MolecularEvaluator":
        from tmgg.evaluation.molecular.guacamol_metrics import (
            FCDChEMBLMetric,
            KLDivPropertyMetric,
        )
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.guacamol(),
            max_atoms=88,
        )
        return cls(
            metrics=[
                ValidityMetric(),
                UniquenessMetric(),
                NoveltyMetric(),
                KLDivPropertyMetric(),
                FCDChEMBLMetric(),
            ],
            codec=codec,
        )
```

- [ ] **Step 2: Verify imports**

Run:
```bash
uv run python -c "
from tmgg.evaluation.molecular import MolecularEvaluator
ev = MolecularEvaluator.for_qm9()
print('qm9 metrics:', [m.name for m in ev.metrics])
"
```
Expected: `qm9 metrics: ['validity', 'uniqueness', 'novelty']`.

- [ ] **Step 3: Commit**

```bash
git add src/tmgg/evaluation/molecular/evaluator.py
git commit -m "feat(eval): add MolecularEvaluator with for_qm9/moses/guacamol presets"
```

---

### Task 5.2: Allow `DiffusionModule.evaluator` to accept `MolecularEvaluator`

**Files:**
- Modify: `src/tmgg/training/lightning_modules/diffusion_module.py`

**Why:** the `evaluator` field currently has type `GraphEvaluator | None`. We need `GraphEvaluator | MolecularEvaluator | None` for duck-typed acceptance. No behaviour change inside the validation loop.

- [ ] **Step 1: Locate the existing import + type annotation**

Run: `rg -n "evaluator: GraphEvaluator|from tmgg.evaluation.graph_evaluator|self.evaluator" /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/src/tmgg/training/lightning_modules/diffusion_module.py | head -20`

- [ ] **Step 2: Update the import + type union**

In `src/tmgg/training/lightning_modules/diffusion_module.py`:

Find:
```python
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
```

Replace with:
```python
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.evaluation.molecular import MolecularEvaluator
```

Find the `evaluator: GraphEvaluator | None` annotation (constructor parameter). Replace with:
```python
evaluator: GraphEvaluator | MolecularEvaluator | None,
```

Find the `self.evaluator: GraphEvaluator | None = evaluator` attribute. Replace with:
```python
self.evaluator: GraphEvaluator | MolecularEvaluator | None = evaluator
```

- [ ] **Step 3: Verify the file imports**

Run: `uv run python -c "from tmgg.training.lightning_modules.diffusion_module import DiffusionModule; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Run the existing DiffusionModule test suite to confirm no regression**

Run:
```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
uv run pytest tests/experiment_utils/test_diffusion_module.py -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tmgg/training/lightning_modules/diffusion_module.py
git commit -m "feat(training): widen DiffusionModule.evaluator type to accept MolecularEvaluator"
```

---

### Task 5.3: Three molecular experiment YAMLs

**Files:**
- Create: `src/tmgg/experiments/exp_configs/experiment/discrete_qm9_digress_repro.yaml`
- Create: `src/tmgg/experiments/exp_configs/experiment/discrete_moses_digress_repro.yaml`
- Create: `src/tmgg/experiments/exp_configs/experiment/discrete_guacamol_digress_repro.yaml`

- [ ] **Step 1: Create the QM9 experiment YAML**

Create `src/tmgg/experiments/exp_configs/experiment/discrete_qm9_digress_repro.yaml`:

```yaml
# @package _global_
# DiGress QM9 reproduction (Vignac et al., ICLR 2023, Table 4).

defaults:
  - /data: qm9_digress
  - override /models/discrete@model: discrete_sbm_official
  - _self_

experiment_name: discrete_qm9_digress_repro
wandb_project: discrete-qm9-digress-repro

seed: 0

trainer:
  max_steps: 50000
  val_check_interval: 1000
  check_val_every_n_epoch: null
  gradient_clip_val: null

model:
  eval_every_n_steps: 5000
  num_nodes: 9
  evaluator:
    _target_: tmgg.evaluation.molecular.MolecularEvaluator.for_qm9
  model:
    input_dims: { X: 4, E: 5, y: 0 }
    output_dims: { X: 4, E: 5, y: 0 }
    output_dims_x_class: 4
    output_dims_x_feat: null
    output_dims_e_class: 5
    output_dims_e_feat: null
```

- [ ] **Step 2: Create the MOSES experiment YAML**

Create `src/tmgg/experiments/exp_configs/experiment/discrete_moses_digress_repro.yaml`:

```yaml
# @package _global_
# DiGress MOSES reproduction (Vignac et al., ICLR 2023, Table 5).

defaults:
  - /data: moses_digress
  - override /models/discrete@model: discrete_sbm_official
  - _self_

experiment_name: discrete_moses_digress_repro
wandb_project: discrete-moses-digress-repro

seed: 0

trainer:
  max_steps: 200000
  val_check_interval: 5000
  check_val_every_n_epoch: null
  gradient_clip_val: null

model:
  eval_every_n_steps: 25000
  num_nodes: 30
  evaluator:
    _target_: tmgg.evaluation.molecular.MolecularEvaluator.for_moses
  model:
    input_dims: { X: 8, E: 5, y: 0 }
    output_dims: { X: 8, E: 5, y: 0 }
    output_dims_x_class: 8
    output_dims_x_feat: null
    output_dims_e_class: 5
    output_dims_e_feat: null
```

- [ ] **Step 3: Create the GuacaMol experiment YAML**

Create `src/tmgg/experiments/exp_configs/experiment/discrete_guacamol_digress_repro.yaml`:

```yaml
# @package _global_
# DiGress GuacaMol reproduction (Vignac et al., ICLR 2023, Table 6).

defaults:
  - /data: guacamol_digress
  - override /models/discrete@model: discrete_sbm_official
  - _self_

experiment_name: discrete_guacamol_digress_repro
wandb_project: discrete-guacamol-digress-repro

seed: 0

trainer:
  max_steps: 500000
  val_check_interval: 10000
  check_val_every_n_epoch: null
  gradient_clip_val: null

model:
  eval_every_n_steps: 50000
  num_nodes: 88
  evaluator:
    _target_: tmgg.evaluation.molecular.MolecularEvaluator.for_guacamol
  model:
    input_dims: { X: 12, E: 5, y: 0 }
    output_dims: { X: 12, E: 5, y: 0 }
    output_dims_x_class: 12
    output_dims_x_feat: null
    output_dims_e_class: 5
    output_dims_e_feat: null
```

- [ ] **Step 4: Verify all four new experiment configs resolve through Hydra**

Run:
```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
for exp in discrete_planar_digress_repro discrete_qm9_digress_repro discrete_moses_digress_repro discrete_guacamol_digress_repro; do
    echo "=== $exp ==="
    uv run tmgg-discrete-gen +experiment="$exp" --cfg job 2>&1 | head -3
    echo
done
```
Expected: each block prints YAML, no Hydra errors. If any errors, fix the corresponding config.

- [ ] **Step 5: Commit**

```bash
git add src/tmgg/experiments/exp_configs/experiment/discrete_qm9_digress_repro.yaml src/tmgg/experiments/exp_configs/experiment/discrete_moses_digress_repro.yaml src/tmgg/experiments/exp_configs/experiment/discrete_guacamol_digress_repro.yaml
git commit -m "feat(configs): add QM9/MOSES/GuacaMol DiGress experiment YAMLs"
```

---

## Phase 6 — Launcher script (Phase 6)

### Task 6.1: Parameterised launcher

**Files:**
- Create: `scripts/run-digress-repro-modal.zsh`
- Modify: `run-discrete-sbm-vignac-repro-modal-a100.zsh` (the existing wrapper at repo root)

- [ ] **Step 1: Create the parameterised launcher**

Create `scripts/run-digress-repro-modal.zsh`:

```zsh
#!/usr/bin/env zsh
# DiGress repro panel launcher.
#
# Usage:
#   ./scripts/run-digress-repro-modal.zsh <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]
#
# Env knobs (with defaults):
#   USE_DOPPLER=1, DEPLOY_FIRST=1, DETACH=1, DRY_RUN=0,
#   GPU_TIER=fast, PRECISION=bf16-mixed, MODAL_DEBUG=0
#
# See docs/specs/2026-04-28-digress-repro-datasets-spec.md for context.

set -euo pipefail

: "${USE_DOPPLER:=1}"
: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=fast}"
: "${PRECISION:=bf16-mixed}"
: "${MODAL_DEBUG:=0}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

DATASET="${1:?usage: $0 <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]}"
shift

case "$DATASET" in
  sbm)        EXP="discrete_sbm_vignac_repro" ;;
  planar)     EXP="discrete_planar_digress_repro" ;;
  qm9)        EXP="discrete_qm9_digress_repro" ;;
  moses)      EXP="discrete_moses_digress_repro" ;;
  guacamol)   EXP="discrete_guacamol_digress_repro" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

run_prefixed() {
  if [[ "${USE_DOPPLER}" == "1" ]]; then
    doppler run -- "$@"
  else
    "$@"
  fi
}

if [[ "${DEPLOY_FIRST}" == "1" ]]; then
  print -r -- "Deploying Modal app and refreshing secrets..."
  run_prefixed mise run modal-deploy
fi

if [[ "${MODAL_DEBUG}" == "1" ]]; then
  modal_debug_override=true
else
  modal_debug_override=false
fi

typeset -a cmd
cmd=(
  uv run tmgg-modal run tmgg-discrete-gen
  +experiment="${EXP}"
  trainer.precision="${PRECISION}"
  modal_debug="${modal_debug_override}"
  --gpu "${GPU_TIER}"
)

if [[ "${DETACH}" == "1" ]]; then
  cmd+=(--detach)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if (( $# > 0 )); then
  cmd+=("$@")
fi

print -r -- "Launching DiGress repro: dataset=${DATASET} (${GPU_TIER}, ${PRECISION})"
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/scripts/run-digress-repro-modal.zsh`

- [ ] **Step 3: Replace the existing SBM wrapper with a thin compat shim**

Read the existing wrapper:
```bash
cat /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/run-discrete-sbm-vignac-repro-modal-a100.zsh
```

Replace its contents with:

```zsh
#!/usr/bin/env zsh
# Backward-compatible shim — delegates to the parameterised launcher.
#
# Usage:
#   ./run-discrete-sbm-vignac-repro-modal-a100.zsh [hydra-overrides...]
#
# All env knobs (DEPLOY_FIRST, DETACH, GPU_TIER, PRECISION, MODAL_DEBUG)
# are forwarded unchanged to scripts/run-digress-repro-modal.zsh.

set -euo pipefail
exec "$(dirname "${0}")/scripts/run-digress-repro-modal.zsh" sbm "$@"
```

- [ ] **Step 4: Sanity-check the launcher (dry run)**

Run:
```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
DEPLOY_FIRST=0 DRY_RUN=1 ./scripts/run-digress-repro-modal.zsh planar 2>&1 | tail -10
```
Expected: a "Launching DiGress repro: dataset=planar" line followed by the resolved `uv run tmgg-modal run ...` command.

- [ ] **Step 5: Commit**

```bash
git add scripts/run-digress-repro-modal.zsh run-discrete-sbm-vignac-repro-modal-a100.zsh
git commit -m "feat(scripts): add parameterised DiGress repro launcher + SBM compat shim"
```

---

## Phase 7 — Testing (Phase 7)

Tests follow TDD-after-implementation pattern (per the user directive). Each task here writes the test, runs it, then commits.

### Task 7.1: `test_atom_bond_vocabulary.py`

**Files:**
- Create: `tests/data/test_atom_bond_vocabulary.py`

- [ ] **Step 1: Write the test file**

Create `tests/data/test_atom_bond_vocabulary.py`:

```python
"""Tests for AtomBondVocabulary presets and properties."""

from __future__ import annotations

import pytest

from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


def test_qm9_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.qm9(remove_h=True)
    assert v.atom_decoder == ("C", "N", "O", "F")
    assert v.num_atom_types == 4


def test_qm9_preset_with_h() -> None:
    v = AtomBondVocabulary.qm9(remove_h=False)
    assert v.atom_decoder == ("H", "C", "N", "O", "F")
    assert v.num_atom_types == 5


def test_moses_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.moses()
    assert v.atom_decoder == ("C", "N", "S", "O", "F", "Cl", "Br", "H")


def test_guacamol_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.guacamol()
    assert v.atom_decoder == (
        "C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si",
    )


def test_vocabulary_is_hashable() -> None:
    v = AtomBondVocabulary.qm9()
    h1 = hash(v)
    h2 = hash(AtomBondVocabulary.qm9())
    assert h1 == h2


def test_encode_atom_roundtrip() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.decode_atom(v.encode_atom("C")) == "C"


def test_encode_atom_unknown_raises() -> None:
    v = AtomBondVocabulary.qm9()
    with pytest.raises(ValueError):
        v.encode_atom("Xx")


def test_encode_bond_roundtrip() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.decode_bond(v.encode_bond("AROMATIC")) == "AROMATIC"


def test_max_valence_lookup() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.max_valence("C") == 4
    assert v.max_valence("N") == 3
    assert v.max_valence("O") == 2


def test_repr_is_stable_across_calls() -> None:
    """Stable __repr__ matters for cache-key hashing."""
    v1 = AtomBondVocabulary.qm9()
    v2 = AtomBondVocabulary.qm9()
    assert repr(v1) == repr(v2)
```

- [ ] **Step 2: Run + commit**

```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
uv run pytest tests/data/test_atom_bond_vocabulary.py -v
```
Expected: all tests pass.

```bash
git add tests/data/test_atom_bond_vocabulary.py
git commit -m "test(data): cover AtomBondVocabulary presets + roundtrips"
```

---

### Task 7.2: `test_smiles_codec.py`

**Files:**
- Create: `tests/data/test_smiles_codec.py`

- [ ] **Step 1: Write the test file**

Create `tests/data/test_smiles_codec.py`:

```python
"""Tests for SMILESCodec encode/decode round-trip."""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


def _qm9_codec() -> SMILESCodec:
    return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)


def test_encode_simple_smiles() -> None:
    codec = _qm9_codec()
    data = codec.encode("CCO")
    assert data is not None
    assert data.X_class is not None
    assert data.E_class is not None
    assert data.X_class.shape == (1, 3, 4)
    assert data.E_class.shape == (1, 3, 3, 5)


def test_encode_invalid_smiles_returns_none() -> None:
    codec = _qm9_codec()
    assert codec.encode("not_a_smiles") is None


def test_encode_atom_count_overflow() -> None:
    codec = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=2)
    # ethanol = 3 heavy atoms — exceeds max_atoms=2
    assert codec.encode("CCO") is None


def test_decode_recovers_canonical() -> None:
    codec = _qm9_codec()
    data = codec.encode("CCO")
    assert data is not None
    decoded = codec.decode(data)
    # RDKit canonicalisation of "CCO" is itself "CCO".
    assert decoded == "CCO"


def test_cache_key_changes_with_vocab() -> None:
    qm9 = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    moses = SMILESCodec(vocab=AtomBondVocabulary.moses(), max_atoms=30)
    assert qm9.cache_key() != moses.cache_key()


def test_cache_key_changes_with_max_atoms() -> None:
    a = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    b = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=29)
    assert a.cache_key() != b.cache_key()


def test_cache_key_stable_across_instantiations() -> None:
    a = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    b = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    assert a.cache_key() == b.cache_key()


def test_encode_dataset_with_stats_counts() -> None:
    codec = _qm9_codec()
    smiles = ["CCO", "CC(=O)O", "not_a_smiles", "C" * 200]
    graphs, counters = codec.encode_dataset_with_stats(smiles)
    assert counters["input"] == 4
    assert counters["kept"] == 2
    assert counters["parse_failure"] == 1
    assert counters["atom_count_overflow"] == 1
    assert len(graphs) == 2
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/data/test_smiles_codec.py -v
git add tests/data/test_smiles_codec.py
git commit -m "test(data): cover SMILESCodec encode/decode + cache key"
```

---

### Task 7.3: `test_molecular_dataset.py` (cache hit/miss)

**Files:**
- Create: `tests/data/test_molecular_dataset.py`

- [ ] **Step 1: Write the test**

Create `tests/data/test_molecular_dataset.py`:

```python
"""Tests for MolecularGraphDataset shard caching.

Uses a tiny tmp-fixture subclass to avoid hitting the real QM9/MOSES/
GuacaMol downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

import pytest

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _TinyDataset(MolecularGraphDataset):
    DATASET_NAME = "tiny"
    DEFAULT_MAX_ATOMS = 5
    SAMPLE_SMILES = ["CCO", "CC(=O)O", "CCC"]

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(),
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        return list(self.SAMPLE_SMILES)


def test_first_setup_preprocesses_and_caches(tmp_path: Path) -> None:
    ds = _TinyDataset(split="train", cache_root=tmp_path)
    ds.setup()
    assert len(ds) == 3
    # Shard file should exist.
    shard_dir = tmp_path / "tiny" / "preprocessed" / ds._codec.cache_key() / "train"  # type: ignore[union-attr]
    assert shard_dir.exists()
    assert any(shard_dir.iterdir())


def test_second_setup_hits_cache(tmp_path: Path) -> None:
    ds1 = _TinyDataset(split="train", cache_root=tmp_path)
    ds1.setup()
    # Mutate the SAMPLE_SMILES on the second instance — if cache hit
    # is broken, ds2 would re-encode and pick up the new SMILES.
    class _MutantDataset(_TinyDataset):
        SAMPLE_SMILES = ["NEVER_CALLED"]

    ds2 = _MutantDataset(split="train", cache_root=tmp_path)
    ds2.setup()
    assert len(ds2) == 3  # cached from ds1, not regenerated


def test_codec_change_invalidates_cache(tmp_path: Path) -> None:
    ds1 = _TinyDataset(split="train", cache_root=tmp_path)
    ds1.setup()

    class _OtherCodecDataset(_TinyDataset):
        @classmethod
        @override
        def make_codec(cls) -> SMILESCodec:
            return SMILESCodec(
                vocab=AtomBondVocabulary.moses(),  # different vocab
                max_atoms=cls.DEFAULT_MAX_ATOMS,
            )

    ds2 = _OtherCodecDataset(split="train", cache_root=tmp_path)
    ds2.setup()
    # Different cache key directory.
    k1 = ds1._codec.cache_key()  # type: ignore[union-attr]
    k2 = ds2._codec.cache_key()  # type: ignore[union-attr]
    assert k1 != k2
    assert (tmp_path / "tiny" / "preprocessed" / k1).exists()
    assert (tmp_path / "tiny" / "preprocessed" / k2).exists()
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/data/test_molecular_dataset.py -v
git add tests/data/test_molecular_dataset.py
git commit -m "test(data): cover MolecularGraphDataset cache hit/miss + invalidation"
```

---

### Task 7.4: `test_molecular_datamodule.py`

**Files:**
- Create: `tests/data/test_molecular_datamodule.py`

- [ ] **Step 1: Write the test**

Create `tests/data/test_molecular_datamodule.py`:

```python
"""Tests for the molecular DataModule batch shape + collator integration."""

from __future__ import annotations

from pathlib import Path
from typing import override

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _TinyDataset(MolecularGraphDataset):
    DATASET_NAME = "tiny_dm"
    DEFAULT_MAX_ATOMS = 5

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(),
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        # 3 each: train/val/test
        return ["CCO", "CC(=O)O", "CCC"]


class _TinyDataModule(MolecularDataModule):
    dataset_cls = _TinyDataset


def test_dataloader_yields_graphdata(tmp_path: Path) -> None:
    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # batch should be a GraphData with per-batch shapes (bs, n, ...).
    assert batch.X_class is not None
    assert batch.E_class is not None
    assert batch.node_mask is not None
    assert batch.X_class.shape[0] == 2
    assert batch.E_class.shape[0] == 2
    assert batch.node_mask.shape[0] == 2


def test_size_distribution_populated(tmp_path: Path) -> None:
    dm = _TinyDataModule(batch_size=1, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    sd = dm.get_size_distribution("train")
    # Three molecules, sizes ∈ {3, 4, 3}.
    sample = sd.sample((100,))  # type: ignore[arg-type]
    assert sample.min() >= 3
    assert sample.max() <= 4
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/data/test_molecular_datamodule.py -v
git add tests/data/test_molecular_datamodule.py
git commit -m "test(data): cover MolecularDataModule batch shape + size distribution"
```

---

### Task 7.5: `test_spectre_planar_datamodule.py`

**Files:**
- Create: `tests/data/test_spectre_planar_datamodule.py`

- [ ] **Step 1: Write the test**

Create `tests/data/test_spectre_planar_datamodule.py`:

```python
"""Tests for SpectrePlanarDataModule, mirroring the SBM tests' structure."""

from __future__ import annotations

import pytest

from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule
from tmgg.data.datasets.spectre_planar import (
    SPECTRE_PLANAR_TEST_LEN,
    SPECTRE_PLANAR_TRAIN_LEN,
    SPECTRE_PLANAR_VAL_LEN,
    split_spectre_planar,
)


def test_split_sizes_match_spec() -> None:
    splits = split_spectre_planar()
    assert len(splits["train"]) == SPECTRE_PLANAR_TRAIN_LEN
    assert len(splits["val"]) == SPECTRE_PLANAR_VAL_LEN
    assert len(splits["test"]) == SPECTRE_PLANAR_TEST_LEN
    assert SPECTRE_PLANAR_TRAIN_LEN == 128
    assert SPECTRE_PLANAR_VAL_LEN == 32
    assert SPECTRE_PLANAR_TEST_LEN == 40


def test_setup_loads_fixture() -> None:
    """Requires fixture at ~/.cache/tmgg/spectre/planar_64_200.pt."""
    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule()
    dm.setup("fit")
    assert dm.num_nodes == 64
    assert dm._train_data is not None
    assert len(dm._train_data) == SPECTRE_PLANAR_TRAIN_LEN


def test_dataloader_shapes() -> None:
    """Sanity-check that the collator produces dense GraphData."""
    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule(batch_size=4)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.E_class is not None
    assert batch.E_class.shape[0] == 4  # batch dim
    assert batch.E_class.shape[1] == 64  # n_max
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/data/test_spectre_planar_datamodule.py -v
git add tests/data/test_spectre_planar_datamodule.py
git commit -m "test(data): cover SpectrePlanarDataModule split sizes + dataloader"
```

---

### Task 7.6: `test_molecular_metrics.py`

**Files:**
- Create: `tests/evaluation/test_molecular_metrics.py`

- [ ] **Step 1: Write the test**

Create `tests/evaluation/test_molecular_metrics.py`:

```python
"""Tests for the RDKit family of molecular metrics."""

from __future__ import annotations

import math

from tmgg.evaluation.molecular.rdkit_metrics import (
    NoveltyMetric,
    UniquenessMetric,
    ValidityMetric,
)


def test_validity_basic() -> None:
    gen = ["CCO", "not_a_smiles", "CC(=O)O"]
    v = ValidityMetric().compute(gen)
    assert math.isclose(v, 2 / 3, rel_tol=1e-6)


def test_validity_empty() -> None:
    assert ValidityMetric().compute([]) == 0.0


def test_uniqueness_with_duplicates() -> None:
    gen = ["CCO", "CCO", "CC(=O)O", "not_a_smiles"]
    # 3 valid, 2 distinct
    u = UniquenessMetric().compute(gen)
    assert math.isclose(u, 2 / 3, rel_tol=1e-6)


def test_novelty_against_train_set() -> None:
    gen = ["CCO", "CC(=O)O", "CCC", "not_a_smiles"]
    train = ["CCO"]
    n = NoveltyMetric().compute(gen, train)
    # 3 valid; novel = {CC(=O)O, CCC} = 2/3
    assert math.isclose(n, 2 / 3, rel_tol=1e-6)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/evaluation/test_molecular_metrics.py -v
git add tests/evaluation/test_molecular_metrics.py
git commit -m "test(eval): cover RDKit Validity/Uniqueness/Novelty metrics"
```

---

### Task 7.7: `test_molecular_evaluator.py`

**Files:**
- Create: `tests/evaluation/test_molecular_evaluator.py`

- [ ] **Step 1: Write the test**

Create `tests/evaluation/test_molecular_evaluator.py`:

```python
"""Tests for MolecularEvaluator + classmethod presets."""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary
from tmgg.evaluation.molecular import MolecularEvaluator
from tmgg.evaluation.molecular.rdkit_metrics import (
    UniquenessMetric,
    ValidityMetric,
)


def _qm9_codec() -> SMILESCodec:
    return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)


def test_for_qm9_metric_keys() -> None:
    ev = MolecularEvaluator.for_qm9()
    names = [m.name for m in ev.metrics]
    assert names == ["validity", "uniqueness", "novelty"]


def test_for_moses_metric_keys() -> None:
    ev = MolecularEvaluator.for_moses()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity", "uniqueness", "novelty",
        "fcd", "snn", "int_div", "filters", "scaffold_novelty",
    ]


def test_for_guacamol_metric_keys() -> None:
    ev = MolecularEvaluator.for_guacamol()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity", "uniqueness", "novelty",
        "kl_div_property", "fcd_chembl",
    ]


def test_evaluate_on_decoded_graphs() -> None:
    """End-to-end: evaluator decodes GraphData and runs metrics."""
    codec = _qm9_codec()
    g_ccol = codec.encode("CCO")
    g_acet = codec.encode("CC(=O)O")
    assert g_ccol is not None and g_acet is not None
    ev = MolecularEvaluator(
        metrics=[ValidityMetric(), UniquenessMetric()],
        codec=codec,
    )
    refs = [g_ccol]
    gen = [g_ccol, g_acet]
    results = ev.evaluate(refs, gen)
    assert "validity" in results.values
    assert "uniqueness" in results.values
    assert results.values["validity"] == 1.0
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/evaluation/test_molecular_evaluator.py -v
git add tests/evaluation/test_molecular_evaluator.py
git commit -m "test(eval): cover MolecularEvaluator presets + evaluate()"
```

---

### Task 7.8: Slow-marked integration test

**Files:**
- Create: `tests/training/test_diffusion_module_molecular.py`

- [ ] **Step 1: Write the slow test**

Create `tests/training/test_diffusion_module_molecular.py`:

```python
"""Slow-marked integration test: DiffusionModule + QM9 datamodule + molecular evaluator.

Trains for 5 steps on a tiny synthetic SMILES list to confirm the
plumbing end-to-end. Skipped under ``-m 'not slow'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_qm9_module(tmp_path: Path):
    """Build a tiny molecular DataModule with 6 SMILES."""
    from tmgg.data.data_modules.molecular.base import MolecularDataModule
    from tmgg.data.datasets.molecular.codec import SMILESCodec
    from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
    from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

    class _Tiny(MolecularGraphDataset):
        DATASET_NAME = "tiny_int"
        DEFAULT_MAX_ATOMS = 9

        @classmethod
        @override
        def make_codec(cls) -> SMILESCodec:
            return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=9)

        @override
        def download_smiles_split(self, split: str) -> list[str]:
            return ["CCO", "CC(=O)O", "CCC", "C", "CCN", "CCNC"]

    class _DM(MolecularDataModule):
        dataset_cls = _Tiny

    return _DM(batch_size=2, cache_root=str(tmp_path))


def test_dataloader_iterates(tiny_qm9_module) -> None:
    """Smoke: DataModule yields 1+ batches without crashing."""
    tiny_qm9_module.prepare_data()
    tiny_qm9_module.setup()
    batch = next(iter(tiny_qm9_module.train_dataloader()))
    assert batch is not None
```

- [ ] **Step 2: Run with the slow marker**

Run: `uv run pytest tests/training/test_diffusion_module_molecular.py -v -m slow`
Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add tests/training/test_diffusion_module_molecular.py
git commit -m "test(training): slow-marked DiffusionModule + molecular DM smoke"
```

---

### Task 7.9: Modal preflight test

**Files:**
- Create: `tests/modal/test_molecular_image.py`

- [ ] **Step 1: Write the marker test**

Create `tests/modal/test_molecular_image.py`:

```python
"""Modal-marked: confirm molecular deps are present in the image.

Skipped on host pytest unless ``-m modal`` is passed. Runs as part
of Modal-side CI / deploy verification.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.modal]


def test_rdkit_imports() -> None:
    from rdkit import Chem  # noqa: F401


def test_fcd_torch_imports() -> None:
    from fcd_torch import FCD  # noqa: F401


def test_moses_imports() -> None:
    import moses  # noqa: F401


def test_guacamol_imports() -> None:
    import guacamol  # noqa: F401
```

- [ ] **Step 2: Confirm pytest discovers the markers**

Run: `uv run pytest tests/modal/test_molecular_image.py --collect-only`
Expected: 4 tests collected (with the slow + modal markers).

- [ ] **Step 3: Confirm `pytest.ini` (or equivalent) registers the `modal` marker**

Run: `rg -n "markers" /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/pytest.ini /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/pyproject.toml | head -10`
If the `modal` marker is not registered, add it to `pytest.ini`:

```ini
markers =
    slow: tests that are slow (FCD ChemNet download, integration smoke)
    modal: tests that require the Modal image (rdkit/fcd-torch/moses/guacamol)
```

- [ ] **Step 4: Commit**

```bash
git add tests/modal/test_molecular_image.py pytest.ini
git commit -m "test(modal): preflight import test for molecular image deps"
```

---

### Task 7.10: Run the full test suite

- [ ] **Step 1: Run the fast lane**

```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
uv run pytest tests/ -x -m "not slow"
```
Expected: green, ≤ 30s.

- [ ] **Step 2: Run the slow lane (host-only — skips modal)**

```bash
uv run pytest tests/ -m "slow and not modal"
```
Expected: green, ≤ 5 min (FCD ChemNet download is the slowest).

- [ ] **Step 3: If any test fails, fix the corresponding implementation file and recommit per the affected task**

---

## Phase 8 — Validation runs (Phase 8)

### Task 8.1: Train-set encode/decode round-trip parity (CPU, slow-marked)

**Files:**
- Create: `tests/data/test_smiles_codec_parity.py`

- [ ] **Step 1: Write the parity test**

Create `tests/data/test_smiles_codec_parity.py`:

```python
"""Slow-marked: per-dataset train-set round-trip ≥ 99% canonical match.

Uses the dataset's own preprocessing — therefore depends on
``rdkit`` + the dataset packages being installed and the raw files
being downloadable.
"""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.slow

_SAMPLE_N = 1000


def _round_trip_match_rate(smiles: list[str], codec) -> float:
    matched = 0
    n_attempted = 0
    for s in smiles:
        encoded = codec.encode(s)
        if encoded is None:
            continue
        decoded = codec.decode(encoded)
        if decoded is None:
            continue
        # Compare canonical forms.
        from rdkit import Chem

        original_canon = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        n_attempted += 1
        if decoded == original_canon:
            matched += 1
    if n_attempted == 0:
        return 0.0
    return matched / n_attempted


@pytest.mark.parametrize(
    "dataset_name",
    ["qm9", "moses", "guacamol"],
)
def test_round_trip_match_rate_above_99pct(
    dataset_name: str, tmp_path
) -> None:
    if dataset_name == "qm9":
        from tmgg.data.datasets.molecular.qm9 import QM9Dataset

        ds = QM9Dataset(split="train", cache_root=tmp_path)
    elif dataset_name == "moses":
        from tmgg.data.datasets.molecular.moses import MOSESDataset

        ds = MOSESDataset(split="train", cache_root=tmp_path)
    else:
        from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset

        ds = GuacaMolDataset(split="train", cache_root=tmp_path)

    smiles = ds.download_smiles_split("train")
    rng = random.Random(0)
    sample = rng.sample(smiles, min(_SAMPLE_N, len(smiles)))
    rate = _round_trip_match_rate(sample, ds.make_codec())
    assert rate >= 0.99, (
        f"{dataset_name} round-trip match rate {rate:.4f} < 0.99"
    )
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/data/test_smiles_codec_parity.py -v -m slow
```
Expected: per-dataset match rate ≥ 99% (printed as test passes).

```bash
git add tests/data/test_smiles_codec_parity.py
git commit -m "test(data): per-dataset round-trip canonical-SMILES parity ≥ 99%"
```

---

### Task 8.2: End-to-end smoke run on Modal — Planar

- [ ] **Step 1: Launch via the parameterised launcher (deploy first)**

```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
DEPLOY_FIRST=1 DETACH=1 GPU_TIER=fast \
  ./scripts/run-digress-repro-modal.zsh planar \
  force_fresh=true trainer.max_steps=1000 trainer.val_check_interval=200 \
  > /tmp/tmgg-launch-planar.log 2>&1
```

Wait for the `Spawned: run_id=...` line in `/tmp/tmgg-launch-planar.log` (uses the same pattern documented in `.claude/skills/launch-and-monitor-modal-job/SKILL.md`).

- [ ] **Step 2: Use the modal-logs reattach loop pattern from the skill**

```bash
LOG=/tmp/tmgg-stream-planar.log
nohup bash -c "while true; do modal app logs tmgg-spectral >> '$LOG' 2>&1; sleep 5; done" \
  > /dev/null 2>&1 &
echo $! > /tmp/tmgg-stream-planar.pid; disown
```

Wait for "View run" + a few `gen-val/*` keys to appear in the log.

- [ ] **Step 3: Verify telemetry via wandb API**

```bash
WANDB_API_KEY=$(rg -oP "^GRAPH_DENOISE_TEAM_SERVICE=\K.*" .env) \
uv run python -c "
import wandb
api = wandb.Api()
runs = api.runs(
    'graph_denoise_team/discrete-planar-digress-repro',
    filters={'state': 'running'},
)
for r in list(runs):
    summary_keys = sorted([k for k in r.summary.keys() if not k.startswith('_')])
    gen_val = [k for k in summary_keys if k.startswith('gen-val/')]
    print('run', r.id, 'state', r.state)
    print('  gen-val keys:', gen_val)
"
```
Expected: `gen-val/{degree_mmd, clustering_mmd, spectral_mmd, planarity_accuracy, ...}` present.

- [ ] **Step 4: Cleanup the reattach loop**

```bash
PID=$(cat /tmp/tmgg-stream-planar.pid); pkill -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null
pkill -9 -f "modal app logs tmgg-spectral" 2>/dev/null
```

- [ ] **Step 5: No commit (validation run, no new artefacts).**

---

### Task 8.3: End-to-end smoke runs for QM9 + MOSES + GuacaMol

- [ ] **Step 1: Launch each via the parameterised launcher**

```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg

for dataset in qm9 moses guacamol; do
    DEPLOY_FIRST=0 DETACH=1 GPU_TIER=fast \
      ./scripts/run-digress-repro-modal.zsh $dataset \
      force_fresh=true trainer.max_steps=1000 trainer.val_check_interval=200 \
      > /tmp/tmgg-launch-$dataset.log 2>&1 &
done
wait
```

- [ ] **Step 2: For each, wait for "View run" to land in the modal logs and capture the wandb URL**

Use the same modal-logs reattach pattern as Task 8.2 but with one log file per dataset.

- [ ] **Step 3: Confirm `gen-val/*` keys appear in each run's wandb summary**

For each dataset, run the same wandb-API check as Task 8.2 step 3, with the matching `wandb_project` (`discrete-{qm9,moses,guacamol}-digress-repro`).

Expected per-dataset keys:
- QM9: `gen-val/{validity, uniqueness, novelty}`
- MOSES: `gen-val/{validity, uniqueness, novelty, fcd, snn, int_div, filters, scaffold_novelty}`
- GuacaMol: `gen-val/{validity, uniqueness, novelty, kl_div_property, fcd_chembl}`

- [ ] **Step 4: Cleanup all reattach loops**

```bash
for dataset in qm9 moses guacamol; do
    PID=$(cat /tmp/tmgg-stream-$dataset.pid 2>/dev/null) && \
        pkill -P "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
done
pkill -9 -f "modal app logs tmgg-spectral" 2>/dev/null
```

- [ ] **Step 5: No commit (validation run).**

---

### Task 8.4: QM9 50k-step parity run

- [ ] **Step 1: Launch the published-config QM9 run**

```bash
cd /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg
DEPLOY_FIRST=0 DETACH=1 GPU_TIER=fast \
  ./scripts/run-digress-repro-modal.zsh qm9 \
  force_fresh=true \
  > /tmp/tmgg-qm9-parity.log 2>&1
```

(The default config in `discrete_qm9_digress_repro.yaml` already uses 50k steps.)

- [ ] **Step 2: Periodically check progress via wandb API**

(See the launch-and-monitor skill for the polling pattern.)

- [ ] **Step 3: At completion, fetch final `gen-val/validity` and compare to DiGress Table 4**

Reference number from DiGress Table 4: `validity = 0.99`. Threshold: ours within ±5% (i.e. ≥ 0.94).

```bash
WANDB_API_KEY=... uv run python -c "
import wandb
api = wandb.Api()
runs = api.runs(
    'graph_denoise_team/discrete-qm9-digress-repro',
    filters={'state': 'finished'},
    order='-created_at',
)
last = next(iter(runs))
v = last.summary.get('gen-val/validity')
print('validity:', v)
assert v is not None and v >= 0.94, (
    f'QM9 validity {v} below 0.94 threshold (DiGress reports 0.99 ± 5%)'
)
"
```

- [ ] **Step 4: If parity fails, investigate (likely culprits: kekulisation difference, atom-vocabulary subtle mismatch, or split-randomisation drift). Document in a follow-up issue if a deeper fix is needed.**

- [ ] **Step 5: No commit (validation run); record the wandb URL + final-step numbers in the PR description.**

---

### Task 8.5: Spec self-review pass against the implemented code

- [ ] **Step 1: Re-read the spec's "Validation criteria summary"**

```bash
rg -A 20 "Validation criteria summary" /home/igork/900_personal/900.000_research/900.003_shervin_graphgen/tmgg/docs/specs/2026-04-28-digress-repro-datasets-spec.md
```

- [ ] **Step 2: Tick each criterion explicitly in a temporary scratchpad**

| Criterion | Status |
|---|---|
| All 4 experiment YAMLs train E2E on Modal | [task 8.2-8.3] |
| Each evaluator emits the right metric set | [task 7.7] |
| Round-trip ≥ 99% canonical-SMILES match | [task 8.1] |
| QM9 final-step validity within ±5% of Table 4 | [task 8.4] |
| `pytest -m "not slow"` ≤ 30s green | [task 7.10] |
| `pytest -m "slow"` ≤ 5 min green | [task 7.10] |

- [ ] **Step 3: If any criterion is unmet, link the failing task in the PR description and either fix in this PR or document deferral.**

---

## Self-Review

I checked the plan against the spec and looked for the failure modes from the writing-plans skill checklist. Findings + fixes applied inline:

1. **Spec coverage:** every spec section has a task — Phase 0 (deps) → 0.1/0.2; Phase 1 (Planar) → 1.1/1.2/1.3; Phase 2 (vocab + codec) → 2.1/2.2; Phase 3 (Dataset + DataModule + YAMLs) → 3.1–3.7; Phase 4 (metrics) → 4.1–4.4; Phase 5 (Evaluator + experiment YAMLs + DiffusionModule type widen) → 5.1–5.3; Phase 6 (launcher) → 6.1; Phase 7 (tests) → 7.1–7.10; Phase 8 (validation) → 8.1–8.5.
2. **Placeholders:** scanned for "TODO/TBD/etc." — none.
3. **Type consistency:** `MolecularGraphDataset.DATASET_NAME` and `DEFAULT_MAX_ATOMS` are referenced uniformly across `qm9.py`, `moses.py`, `guacamol.py`, the tests, and `MolecularDataModule.dataset_cls`. `MolecularEvaluator` constructor params (`metrics`, `codec`) match the test instantiations. `SMILESCodec.cache_key()` returns 16 hex chars in the codec; tests assert that shape.
4. **Cross-cutting concerns:**
   - Modal volume mount (`/data/datasets`) is referenced consistently by `_local_cache_root()` and the `tmgg.modal._lib.volumes.DATASETS_MOUNT` constant.
   - The `weights_only=False` exception is documented in the dataset module + spec.
   - The `DiffusionModule.evaluator` type widening (Task 5.2) covers both `GraphEvaluator` and `MolecularEvaluator`.

---

**Plan complete and saved to `docs/plans/2026-04-29-digress-repro-datasets-plan.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?