"""Lightning DataModule for the SPECTRE SBM fixture.

Unlike :class:`SyntheticCategoricalDataModule`, which generates SBM
graphs at runtime, this datamodule loads the upstream SPECTRE fixture
(``sbm_200.pt``) used by DiGress's public SBM experiment. The fixture is
fixed in both graph count (200) and per-graph node count (variable in
[44, 187]); splits match upstream DiGress exactly.

Downstream wiring is identical to the synthetic path: every dataloader
yields dense :class:`~tmgg.data.datasets.graph_types.GraphData` batches
via the shared :func:`_collate_pyg_to_graphdata` collator, so
``DiffusionModule`` picks this datamodule up unchanged.

See :mod:`tmgg.data.datasets.spectre_sbm` for the load/split helpers
and ``docs/reports/2026-04-15-upstream-digress-parity-audit.md`` for
the numerical parity rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from tmgg.data.data_modules.base_data_module import BaseGraphDataModule
from tmgg.data.data_modules.multigraph_data_module import (
    _collate_pyg_raw,
    _collate_pyg_to_graphdata,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.spectre_sbm import (
    SpectreSBMDataset,
    load_spectre_sbm_fixture,
    split_spectre_sbm,
)
from tmgg.utils.noising.size_distribution import SizeDistribution


class SpectreSBMDataModule(BaseGraphDataModule):
    """Load the SPECTRE SBM fixture and serve it via the TMGG collator.

    Parameters
    ----------
    batch_size, num_workers, pin_memory, seed
        Forwarded to :class:`BaseGraphDataModule`. ``seed`` only affects
        the training-split dataloader shuffle; the SPECTRE split itself
        is seeded separately (upstream-matching, see
        :func:`split_spectre_sbm`).
    cache_dir
        Directory under which ``sbm_200.pt`` is cached when the fixture
        has to be downloaded. When ``None``, uses the default cache
        (``~/.cache/tmgg/spectre/``).
    fixture_path
        Path to a pre-downloaded fixture. When set, no network call is
        attempted. Useful when running inside a Modal container whose
        image already contains the file, or for tests.
    """

    # The fixture contains variable-size graphs; ``num_nodes`` is set to
    # the maximum observed node count for size-distribution-aware code
    # paths that need a padding ceiling. Individual graphs retain their
    # real node counts via the :class:`GraphData.node_mask`.
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
        num_nodes_max_static: int = 200,
        **_metadata: object,
    ) -> None:
        # ``**_metadata`` swallows informational keys (notably
        # ``eval_meta``) that some upstream config blocks attach to the
        # data namespace for downstream Hydra interpolation. They have
        # no runtime effect on the datamodule itself; rejecting them
        # would force every parity-fix config to special-case the
        # ``+data=spectre_sbm`` overlay.
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        # Parity #42 / D-11: safe upper bound on node count, exposed
        # for Hydra interpolation by model configs that need
        # ``max_n_nodes``. Default 200 covers the SPECTRE SBM fixture's
        # observed maximum of 187.
        self.num_nodes_max_static = num_nodes_max_static
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        self._fixture_path: Path | None = Path(fixture_path) if fixture_path else None

        # Populated by setup().
        self._train_data: list[Data] | None = None
        self._val_data: list[Data] | None = None
        self._test_data: list[Data] | None = None
        self._train_node_counts: list[int] | None = None

        # Informational metadata; mirrors ``MultiGraphDataModule`` so
        # helper code that reads ``num_nodes`` keeps working.
        self.graph_type = "sbm"
        self.num_nodes = 0  # updated by setup() from the fixture

        self.save_hyperparameters()

    @override
    def setup(self, stage: str | None = None) -> None:
        """Load the fixture and materialise PyG Data lists for each split."""
        if self._train_data is not None:
            return

        fixture = self._fixture_path
        if fixture is None and self._cache_dir is not None:
            fixture = self._cache_dir / "sbm_200.pt"

        adjs, n_nodes = load_spectre_sbm_fixture(path=fixture)
        splits = split_spectre_sbm()

        # Track the training-split node counts for size-distribution use,
        # and remember the max for ``num_nodes``. Other splits iterate
        # their adjacencies lazily via the dataset indexer.
        train_adjs = [adjs[i] for i in splits["train"]]
        val_adjs = [adjs[i] for i in splits["val"]]
        test_adjs = [adjs[i] for i in splits["test"]]

        self._train_data = [
            SpectreSBMDataset(train_adjs)[i] for i in range(len(train_adjs))
        ]
        self._val_data = [SpectreSBMDataset(val_adjs)[i] for i in range(len(val_adjs))]
        self._test_data = [
            SpectreSBMDataset(test_adjs)[i] for i in range(len(test_adjs))
        ]

        self._train_node_counts = [n_nodes[i] for i in splits["train"]]
        # Derive num_nodes from train+val only, never test. Mirrors upstream
        # DiGress AbstractDatasetInfos.complete_infos (abstract_dataset.py:95-100).
        # Including test would leak test-set graph sizes into model construction.
        train_val_node_counts = [n_nodes[i] for i in splits["train"]] + [
            n_nodes[i] for i in splits["val"]
        ]
        self.num_nodes = int(max(train_val_node_counts))

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        if self._train_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=_collate_pyg_to_graphdata,
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        if self._val_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=_collate_pyg_to_graphdata,
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        if self._test_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=_collate_pyg_to_graphdata,
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[object]:
        """Raw PyG ``Batch`` training loader for the parity-port π estimator."""
        if self._train_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=_collate_pyg_raw,
        )

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the empirical size distribution for a split.

        Unlike the synthetic SBM path, SPECTRE graphs have strongly
        variable node counts (44-187) and the size distribution is a
        first-class training signal — ``log p(N)`` enters the VLB.
        """
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
            # Pre-setup call with no node count information yet.
            return SizeDistribution.fixed(self.num_nodes or 1)

        node_counts = [int(d.num_nodes) for d in data_list]  # type: ignore[arg-type]
        return SizeDistribution.from_node_counts(node_counts)
