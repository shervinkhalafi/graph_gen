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
from typing import TYPE_CHECKING, cast, override

from torch.utils.data import DataLoader

from tmgg.data.data_modules.base_data_module import BaseGraphDataModule
from tmgg.data.data_modules.multigraph_data_module import (
    GraphDataCollator,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.utils.noising.size_distribution import SizeDistribution

if TYPE_CHECKING:
    from torch_geometric.data import Data


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
        self.cache_root: Path | None = Path(cache_root) if cache_root else None
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
            n_max_static=self.num_nodes_max_static if self.pad_to_static_n_max else None
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        if self._train_dataset is None:
            raise RuntimeError(f"{type(self).__name__} not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(cast("list[Data]", list(self._train_dataset._graphs or []))),
            shuffle=True,
            collate_fn=self._dense_collator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        if self._val_dataset is None:
            raise RuntimeError(f"{type(self).__name__} not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(cast("list[Data]", list(self._val_dataset._graphs or []))),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        if self._test_dataset is None:
            raise RuntimeError(f"{type(self).__name__} not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(cast("list[Data]", list(self._test_dataset._graphs or []))),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        split_map = {
            "train": self._train_dataset,
            "val": self._val_dataset,
            "test": self._test_dataset,
        }
        if split is None:
            datasets = [d for d in split_map.values() if d is not None]
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
