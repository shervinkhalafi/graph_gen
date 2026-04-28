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
            raise NotImplementedError(f"{type(self).__name__} must set DATASET_NAME.")
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
        return self.cache_root / self.DATASET_NAME / "preprocessed" / codec.cache_key()

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

                    time.sleep(2**attempt)
        raise RuntimeError(f"Failed to download {url} after 3 attempts: {last_exc!r}")

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
                    self.DATASET_NAME,
                    split,
                )

    def setup(self) -> None:
        """Populate ``self._graphs`` for the configured split."""
        codec = self.make_codec()
        self._codec = codec
        shard_dir = self._shard_dir()
        if shard_dir.exists() and any(shard_dir.iterdir()):
            self._graphs = self._load_shards()
            return

        logger.info(
            "%s/%s: preprocessing (cache miss at %s)",
            self.DATASET_NAME,
            self.split,
            shard_dir,
        )
        smiles = self.download_smiles_split(self.split)
        graphs, counters = codec.encode_dataset_with_stats(smiles)
        logger.info(
            "%s/%s: preprocessed; counters=%s",
            self.DATASET_NAME,
            self.split,
            counters,
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

    def __len__(self) -> int:
        if self._graphs is None:
            raise RuntimeError(f"{type(self).__name__} not set up. Call setup() first.")
        return len(self._graphs)

    @override
    def __getitem__(self, idx: int) -> GraphData:
        if self._graphs is None:
            raise RuntimeError(f"{type(self).__name__} not set up. Call setup() first.")
        return self._graphs[idx]
