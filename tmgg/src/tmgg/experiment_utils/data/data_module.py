"""PyTorch Lightning data module for graph data."""

import random
from typing import Any, override

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data.dataset import GraphDataset
from tmgg.experiment_utils.data.dataset_wrappers import (
    ANUDatasetWrapper,
    ClassicalGraphsWrapper,
    NXGraphWrapperWrapper,
)
from tmgg.experiment_utils.data.pyg_datasets import PyGDatasetWrapper
from tmgg.experiment_utils.data.sbm import generate_block_sizes, generate_sbm_adjacency
from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset

# Legacy wrapper mapping for backward compatibility
# Prefer using create_dataset_wrapper() for new code
DATASET_WRAPPERS = {
    "anu": ANUDatasetWrapper,
    "classical": ClassicalGraphsWrapper,
    "nx": NXGraphWrapperWrapper,
}

# New synthetic graph types
SYNTHETIC_GRAPH_TYPES = {"d_regular", "regular", "lfr", "tree", "erdos_renyi", "er"}

# PyG benchmark datasets
PYG_DATASETS = {"qm9", "enzymes", "proteins"}


class GraphDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for various graph datasets."""

    dataset_name: str
    dataset_config: dict[str, Any]
    num_samples_per_graph: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    val_split: float
    test_split: float
    noise_levels: list[float]
    train_adjacency_matrices: list[torch.Tensor] | None
    val_adjacency_matrices: list[torch.Tensor] | None
    test_adjacency_matrices: list[torch.Tensor] | None
    all_partitions: list[list[int]] | None
    train_partitions: list[list[int]]

    def __init__(
        self,
        dataset_name: str,
        dataset_config: dict[str, Any],
        num_samples_per_graph: int = 1000,
        batch_size: int = 100,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.2,
        noise_levels: list[float] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the GraphDataModule.

        Args:
            dataset_name: Name of the dataset to use (e.g., "sbm", "classical").
            dataset_config: Dictionary of parameters for the chosen dataset.
            num_samples_per_graph: Number of samples (permutations) per graph.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            val_split: Fraction of data to use for validation (for non-SBM datasets).
            test_split: Fraction of data to use for testing (for non-SBM datasets).
            noise_levels: List of noise levels for evaluation.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples_per_graph = num_samples_per_graph
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.noise_levels = noise_levels or [0.005, 0.02, 0.05, 0.1, 0.25, 0.4, 0.5]

        self.train_adjacency_matrices = None
        self.val_adjacency_matrices = None
        self.test_adjacency_matrices = None
        self.all_partitions = None  # For SBM
        self.train_partitions = []

    @override
    def prepare_data(self) -> None:
        """Download or prepare data. Called only once per node."""
        if self.dataset_name == "sbm":
            sbm_params = self.dataset_config

            # Check if explicit block_sizes provided (for denoising script replication)
            if "block_sizes" in sbm_params:
                self.all_partitions = [sbm_params["block_sizes"]]
            else:
                # Use random partitions (for notebook replication)
                self.all_partitions = generate_block_sizes(
                    sbm_params["num_nodes"],
                    min_blocks=sbm_params.get("min_blocks", 2),
                    max_blocks=sbm_params.get("max_blocks", 4),
                    min_size=sbm_params.get("min_block_size", 2),
                    max_size=sbm_params.get("max_block_size", 15),
                )
        elif self.dataset_name in DATASET_WRAPPERS:
            # Instantiating the wrapper might trigger downloads
            wrapper_cls = DATASET_WRAPPERS[self.dataset_name]
            _ = wrapper_cls(**self.dataset_config)

    @override
    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if self.dataset_name == "sbm":
            self._setup_sbm(stage)
        elif self.dataset_name in DATASET_WRAPPERS:
            self._setup_from_wrapper()
        elif self.dataset_name.lower() in SYNTHETIC_GRAPH_TYPES:
            self._setup_synthetic_graphs()
        elif self.dataset_name.lower() in PYG_DATASETS:
            self._setup_pyg_dataset()
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _setup_sbm(self, stage: str | None = None) -> None:
        sbm_params = self.dataset_config
        p_intra: float = sbm_params.get("p_intra", 1.0)
        q_inter: float = sbm_params.get("q_inter", 0.0)

        # all_partitions must be set by prepare_data() before setup() is called
        if self.all_partitions is None:
            raise RuntimeError(
                "all_partitions is None. prepare_data() must be called before setup()."
            )
        all_partitions = self.all_partitions  # Local reference for type narrowing

        # Handle fixed block sizes (for denoising script replication)
        if "block_sizes" in sbm_params:
            # Use the single fixed partition for both train and test
            num_train_partitions = 1
            num_test_partitions = 1
            # For fixed block sizes, we only need 1 partition total since train/test share it
            total_needed = 1
        else:
            # Use random partitions (for notebook replication)
            num_train_partitions = sbm_params.get("num_train_partitions", 10)
            num_test_partitions = sbm_params.get("num_test_partitions", 10)
            total_needed = num_train_partitions + num_test_partitions

        if len(all_partitions) < total_needed:
            raise ValueError(
                f"Not enough valid SBM partitions ({len(all_partitions)}) for "
                f"requested train ({num_train_partitions}) and test ({num_test_partitions}) partitions."
            )

        if (stage == "fit" or stage is None) and self.train_adjacency_matrices is None:
            if "block_sizes" in sbm_params:
                # For fixed block sizes, use the same partition
                train_partitions = all_partitions
            else:
                # For random partitions, sample different ones
                train_partitions = random.sample(all_partitions, num_train_partitions)
            self.train_partitions = train_partitions  # Save for test set exclusion

            if "block_sizes" in sbm_params:
                # For fixed block sizes, use the same partition for test too
                remaining_partitions = all_partitions
            else:
                # For random partitions, use different ones for test
                remaining_partitions = [
                    p for p in all_partitions if p not in train_partitions
                ]

            num_val_partitions = min(5, max(1, len(remaining_partitions) // 2))
            val_partitions = random.sample(remaining_partitions, num_val_partitions)

            self.train_adjacency_matrices = [
                torch.tensor(
                    generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32
                )
                for p in train_partitions
            ]
            self.val_adjacency_matrices = [
                torch.tensor(
                    generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32
                )
                for p in val_partitions
            ]

        if (stage == "test" or stage is None) and self.test_adjacency_matrices is None:
            train_partitions = getattr(self, "train_partitions", [])
            if "block_sizes" in sbm_params:
                # For fixed block sizes, reuse the same partition for test
                test_partitions = all_partitions
            else:
                # For random partitions, exclude train partitions
                test_partitions = random.sample(
                    [p for p in all_partitions if p not in train_partitions],
                    num_test_partitions,
                )
            self.test_adjacency_matrices = [
                torch.tensor(
                    generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32
                )
                for p in test_partitions
            ]

    def _setup_from_wrapper(self) -> None:
        if self.train_adjacency_matrices is not None:
            return

        wrapper_cls = DATASET_WRAPPERS[self.dataset_name]
        dataset_wrapper = wrapper_cls(**self.dataset_config)
        all_matrices = dataset_wrapper.get_adjacency_matrices()
        dtypes = [x.dtype for x in all_matrices]
        assert all(x == torch.float for x in dtypes), f"{dtypes=}"

        random.shuffle(all_matrices)

        num_graphs = len(all_matrices)
        num_test = int(self.test_split * num_graphs)
        num_val = int(self.val_split * num_graphs)

        self.test_adjacency_matrices = all_matrices[:num_test]
        self.val_adjacency_matrices = all_matrices[num_test : num_test + num_val]
        self.train_adjacency_matrices = all_matrices[num_test + num_val :]

    def _setup_synthetic_graphs(self) -> None:
        """Set up synthetic graph datasets (d-regular, LFR, tree, ER)."""
        if self.train_adjacency_matrices is not None:
            return

        # Map dataset name to graph type
        name_to_type = {
            "d_regular": "regular",
            "regular": "regular",
            "lfr": "lfr",
            "tree": "tree",
            "erdos_renyi": "erdos_renyi",
            "er": "erdos_renyi",
        }
        graph_type = name_to_type[self.dataset_name.lower()]

        # Get parameters from config
        n = self.dataset_config.get("num_nodes", 50)
        num_graphs = self.dataset_config.get("num_graphs", 1000)
        seed = self.dataset_config.get("seed", 42)

        # Extract type-specific parameters
        type_kwargs = {}
        if graph_type == "regular":
            type_kwargs["d"] = self.dataset_config.get("d", 3)
        elif graph_type == "erdos_renyi":
            type_kwargs["p"] = self.dataset_config.get("p", 0.1)
        elif graph_type == "lfr":
            type_kwargs["tau1"] = self.dataset_config.get("tau1", 3.0)
            type_kwargs["tau2"] = self.dataset_config.get("tau2", 1.5)
            type_kwargs["mu"] = self.dataset_config.get("mu", 0.1)
            type_kwargs["average_degree"] = self.dataset_config.get(
                "average_degree", None
            )
            type_kwargs["min_community"] = self.dataset_config.get(
                "min_community", None
            )

        # Generate graphs
        dataset = SyntheticGraphDataset(
            graph_type=graph_type,
            n=n,
            num_graphs=num_graphs,
            seed=seed,
            **type_kwargs,
        )

        # Split into train/val/test (70/10/20 default)
        train_ratio = 1 - self.val_split - self.test_split
        train, val, test = dataset.train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=self.val_split,
            seed=seed,
        )

        # Convert to list of tensors
        self.train_adjacency_matrices = [torch.from_numpy(A).float() for A in train]
        self.val_adjacency_matrices = [torch.from_numpy(A).float() for A in val]
        self.test_adjacency_matrices = [torch.from_numpy(A).float() for A in test]

    def _setup_pyg_dataset(self) -> None:
        """Set up PyTorch Geometric benchmark datasets."""
        if self.train_adjacency_matrices is not None:
            return

        # Get parameters from config
        root: str | None = self.dataset_config.get("root", None)
        max_graphs: int | None = self.dataset_config.get("max_graphs", None)
        seed: int = self.dataset_config.get("seed", 42)

        # Load dataset
        dataset = PyGDatasetWrapper(
            dataset_name=self.dataset_name,
            root=root,
            max_graphs=max_graphs,
        )

        # Split into train/val/test
        train_ratio = 1 - self.val_split - self.test_split
        train, val, test = dataset.train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=self.val_split,
            seed=seed,
        )

        # Convert to list of tensors
        self.train_adjacency_matrices = [torch.from_numpy(A).float() for A in train]
        self.val_adjacency_matrices = [torch.from_numpy(A).float() for A in val]
        self.test_adjacency_matrices = [torch.from_numpy(A).float() for A in test]

    def _create_dataloader(
        self,
        adjacency_matrices: list[torch.Tensor],
        samples_per_graph: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create a data loader with common parameters."""
        dataset = GraphDataset(adjacency_matrices, samples_per_graph)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create training data loader."""
        if self.train_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.train_adjacency_matrices, self.num_samples_per_graph, shuffle=True
        )

    @override
    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create validation data loader."""
        if self.val_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.val_adjacency_matrices, self.num_samples_per_graph // 2, shuffle=False
        )

    @override
    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create test data loader."""
        if self.test_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.test_adjacency_matrices, self.num_samples_per_graph // 2, shuffle=False
        )

    def get_sample_adjacency_matrix(self, stage: str = "train") -> torch.Tensor:
        """Get a sample adjacency matrix for visualization."""
        matrices: list[torch.Tensor] | None = None
        if stage == "train":
            matrices = self.train_adjacency_matrices
        elif stage == "val":
            matrices = self.val_adjacency_matrices
        elif stage == "test":
            matrices = self.test_adjacency_matrices

        if not matrices:
            raise RuntimeError(
                f"No data available for stage '{stage}'. "
                "Please ensure setup() has been called and the dataset is not empty."
            )

        return random.choice(matrices)
