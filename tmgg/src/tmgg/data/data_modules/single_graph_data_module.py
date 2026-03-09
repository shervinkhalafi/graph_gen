"""Single-graph data module for denoising experiments.

Implements the single-graph training protocol where a model is trained on
a single graph with multiple noise realizations, testing generalization to
new noise samples and unseen graph instances from the same distribution.
"""

from typing import Any, override

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.sbm import generate_sbm_batch
from tmgg.data.datasets.synthetic_graphs import SyntheticGraphDataset

from .base_data_module import BaseGraphDataModule


class SingleGraphDataset(Dataset[GraphData]):
    """Dataset that returns the same graph repeated for noise sampling.

    During training, the dataloader returns copies of the same graph. The
    Lightning module applies fresh noise to each sample, creating diverse
    training examples from a single graph structure.

    Parameters
    ----------
    graph : np.ndarray
        Single adjacency matrix of shape (n, n).
    num_samples : int
        Number of samples per epoch.
    """

    graph_data: GraphData
    num_samples: int

    def __init__(self, graph: npt.NDArray[np.float32], num_samples: int):
        self.graph_data = GraphData.from_adjacency(torch.from_numpy(graph).float())
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> GraphData:
        # Return the same graph each time (noise applied in training step)
        return self.graph_data


class SingleGraphDataModule(BaseGraphDataModule):
    """Lightning DataModule for single-graph training protocol.

    Supports two modes controlled by `same_graph_all_splits`:

    When `same_graph_all_splits=True` (Stage 1 protocol):
    - All splits (train/val/test) use the identical graph G
    - Only noise varies across samples
    - Tests generalization to new noise realizations on a known structure

    When `same_graph_all_splits=False` (default, Stage 2+ protocol):
    - Training uses graph G, validation/test use different graphs G', G''
    - Tests generalization to both new noise and unseen graph structures

    Parameters
    ----------
    graph_type : str
        Type of graph to generate. Supported types:
        - ``"sbm"``: Stochastic Block Model
        - ``"erdos_renyi"``: Erdos-Renyi random graphs
        - ``"regular"``: d-regular graphs
        - ``"tree"``: Random trees
        - ``"ring_of_cliques"``: NetworkX ring of cliques
        - ``"lfr"``: LFR benchmark graphs (community structure)
        - ``"pyg_enzymes"``, ``"pyg_qm9"``, ``"pyg_proteins"``: PyG datasets
    num_nodes : int
        Number of nodes.
    graph_config : dict or None
        Additional parameters for graph generation (e.g., ``p`` for ER,
        ``d`` for regular, SBM parameters).
    num_train_samples : int
        Number of training samples per epoch (noise realizations).
    num_val_samples : int
        Number of validation samples.
    num_test_samples : int
        Number of test samples.
    batch_size : int
        Batch size for dataloaders.
    num_workers : int
        Number of dataloader workers.
    train_seed : int
        Seed for generating the training graph.
    val_test_seed : int
        Seed for generating validation/test graphs (used only when
        ``same_graph_all_splits=False``).
    same_graph_all_splits : bool
        If True, val/test use the same graph as training (Stage 1 protocol).
        If False, val/test use different graphs (Stage 2+ protocol).
    noise_levels : list of float
        Accepted for Hydra compatibility but not stored — the training
        module owns noise configuration.
    noise_type : str
        Accepted for Hydra compatibility but not stored — the training
        module owns noise configuration.

    Examples
    --------
    >>> dm = SingleGraphDataModule(
    ...     graph_type="sbm",
    ...     num_nodes=50,
    ...     graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
    ...     num_train_samples=1000,
    ...     num_val_samples=100,
    ...     num_test_samples=100,
    ...     batch_size=16,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    graph_type: str
    num_nodes: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    train_seed: int
    val_test_seed: int
    same_graph_all_splits: bool
    graph_config: dict[str, Any]
    train_graph: npt.NDArray[np.float32] | None
    val_graph: npt.NDArray[np.float32] | None
    test_graph: npt.NDArray[np.float32] | None
    _actual_num_nodes: int

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        graph_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        num_train_samples: int = 1000,
        num_val_samples: int = 100,
        num_test_samples: int = 100,
        batch_size: int = 16,
        num_workers: int = 0,
        train_seed: int = 42,
        val_test_seed: int = 123,
        same_graph_all_splits: bool = False,
        noise_levels: list[float] | None = None,
        noise_type: str = "digress",
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            seed=train_seed,
        )
        self.save_hyperparameters()

        self.graph_type = graph_type.lower()
        self.num_nodes = num_nodes
        self.graph_config = graph_config or {}
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.train_seed = train_seed
        self.val_test_seed = val_test_seed
        self.same_graph_all_splits = same_graph_all_splits

        self.train_graph = None
        self.val_graph = None
        self.test_graph = None
        self._actual_num_nodes = num_nodes

    def _generate_graph(self, seed: int) -> npt.NDArray[np.float32]:
        """Generate a single graph of the specified type.

        For SBM, delegates to ``generate_sbm_batch(num_graphs=1)``.
        For PyG datasets, uses ``_load_pyg_graph``. For all other types,
        delegates to ``SyntheticGraphDataset(num_graphs=1)``.

        Parameters
        ----------
        seed : int
            Random seed for generation.

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape ``(n, n)`` with dtype float32.
        """
        if self.graph_type == "sbm":
            return generate_sbm_batch(
                num_graphs=1,
                num_nodes=self.num_nodes,
                num_blocks=self.graph_config.get("num_blocks", 3),
                p_intra=self.graph_config.get("p_intra", 0.7),
                p_inter=self.graph_config.get("p_inter", 0.05),
                seed=seed,
            )[0]

        if self.graph_type.startswith("pyg_"):
            return self._load_pyg_graph(seed)

        # All other types: delegate to SyntheticGraphDataset
        dataset = SyntheticGraphDataset(
            graph_type=self.graph_type,
            num_nodes=self.num_nodes,
            num_graphs=1,
            seed=seed,
            **self.graph_config,
        )
        return dataset.get_adjacency_matrices()[0]

    def _load_pyg_graph(self, seed: int) -> npt.NDArray[np.float32]:
        """Load a single graph from a PyTorch Geometric dataset.

        Delegates to ``PyGDatasetWrapper`` for loading and symmetrization,
        then selects a single graph and trims padding.

        Parameters
        ----------
        seed : int
            Used to deterministically select graph index if not specified.

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n, n).
        """
        from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

        dataset_name = self.graph_type.removeprefix("pyg_")
        root = self.graph_config.get("root", None)

        dataset = PyGDatasetWrapper(dataset_name=dataset_name, root=root)

        graph_idx: int | None = self.graph_config.get("graph_idx", None)
        if graph_idx is None:
            rng = np.random.default_rng(seed)
            graph_idx = int(rng.integers(0, len(dataset)))

        if graph_idx >= len(dataset):
            raise ValueError(
                f"graph_idx {graph_idx} out of range for dataset "
                f"with {len(dataset)} graphs"
            )

        A_np = dataset.adjacencies[graph_idx]

        # Trim padding: PyGDatasetWrapper pads to max_n, single-graph needs actual size
        n = int(dataset.num_nodes[graph_idx])
        A_np = A_np[:n, :n]

        self._actual_num_nodes = n
        return A_np

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs for train/val/test.

        Parameters
        ----------
        stage : str, optional
            Stage: "fit", "validate", "test", or None (all).
        """
        # Generate training graph
        if stage in (None, "fit"):
            self.train_graph = self._generate_graph(self.train_seed)

        if self.same_graph_all_splits:
            # Stage 1 protocol: all splits use identical graph, only noise varies
            if stage in (None, "fit", "validate"):
                self.val_graph = self.train_graph
            if stage in (None, "test"):
                self.test_graph = self.train_graph
        else:
            # Stage 2+ protocol: different graphs for val/test
            if stage in (None, "fit", "validate"):
                self.val_graph = self._generate_graph(self.val_test_seed)
            if stage in (None, "test"):
                self.test_graph = self._generate_graph(self.val_test_seed + 1000)

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        """Return training dataloader."""
        if self.train_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.train_graph, self.num_train_samples)
        return self._make_dataloader(
            dataset, shuffle=True, collate_fn=GraphData.collate
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        """Return validation dataloader."""
        if self.val_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.val_graph, self.num_val_samples)
        return self._make_dataloader(
            dataset, shuffle=False, collate_fn=GraphData.collate
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        """Return test dataloader."""
        if self.test_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.test_graph, self.num_test_samples)
        return self._make_dataloader(
            dataset, shuffle=False, collate_fn=GraphData.collate
        )

    @override
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata about the dataset."""
        return {
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "same_graph_all_splits": self.same_graph_all_splits,
        }

    def get_train_graph(self) -> npt.NDArray[np.float32]:
        """Return the training graph.

        Returns
        -------
        np.ndarray
            Training adjacency matrix.
        """
        if self.train_graph is None:
            raise RuntimeError("Call setup() before accessing graphs")
        return self.train_graph

    def get_val_graph(self) -> npt.NDArray[np.float32]:
        """Return the validation graph.

        Returns
        -------
        np.ndarray
            Validation adjacency matrix.
        """
        if self.val_graph is None:
            raise RuntimeError("Call setup() before accessing graphs")
        return self.val_graph

    def get_test_graph(self) -> npt.NDArray[np.float32]:
        """Return the test graph.

        Returns
        -------
        np.ndarray
            Test adjacency matrix.
        """
        if self.test_graph is None:
            raise RuntimeError("Call setup() before accessing graphs")
        return self.test_graph

    def get_sample_adjacency_matrix(self, stage: str = "train") -> torch.Tensor:
        """Get a sample adjacency matrix for visualization.

        Parameters
        ----------
        stage : str
            Which split to sample from: "train", "val", or "test".

        Returns
        -------
        torch.Tensor
            Adjacency matrix as a tensor.
        """
        graph: npt.NDArray[np.float32] | None
        if stage == "train":
            graph = self.train_graph
        elif stage == "val":
            graph = self.val_graph
        elif stage == "test":
            graph = self.test_graph
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if graph is None:
            raise RuntimeError(f"Call setup() before accessing {stage} graph")

        return torch.from_numpy(graph).float()
