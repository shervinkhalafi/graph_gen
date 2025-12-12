"""Single-graph data module for denoising experiments.

Implements the single-graph training protocol where a model is trained on
a single graph with multiple noise realizations, testing generalization to
new noise samples and unseen graph instances from the same distribution.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from tmgg.experiment_utils.data.sbm import generate_sbm_adjacency
from tmgg.experiment_utils.data.synthetic_graphs import (
    SyntheticGraphDataset,
    generate_erdos_renyi_graphs,
    generate_regular_graphs,
    generate_tree_graphs,
)


class SingleGraphDataset(Dataset):
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

    def __init__(self, graph: np.ndarray, num_samples: int):
        self.graph = torch.from_numpy(graph).float()
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the same graph each time (noise applied in training step)
        return self.graph


class SingleGraphDataModule(pl.LightningDataModule):
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
        - "sbm": Stochastic Block Model
        - "erdos_renyi": Erdős-Rényi random graphs
        - "regular": d-regular graphs
        - "tree": Random trees
        - "ring_of_cliques": NetworkX ring of cliques
        - "lfr": LFR benchmark graphs (community structure)
        - "pyg_enzymes", "pyg_qm9", "pyg_proteins": PyG datasets (sample one)
    n : int
        Number of nodes.
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
        same_graph_all_splits=False).
    same_graph_all_splits : bool
        If True, val/test use the same graph as training (Stage 1 protocol).
        If False, val/test use different graphs (Stage 2+ protocol).
    noise_levels : list of float
        Noise levels used for evaluation (passed to LightningModule).
    noise_type : str
        Type of noise to apply.
    **graph_kwargs
        Additional parameters for graph generation (e.g., p for ER, d for
        regular, SBM parameters).

    Examples
    --------
    >>> dm = SingleGraphDataModule(
    ...     graph_type="sbm",
    ...     n=50,
    ...     num_train_samples=1000,
    ...     num_val_samples=100,
    ...     num_test_samples=100,
    ...     batch_size=16,
    ...     p_intra=0.7,
    ...     p_inter=0.05,
    ...     num_blocks=3,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        graph_type: str = "sbm",
        n: int = 50,
        num_train_samples: int = 1000,
        num_val_samples: int = 100,
        num_test_samples: int = 100,
        batch_size: int = 16,
        num_workers: int = 0,
        train_seed: int = 42,
        val_test_seed: int = 123,
        same_graph_all_splits: bool = False,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "digress",
        **graph_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.graph_type = graph_type.lower()
        self.n = n
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_seed = train_seed
        self.val_test_seed = val_test_seed
        self.same_graph_all_splits = same_graph_all_splits
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.noise_type = noise_type
        self.graph_kwargs = graph_kwargs

        self.train_graph: Optional[np.ndarray] = None
        self.val_graph: Optional[np.ndarray] = None
        self.test_graph: Optional[np.ndarray] = None

    def _generate_graph(self, seed: int) -> np.ndarray:
        """Generate a single graph of the specified type.

        Parameters
        ----------
        seed : int
            Random seed for generation.

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n, n).
        """
        if self.graph_type == "sbm":
            # Generate SBM graph
            p_intra = self.graph_kwargs.get("p_intra", 0.7)
            p_inter = self.graph_kwargs.get("p_inter", 0.05)
            num_blocks = self.graph_kwargs.get("num_blocks", 3)

            rng = np.random.default_rng(seed)
            # Roughly equal block sizes
            block_sizes = [self.n // num_blocks] * num_blocks
            block_sizes[-1] += self.n - sum(block_sizes)

            A = generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng)
            # Ensure symmetry (use upper triangle)
            A = np.triu(A) + np.triu(A, 1).T
            np.fill_diagonal(A, 0)  # No self-loops
            return A.astype(np.float32)

        elif self.graph_type == "erdos_renyi":
            p = self.graph_kwargs.get("p", 0.1)
            A = generate_erdos_renyi_graphs(self.n, p, 1, seed=seed)[0]
            return A

        elif self.graph_type == "regular":
            d = self.graph_kwargs.get("d", 3)
            A = generate_regular_graphs(self.n, d, 1, seed=seed)[0]
            return A

        elif self.graph_type == "tree":
            A = generate_tree_graphs(self.n, 1, seed=seed)[0]
            return A

        elif self.graph_type == "ring_of_cliques":
            num_cliques = self.graph_kwargs.get("num_cliques", 4)
            clique_size = self.graph_kwargs.get("clique_size", 5)
            G = nx.ring_of_cliques(num_cliques, clique_size)
            A = nx.to_numpy_array(G)
            return A.astype(np.float32)

        elif self.graph_type == "lfr":
            # LFR benchmark graph with planted community structure
            tau1 = self.graph_kwargs.get("tau1", 3.0)
            tau2 = self.graph_kwargs.get("tau2", 1.5)
            mu = self.graph_kwargs.get("mu", 0.1)
            avg_degree = self.graph_kwargs.get("average_degree", 5)
            min_community = self.graph_kwargs.get("min_community", 10)
            G = nx.LFR_benchmark_graph(
                self.n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=avg_degree,
                min_community=min_community,
                seed=seed,
            )
            A = nx.to_numpy_array(G)
            return A.astype(np.float32)

        elif self.graph_type.startswith("pyg_"):
            # Load from PyTorch Geometric dataset
            return self._load_pyg_graph(seed)

        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def _load_pyg_graph(self, seed: int) -> np.ndarray:
        """Load a single graph from a PyTorch Geometric dataset.

        Parameters
        ----------
        seed : int
            Used to deterministically select graph index if not specified.

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n, n).
        """
        try:
            from torch_geometric.datasets import ENZYMES, PROTEINS, QM9
            from torch_geometric.utils import to_dense_adj
        except ImportError as e:
            raise ImportError(
                "PyTorch Geometric required for PyG datasets. "
                "Install with: pip install torch-geometric"
            ) from e

        # Map graph_type to dataset class
        dataset_map = {
            "pyg_enzymes": ENZYMES,
            "pyg_proteins": PROTEINS,
            "pyg_qm9": QM9,
        }

        dataset_name = self.graph_type.lower()
        if dataset_name not in dataset_map:
            raise ValueError(
                f"Unknown PyG dataset: {dataset_name}. "
                f"Supported: {list(dataset_map.keys())}"
            )

        # Load dataset (downloads if necessary)
        dataset_cls = dataset_map[dataset_name]
        root = self.graph_kwargs.get("root", f"./data/{dataset_name}")
        dataset = dataset_cls(root=root)

        # Select graph by index or use seed to pick one
        graph_idx = self.graph_kwargs.get("graph_idx", None)
        if graph_idx is None:
            rng = np.random.default_rng(seed)
            graph_idx = rng.integers(0, len(dataset))

        if graph_idx >= len(dataset):
            raise ValueError(
                f"graph_idx {graph_idx} out of range for dataset "
                f"with {len(dataset)} graphs"
            )

        data = dataset[graph_idx]

        # Convert edge_index to dense adjacency matrix
        A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        A = A.numpy().astype(np.float32)

        # Ensure symmetric (undirected)
        A = (A + A.T) / 2
        A = (A > 0).astype(np.float32)

        # Update n to match actual graph size (override user's n)
        self._actual_n = A.shape[0]

        return A

    def setup(self, stage: Optional[str] = None) -> None:
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

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.train_graph, self.num_train_samples)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.val_graph, self.num_val_samples)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_graph is None:
            raise RuntimeError("Call setup() before accessing dataloaders")

        dataset = SingleGraphDataset(self.test_graph, self.num_test_samples)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_train_graph(self) -> np.ndarray:
        """Return the training graph.

        Returns
        -------
        np.ndarray
            Training adjacency matrix.
        """
        if self.train_graph is None:
            raise RuntimeError("Call setup() before accessing graphs")
        return self.train_graph

    def get_val_graph(self) -> np.ndarray:
        """Return the validation graph.

        Returns
        -------
        np.ndarray
            Validation adjacency matrix.
        """
        if self.val_graph is None:
            raise RuntimeError("Call setup() before accessing graphs")
        return self.val_graph

    def get_test_graph(self) -> np.ndarray:
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
