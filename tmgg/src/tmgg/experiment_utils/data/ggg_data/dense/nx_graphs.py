#!/usr/bin/env python
from __future__ import annotations

# In[34]:
import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, override

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.utils.data import ConcatDataset, Dataset

# nx.Graph is not subscriptable at runtime, so we use TYPE_CHECKING guard
# to provide type info only during static analysis
if TYPE_CHECKING:
    _NxDatasetBase = Dataset[tuple[npt.NDArray[np.float64] | None, nx.Graph[Any]]]
else:
    _NxDatasetBase = Dataset

# In[55]:


class BaseNxSet(_NxDatasetBase):
    @abstractmethod
    def __len__(self) -> int:
        """Return number of graphs in dataset."""
        ...

    def draw(self) -> tuple[Figure, Axes | npt.NDArray[Any]]:
        num_graphs = len(self)
        a = int(math.ceil(math.sqrt(num_graphs)))
        fig, ax = plt.subplots(a, a, figsize=[a * 2, a * 2])
        for a_ax, ig in zip(
            ax.flatten() if num_graphs > 1 else [ax], range(num_graphs), strict=False
        ):
            nx.draw(self[ig][1], ax=a_ax)
        fig.tight_layout()
        return fig, ax


# In[58]:


class RingOfCliques(BaseNxSet):
    num_cliques: list[int]
    clique_sizes: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(
        self,
        num_cliques: int | list[int] = 2,
        clique_sizes: int | list[int] = 2,
        repeat: int = 1,
    ):
        super().__init__()
        if isinstance(num_cliques, int):
            num_cliques = [num_cliques]
        if isinstance(clique_sizes, int):
            clique_sizes = [clique_sizes for _ in num_cliques]
        assert len(num_cliques) == len(clique_sizes)
        self.num_cliques = num_cliques
        self.clique_sizes = clique_sizes
        self.graphs = [
            nx.ring_of_cliques(n, c)
            for n, c in zip(num_cliques, clique_sizes, strict=False)
        ] * repeat

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


class LolliPop(BaseNxSet):
    N_path: list[int]
    N_cluster: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(
        self,
        N_path: int | list[int] = 2,
        N_cluster: int | list[int] = 3,
        repeat: int = 1,
    ):
        super().__init__()
        if isinstance(N_path, int):
            N_path = [N_path]
        if isinstance(N_cluster, int):
            N_cluster = [N_cluster for _ in N_path]
        assert len(N_path) == len(N_cluster)
        self.N_path = N_path
        self.N_cluster = N_cluster
        self.graphs = [
            nx.lollipop_graph(k, p) for p, k in zip(N_path, N_cluster, strict=False)
        ] * repeat

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


# In[92]:


class CircularLadder(BaseNxSet):
    N_nodes: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(self, N_nodes: int | list[int] = 3, repeat: int = 1):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [nx.circular_ladder_graph(n) for n in N_nodes] * repeat

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


# In[97]:
def factor_pairs(N: int) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    c = np.array(list(range(2, N)))
    factor = N % c == 0
    factors = c[factor]
    other = N / factors
    return factors.astype(np.int_), other.astype(np.int_)


class StarGraph(BaseNxSet):
    N_nodes: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(
        self,
        N_nodes: int | list[int] | None = None,
        N_arms: int | None = None,
        repeat: int = 1,
    ):
        super().__init__()
        # Support both N_nodes and N_arms for compatibility
        if N_arms is not None:
            N_nodes = N_arms
        if N_nodes is None:
            N_nodes = 3
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [nx.star_graph(n) for n in N_nodes] * repeat

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


class RocAndCircL(BaseNxSet):
    node_counts: list[int]
    roc: RingOfCliques
    cla: CircularLadder
    data: ConcatDataset[tuple[None, nx.Graph[Any]]]

    def __init__(self, Ns: int | list[int], repeat: int = 1):
        if isinstance(Ns, int):
            Ns = [Ns]
        self.node_counts = Ns
        num_cliques, clique_sizes = self.factors(Ns)
        self.roc = RingOfCliques(num_cliques.tolist(), clique_sizes.tolist())
        self.cla = CircularLadder([x // 2 for x in Ns])
        self.data = ConcatDataset([self.roc, self.cla] * repeat)

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return self.data[idx]

    @override
    def __len__(self) -> int:
        return len(self.data)

    def factors(
        self, Ns: list[int]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        fs: list[npt.NDArray[np.int_]] = []
        os: list[npt.NDArray[np.int_]] = []
        for n in Ns:
            f, o = factor_pairs(n)
            fs.append(f)
            os.append(o)
        fs_concat = np.concatenate(fs)
        os_concat = np.concatenate(os)
        return fs_concat, os_concat


class SquareGrid(BaseNxSet):
    N_nodes: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(self, N_nodes: int | list[int] = 1000, repeat: int = 5000):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [self._g_for_N(n) for n in N_nodes] * repeat

    def _g_for_N(self, N_nodes: int) -> nx.Graph[Any]:
        h = int(math.ceil(math.sqrt(N_nodes)))
        w = int(math.floor(N_nodes / h))
        while nx.number_of_nodes(nx.grid_graph([h, w])) < N_nodes:
            w += 1
        return nx.grid_graph([h, w])

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


class TriangleGrid(BaseNxSet):
    N_nodes: list[int]
    graphs: list[nx.Graph[Any]]

    def __init__(self, N_nodes: int | list[int] = 1000, repeat: int = 5000):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [self._g_for_N(n) for n in N_nodes] * repeat

    def _g_for_N(self, N_nodes: int) -> nx.Graph[Any]:
        x = int(math.ceil(math.sqrt(N_nodes)))
        while nx.number_of_nodes(nx.triangular_lattice_graph(x, x)) < N_nodes:
            x += 1
        return nx.triangular_lattice_graph(x, x)

    @override
    def __getitem__(self, idx: int) -> tuple[None, nx.Graph[Any]]:
        return None, self.graphs[idx]

    @override
    def __len__(self) -> int:
        return len(self.graphs)


NX_CLASSES: dict[str, type[BaseNxSet]] = dict(
    nx_star=StarGraph,
    nx_circ_ladder=CircularLadder,
    nx_lollipop=LolliPop,
    nx_roc=RingOfCliques,
    nx_combo=RocAndCircL,
    nx_triangle=TriangleGrid,
    nx_square=SquareGrid,
)


class NXGraphWrapper(
    Dataset[tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64]]]
):
    cls: type[BaseNxSet]
    data: BaseNxSet

    def __init__(self, clsname: str, *dataset_args: Any, **dataset_kwargs: Any):
        self.cls = NX_CLASSES[clsname]
        self.data = self.cls(*dataset_args, **dataset_kwargs)

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(
        self, item: int
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64]]:
        X, g = self.data[item]
        A = nx.to_numpy_array(g)
        return X, A


# In[ ]:

if __name__ == "__main__":
    x = SquareGrid(N_nodes=20, repeat=5)
    g = x[0]
