#!/usr/bin/env python
# coding: utf-8

# In[34]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from torch.utils.data import Dataset, ConcatDataset
import math


# In[55]:


class BaseNxSet(Dataset):
    def draw(self):
        N = len(self)
        a = int(math.ceil(math.sqrt(N)))
        fig, ax = plt.subplots(a, a, figsize=[a * 2, a * 2])
        for a, ig in zip(ax.flatten() if N > 1 else [ax], range(N)):
            nx.draw(self[ig][1], ax=a)
        fig.tight_layout()
        return fig, ax


# In[58]:


class RingOfCliques(BaseNxSet):
    def __init__(self, num_cliques=2, clique_sizes=2, repeat=1):
        super().__init__()
        if isinstance(num_cliques, int):
            num_cliques = [num_cliques]
        if isinstance(clique_sizes, int):
            clique_sizes = [clique_sizes for _ in num_cliques]
        assert len(num_cliques) == len(clique_sizes)
        self.num_cliques = num_cliques
        self.clique_sizes = clique_sizes
        self.graphs = [
            nx.ring_of_cliques(n, c) for n, c in zip(num_cliques, clique_sizes)
        ] * repeat

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


class LolliPop(BaseNxSet):
    def __init__(self, N_path=2, N_cluster=3, repeat=1):
        super().__init__()
        if isinstance(N_path, int):
            N_path = [N_path]
        if isinstance(N_cluster, int):
            N_cluster = [N_cluster for _ in N_path]
        assert len(N_path) == len(N_cluster)
        self.N_path = N_path
        self.N_cluster = N_cluster
        self.graphs = [
            nx.lollipop_graph(k, p) for p, k in zip(N_path, N_cluster)
        ] * repeat

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


# In[92]:


class CircularLadder(BaseNxSet):
    def __init__(self, N_nodes=3, repeat=1):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [nx.circular_ladder_graph(n) for n in N_nodes] * repeat

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


# In[97]:
def factor_pairs(N):
    c = np.array(list(range(2, N)))
    factor = N % c == 0
    factors = c[factor]
    other = N / factors
    factors: np.array
    return factors.astype(np.int), other.astype(np.int)


class StarGraph(BaseNxSet):
    def __init__(self, N_nodes=3, repeat=1):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs = [nx.star_graph(n) for n in N_nodes] * repeat

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


class RocAndCircL(BaseNxSet):
    def __init__(self, Ns, repeat=1):
        if isinstance(Ns, int):
            Ns = [Ns]
        self.N = Ns
        self.roc = RingOfCliques(*self.factors(Ns))
        self.cla = CircularLadder([x // 2 for x in Ns])
        self.data = ConcatDataset([self.roc, self.cla] * repeat)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def factors(self, Ns):
        fs, os = [], []
        for n in Ns:
            f, o = factor_pairs(n)
            fs.append(f)
            os.append(o)
        fs = np.concatenate(fs)
        os = np.concatenate(os)
        return fs, os

class SquareGrid(BaseNxSet):
    def __init__(self, N_nodes=1000, repeat=5000):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs=[self._g_for_N(n) for n in N_nodes]*repeat
    def _g_for_N(self,N_nodes):
        h=int(math.ceil(math.sqrt(N_nodes)))
        w=int(math.floor(N_nodes/h))
        while nx.number_of_nodes(nx.grid_graph([h,w]))<N_nodes:
            w+=1
        return nx.grid_graph([h,w])

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
class TriangleGrid(BaseNxSet):
    def __init__(self, N_nodes=1000, repeat=5000):
        super().__init__()
        if isinstance(N_nodes, int):
            N_nodes = [N_nodes]
        self.N_nodes = N_nodes
        self.graphs=[self._g_for_N(n) for n in N_nodes]*repeat
    def _g_for_N(self,N_nodes):
        x=int(math.ceil(math.sqrt(N_nodes)))
        while nx.number_of_nodes(nx.triangular_lattice_graph(x,x))<N_nodes:
            x+=1
        return nx.triangular_lattice_graph(x,x)

    def __getitem__(self, idx):
        return None, self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


NX_CLASSES = dict(
    nx_star=StarGraph,
    nx_circ_ladder=CircularLadder,
    nx_lollipop=LolliPop,
    nx_roc=RingOfCliques,
    nx_combo=RocAndCircL,
    nx_triangle=TriangleGrid,
    nx_square = SquareGrid
)


class NXGraphWrapper(Dataset):
    def __init__(self, clsname, *dataset_args, **dataset_kwargs):
        self.cls = NX_CLASSES[clsname]
        self.data = self.cls(*dataset_args, **dataset_kwargs)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        X, g = self.data[item]
        A = nx.to_numpy_array(g)
        return X, A


# In[ ]:

if __name__=="__main__":
    x=SquareGrid(N_nodes=20,repeat=5)
    g=x[0]