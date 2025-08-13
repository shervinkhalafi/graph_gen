import torch as pt
import numpy as np
import networkx as nx
import random
from torch.utils.data import Dataset
import torch.distributions as td


class CassicalGraphs(Dataset):
    def __getitem__(self, index: int):
        x = self.X[index]
        A = self.A[index]
        return x, A

    def __len__(self) -> int:
        return self.n_graphs

    def generate(self, node_dist: td.Categorical):
        node_N = (2 + node_dist.sample([self.n_graphs]).int()).tolist()
        if self.model == "ER":
            ps = np.random.rand(self.n_graphs) * 0.9
            self.A = [
                nx.to_numpy_array(nx.fast_gnp_random_graph(n, p))
                for n, p in zip(node_N, ps)
            ]
            node_N = np.array(node_N).astype(np.float)
            self.X = np.stack([node_N, ps], -1)
        elif self.model == "BA":
            Ns = np.array(node_N).astype(np.float)
            ms = np.floor(np.random.rand(self.n_graphs) * 0.9 * Ns).astype(np.int32)
            self.A = [
                nx.to_numpy_array(nx.barabasi_albert_graph(n, int(m)))
                for n, m in zip(node_N, ms)
            ]
            self.X = np.stack([Ns, ms], -1)
        else:
            raise NotImplementedError()

    def __init__(
        self, n_graphs, max_nodes, node_N_weights=None, model="ER", seed=0
    ) -> None:
        super().__init__()
        assert model in {"ER", "BA"}
        assert max_nodes >= 2
        self.model = model
        self.seed = seed
        self.n_graphs = n_graphs
        if node_N_weights is None:
            node_N_weights = np.ones(max_nodes - 1) / max_nodes - 1
        node_dist = td.Categorical(probs=pt.from_numpy(node_N_weights))
        old_state_np = np.random.get_state()
        old_state_py = random.getstate()
        old_state_pt = pt.random.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        pt.random.manual_seed(seed)
        self.generate(node_dist)
        random.setstate(old_state_py)
        np.random.set_state(old_state_np)
        pt.random.set_rng_state(old_state_pt)


if __name__ == "__main__":
    cb = CassicalGraphs(10, 20, model="ER")
    print([x.sum(-1) for x in cb[0]])
    cb = CassicalGraphs(10, 20, model="BA")
    print([x.sum(-1) for x in cb[0]])
