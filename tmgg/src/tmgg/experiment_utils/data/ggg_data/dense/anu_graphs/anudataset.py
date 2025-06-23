"""
A dataset which accessible all the graphs at https://users.cecs.anu.edu.au/~bdm/data/graphs.html
"""
import os
import datetime
import torch as pt
import numpy as np
import networkx as nx
import subprocess as sp
from urllib.parse import urlparse

from torch.utils.data import Dataset
from tqdm import tqdm

ANU_BASE_URL = "https://users.cecs.anu.edu.au/~bdm/data/"


def ANU_URL(f):
    return f"{ANU_BASE_URL}/{f}"


def url_file(url):
    """
    Extracts the filename from a url, assuming the url ends in it
    :param url:
    :return:
    """
    return os.path.basename(urlparse(url).path)


def triangles_(adj_matrix, k_, prev_k=None):
    if prev_k is None:
        k_matrix = pt.matrix_power(adj_matrix.float(), k_)
    else:
        k_matrix = prev_k @ adj_matrix.float()
    egd_l = pt.diagonal(k_matrix, dim1=-2, dim2=-1)
    return egd_l, k_matrix

from pathlib import Path

parent_dir = str(str(Path(os.path.abspath(__file__)).parents[2]))

GRAPH_FILES = dict(
    # simple graphs between 2 and 11 vertices, all and connectd only repectively
    simple_all=[(f"graph{i}.g6") for i in range(2, 10)]
    + [[(f"graph{i}.g6.gz") for i in range(10, 12)]],
    simple_connected=[(f"graph{i}c.g6") for i in range(2, 10)]
    + [(f"graph{i}c.g6.gz") for i in range(10, 12)],
    # eulerian graphs, i.e. simple with every degree even
    eulerian_all=[f"eul{i}.g6" for i in range(2, 11)]
    + ["eul11.g6.gz"]
    + [f"euler12_{i}.g6.gz" for i in range(1, 5)],
    eulerian_connected=[f"eul{i}c.g6" for i in range(2, 11)] + ["eul11c.g6.gz"],
    # every cycle >4 has a chord
    chordal=[f"chordal{i}.g6{'.gz' if i > 11 else ''}" for i in range(4, 14)],
    perfect=[f"perfect{i}.g6{'.gz' if i > 10 else ''}" for i in range(5, 12)],
    # hyphamiltonian: not hamiltonian but if you remove one vertex it's hamiltonian
    hypo=[f"hypo{i}{'some' if i > 16 else ''}.g6" for i in [10, 13, 15, 16, 18, 22, 26]]
    + [
        f"cubhypo{i}{'g5' if i == 28 else ('g6' if i == 30 else '')}.s6"
        for i in [10, 18, 20, 22, 24, 26, 28, 30]
    ],
    semi_regular=[
        "sr25832.g6",
        "sr251256.g6",
        "sr261034.g6",
        "sr271015.g6",
        "sr281264.g6",
        "sr291467.g6",
        "sr351668.g6",
        "sr351899.g6",
        "sr361446.g6",
        "sr361566.g6",
        "sr361566rep.g6",
        "sr371889some.g6",
        "sr401224.g6",
    ],
    # non-isomorphic connected planar graphs
    planar=[f"planar_conn.{i}.g6{'.gz' if i == 10 else ''}" for i in range(1, 11)]
    + [f"planar_conn.11{c}.g6.gz" for c in "ab"],
    ramsey=[
        "r34_1.g6,"
        "r34_2.g6,"
        "r34_3.g6,"
        "r34_4.g6,"
        "r34_5.g6,"
        "r34_6.g6,"
        "r34_7.g6,"
        "r34_8.g6,"
        "r35_1.g6,"
        "r35_2.g6,"
        "r35_3.g6,"
        "r35_4.g6,"
        "r35_5.g6,"
        "r35_6.g6,"
        "r35_7.g6,"
        "r35_8.g6,"
        "r35_9.g6,"
        "r35_10.g6,"
        "r35_11.g6,"
        "r35_12.g6,"
        "r35_13.g6,"
        "r36_1.g6,"
        "r36_2.g6,"
        "r36_3.g6,"
        "r36_4.g6,"
        "r36_5.g6,"
        "r36_6.g6,"
        "r36_7.g6,"
        "r36_8.g6,"
        "r36_9.g6,"
        "r36_10.g6,"
        "r36_17.g6,"
        "r37_22.g6,"
        "r39_35.g6,"
        "r44_1.g6,"
        "r44_2.g6,"
        "r44_3.g6,"
        "r44_4.g6,"
        "r44_5.g6,"
        "r44_6.g6,"
        "r44_7.g6,"
        "r44_8.g6,"
        "r44_9.g6,"
        "r44_15.g6,"
        "r44_16.g6,"
        "r44_17.g6,"
        "r45_24.g6,"
        "r46_35some.g6,"
        "r55_42some.g6,"
    ],
    highly_irregular=[f"highlyirregular{i}.g6" for i in range(1, 16)],
    # G isomorphic to its complements
    self_complementary=[f"selfcomp{i}.g6" for i in [4, 5, 8, 9, 12, 13]]
    + ["selfcomp16.g6.gz"]
    + [f"selfcomp17{c}.g6.gz" for c in "abc"]
    + [[f"selfcomp20{c}.g6.gz" for c in "abcd"]],
    # "We will call an undirected simple graph G edge-4-critical if it is connected, is not (vertex) 3-colourable, and G-e is 3-colourable for every edge e."
    edge4_critical=[f"crit4.{i}.g6" for i in filter(lambda x: x != 5, range(4, 14))]
    + ["crit4.11g4.g6", "crit4.21g5.g6", "crit4.pm4_18.g6"],
)


def download_g6_files(datasets, root_path, parallel=True, exclude_files=()):
    """
    Gets the g6 files from  https://users.cecs.anu.edu.au/~bdm/data/graphs.html in a folder structure of
    root_path/datasetname/files.g6
    and uncompresses .gz files
    :param file_dict:
    :param root_path:
    :return:
    """
    old_path = os.path.abspath(os.getcwd())
    os.makedirs(root_path, exist_ok=True)
    root_abs = os.path.abspath(root_path)
    os.chdir(root_abs)
    file_dict = {k: GRAPH_FILES[k] for k in datasets}
    for k, files in file_dict.items():
        os.makedirs(k, exist_ok=True)
        os.chdir(k)
        dl_procs = []
        for f in files:
            if not (
                os.path.exists(f)
                or f.replace(".gz", "") in exclude_files
                or os.path.exists(k.replace(".gz", ""))
            ):
                p = sp.Popen(["wget", ANU_URL(f)])
                if parallel:
                    dl_procs.append(p)
                else:
                    p.wait()
        for p in dl_procs:
            p.wait()

        for f in tqdm(
            [x for x in os.listdir("") if ".gz" == os.path.splitext(x)[-1]],
            "Unzipping gz files",
        ):
            if not os.path.exists(k.replace(".gz", "")):
                sp.run(["gunzip", "-k", f])

        os.chdir(root_abs)
    os.chdir(old_path)


def convert_g6_to_dense(datasets, root_path, exclude_files=(), create_rand=False):
    """
    Converts the g6 datasets to numpy arrays. Stores all graphs of one dataset in a single .npz array
    :param datasets:
    :param root_path:
    :return:
    """
    dbar = tqdm(datasets)
    for d in dbar:
        dbar.set_description(f"Converting {d}")
        base_f = os.path.join(root_path, f"{d}.npz")
        if os.path.exists(base_f):
            continue

        def filter_files(f):
            keep = os.path.splitext(f)[-1] in {".g6", ".s6"} and f not in exclude_files
            return keep

        all_files = [k for k in os.listdir(os.path.join(root_path, d))]
        all_files = list(filter(filter_files, all_files))
        all_files = [os.path.abspath(os.path.join(root_path, d, k)) for k in all_files]
        X_max_shape = None
        A_max_shape = None
        Xs = []
        As = []
        # read in all files
        fbar = tqdm(all_files, desc="File", leave=False)
        for f in fbar:
            fbar.set_description(f"Reading file {os.path.join(d, f)}")
            graphs = nx.read_graph6(f)
            if create_rand:
                graphs = graphs[5000:]
            for idx, g in tqdm(enumerate(graphs), desc="Graph"):
                A = nx.to_numpy_array(g)
                X = get_X(A)
                if X_max_shape:
                    for i in range(len(X_max_shape)):
                        new = X.shape[i]
                        if new > X_max_shape[i]:
                            X_max_shape[i] = new
                else:
                    X_max_shape = [x for x in X.shape]
                if A_max_shape:
                    for i in range(len(A_max_shape)):
                        new = A.shape[i]
                        if new > A_max_shape[i]:
                            A_max_shape[i] = new
                else:
                    A_max_shape = [x for x in A.shape]
                Xs.append(X)
                As.append(A)

                # TODO need to add size flag for size
                if idx + 1 == 5000 and not create_rand:
                    break

        As_ = np.zeros([len(As)] + A_max_shape)
        Xs_ = np.zeros([len(Xs)] + X_max_shape)
        for i, (X, A) in tqdm(enumerate(zip(Xs, As)), desc="Finalizing batch"):
            num_nodes, node_dim = X.shape
            As_[i, :num_nodes, :num_nodes] = A
            Xs_[i, :num_nodes, :node_dim] = X
        # save complete dataset as single npz compressed archive
        np.savez_compressed(
            f"{base_f}",
            A=As_,
            X=Xs_,
            Xshape=np.array(Xs_.shape),
            Ashape=np.array(As_.shape),
        )


def load_npz_keys(keys, file):
    """
    Small utility to directly load an npz_compressed file
    :param keys:
    :param file:
    :return:
    """
    out = []
    with np.load(file) as d:
        for k in keys:
            out.append(d[k])
    return tuple(out) if len(out) > 1 else out[0]


def get_X(A):
    """
    Gets primitive Node features from a graph
    :param graph:
    :return:
    """

    X = pt.tensor([])
    k_matrix = None
    Asize = A.shape
    triang_adj = pt.from_numpy(A).clone()
    pt.diagonal(triang_adj, dim1=-2, dim2=-1).fill_(0)
    for k_ in range(2, 7 + 1):
        paths, k_matrix = triangles_(triang_adj, k_, prev_k=k_matrix)
        X = pt.cat((X, paths.reshape(Asize[0], -1)), -1)

    return X.numpy()


class ANUDataset(Dataset):
    def __init__(
        self,
        datasets=None,
        data_dir=None,
        num_graphs_per_set=-1,
        exclude_files=None,
        create_rand=False,
    ):
        super().__init__()
        if datasets is None:
            datasets = list(GRAPH_FILES.keys())
        exclude_files = exclude_files if exclude_files is not None else set()
        exclude_files = set([x.replace(".gz", "") for x in exclude_files])
        assert all(d in GRAPH_FILES for d in datasets)
        if data_dir is None:
            data_dir = "anu_graphs"
        self.Py_data = []
        self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.num_graphs_per_set = num_graphs_per_set
        self.datasets = datasets
        download_g6_files(
            datasets, data_dir, parallel=False, exclude_files=exclude_files
        )
        convert_g6_to_dense(
            datasets, data_dir, exclude_files=exclude_files, create_rand=create_rand
        )
        self.load()

    def load(self):
        max_shape = [0, 0, 0]
        num_graphs = 0
        for d in self.datasets:
            Xshape = load_npz_keys(["Xshape"], os.path.join(self.data_dir, f"{d}.npz"))
            max_shape = [max(o, n) for o, n in zip(max_shape, Xshape)]
            N = (
                min(Xshape[0], self.num_graphs_per_set)
                if self.num_graphs_per_set > 0
                else Xshape[0]
            )
            num_graphs += N
        # intialize dense graph combination
        self.X = np.zeros([num_graphs, max_shape[1], max_shape[2]])
        self.A = np.zeros([num_graphs, max_shape[1], max_shape[1]])
        self.num_graphs = 0
        # actually assign the loaded X,A
        for d in self.datasets:
            X, A = load_npz_keys(["X", "A"], os.path.join(self.data_dir, f"{d}.npz"))
            N = (
                min(X.shape[0], self.num_graphs_per_set)
                if self.num_graphs_per_set > 0
                else X.shape[0]
            )
            self.X[
                self.num_graphs : self.num_graphs + N, : X.shape[1], : X.shape[2]
            ] = X[:N]
            self.A[
                self.num_graphs : self.num_graphs + N, : X.shape[1], : X.shape[1]
            ] = A[:N]
            self.num_graphs += N

    @staticmethod
    def log(msg="", date=True):
        print(
            str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + " " + str(msg)
            if date
            else str(msg)
        )

    def __getitem__(self, item):
        return self.X[item], self.A[item]

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    a = ANUDataset(
        data_dir="",
        datasets=["chordal"],
        exclude_files=[
            f"chordal{i}.g6.gz" for i in filter(lambda x: x != 9, range(4, 14))
        ],
    )
    print(a[0])
