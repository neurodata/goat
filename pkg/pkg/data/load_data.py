from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.utils import Bunch
import numpy as np


DATA_VERSION = "2021-03-02"  # set to whatever the most recent one is

DATA_PATH = Path(__file__).parent.parent.parent.parent  # don't judge me judge judy
DATA_PATH = DATA_PATH / "data"

from pathlib import Path
import networkx as nx

# from src.utils import meta_to_array
import numpy as np
import pandas as pd


def _get_folder(path, dataset, version):
    if path is None:
        path = DATA_PATH
    if version is None:
        version = DATA_VERSION
    folder = path / dataset
    folder = folder / version
    return folder


def load_node_meta(dataset=None, path=None, version=None):
    folder = _get_folder(path, dataset, version)
    meta = pd.read_csv(folder / "meta_data.csv", index_col=0)
    meta.sort_index(inplace=True)
    return meta


def load_edgelist(dataset="maggot", graph_type="G", path=None, version=None):
    folder = _get_folder(path, dataset, version)
    edgelist = pd.read_csv(
        folder / f"{graph_type}_edgelist.txt",
        delimiter=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    return edgelist


def load_networkx(
    dataset="maggot", graph_type="G", node_meta=None, path=None, version=None
):
    edgelist = load_edgelist(
        dataset=dataset, graph_type=graph_type, path=path, version=version
    )
    g = nx.from_pandas_edgelist(edgelist, edge_attr="weight", create_using=nx.DiGraph())
    if node_meta is not None:
        meta_data_dict = node_meta.to_dict(orient="index")
        nx.set_node_attributes(g, meta_data_dict)
    return g


def load_adjacency(
    dataset="maggot",
    graph_type="G",
    nodelist=None,
    output="numpy",
    path=None,
    version=None,
):
    g = load_networkx(
        dataset=dataset, graph_type=graph_type, path=path, version=version
    )
    if output == "numpy":
        adj = nx.to_numpy_array(g, nodelist=nodelist)
    elif output == "pandas":
        adj = nx.to_pandas_adjacency(g, nodelist=nodelist)
    return adj


# def load_networkx(graph_type, base_path=None, version=DATA_VERSION):
#     if base_path is None:
#         base_path = DATA_PATH
#     data_path = Path(base_path)
#     data_path = data_path / version
#     file_path = data_path / (graph_type + ".graphml")
#     graph = nx.read_graphml(file_path, node_type=str, edge_key_type="str")
#     return graph


def load_data(graph_type, base_path=None, version=None):
    if base_path is None:
        base_path = DATA_PATH
    if version is None:
        version = DATA_VERSION

    data_path = Path(base_path)
    data_path = data_path / version

    edgelist_path = data_path / (graph_type + ".edgelist")
    meta_path = data_path / "meta_data.csv"

    graph = nx.read_edgelist(
        edgelist_path, create_using=nx.DiGraph, nodetype=int, data=[("weight", int)]
    )
    meta = pd.read_csv(meta_path, index_col=0)
    adj = nx.to_numpy_array(graph, nodelist=meta.index.values, dtype=float)
    missing_nodes = np.setdiff1d(meta.index, list(graph.nodes()))
    for node in missing_nodes:
        graph.add_node(node)

    return Bunch(graph=graph, adj=adj, meta=meta)
