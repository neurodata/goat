from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.utils import Bunch
import numpy as np


DATA_VERSION = "2020-09-23"  # set to whatever the most recent one is

DATA_PATH = Path(__file__).parent.parent.parent.parent  # don't judge me judge judy
DATA_PATH = DATA_PATH / "data"
DATA_PATH = DATA_PATH / "processed"


def load_networkx(graph_type, base_path=None, version=DATA_VERSION):
    if base_path is None:
        base_path = DATA_PATH
    data_path = Path(base_path)
    data_path = data_path / version
    file_path = data_path / (graph_type + ".graphml")
    graph = nx.read_graphml(file_path, node_type=str, edge_key_type="str")
    return graph


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
