from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from .features import build_node_features


def build_real_graph_data(
    dataset_name: str,
    feature_mode: str,
    lap_pe_dim: int,
    random_feature_dim: int,
    rng_seed: int,
    root: str = "data/real_graphs",
) -> Data:
    graph, original_x = _load_planetoid_graph(dataset_name=dataset_name, root=root)
    rng = np.random.default_rng(rng_seed)
    x = build_node_features(
        graph=graph,
        mode=feature_mode,
        lap_pe_dim=lap_pe_dim,
        random_feature_dim=random_feature_dim,
        rng=rng,
    )
    y_raw = _compute_exact_bc(graph)

    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=_edge_index_from_graph(graph),
    )
    data.y_raw = torch.from_numpy(y_raw).float()
    data.num_nodes = int(graph.number_of_nodes())
    data.num_edges_undirected = int(graph.number_of_edges())
    data.dataset_name = dataset_name.lower()
    data.original_x = original_x
    return data


def _load_planetoid_graph(dataset_name: str, root: str) -> tuple[nx.Graph, torch.Tensor]:
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    return graph, data.x.detach().cpu()


def _compute_exact_bc(graph: nx.Graph) -> np.ndarray:
    bc = nx.betweenness_centrality(graph, normalized=True)
    return np.array([bc[node] for node in graph.nodes()], dtype=np.float32)


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    reverse_edges = edges[:, [1, 0]]
    undirected_edges = np.concatenate([edges, reverse_edges], axis=0)
    return torch.from_numpy(undirected_edges.T).long()
