from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from .betweenness import compute_betweenness_centrality
from .features import build_node_features
from .protocol import compute_size_bucket, normalize_data_config


def build_real_graph_data(
    dataset_name: str,
    feature_mode: str,
    lap_pe_dim: int,
    random_feature_dim: int,
    rng_seed: int,
    root: str = "data/real_graphs",
    feature_config: dict | None = None,
    bc_backend: str = "networkit",
) -> Data:
    data_cfg = normalize_data_config(
        {
            "feature_mode": feature_mode,
            "lap_pe_dim": lap_pe_dim,
            "random_feature_dim": random_feature_dim,
            "feature_config": feature_config or {},
            "bc_backend": bc_backend,
        }
    )
    graph, original_x = _load_planetoid_graph(dataset_name=dataset_name, root=root)
    rng = np.random.default_rng(rng_seed)

    x = build_node_features(
        graph=graph,
        mode=data_cfg["feature_mode"],
        lap_pe_dim=data_cfg["lap_pe_dim"],
        random_feature_dim=data_cfg["random_feature_dim"],
        rng=rng,
        feature_config=data_cfg["feature_config"],
    )
    y_raw = compute_betweenness_centrality(
        graph=graph,
        bc_backend=data_cfg["bc_backend"],
        seed=rng_seed,
    )

    num_nodes_actual = graph.number_of_nodes()
    num_edges_actual = graph.number_of_edges()
    avg_degree_actual = 2.0 * num_edges_actual / max(1, num_nodes_actual)
    density_actual = 2.0 * num_edges_actual / max(1, num_nodes_actual * (num_nodes_actual - 1))

    clustering = nx.average_clustering(graph)
    assortativity = nx.degree_assortativity_coefficient(graph) if num_nodes_actual > 1 else 0.0

    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=_edge_index_from_graph(graph),
    )
    data.y_raw = torch.from_numpy(y_raw).float()
    data.num_nodes = int(num_nodes_actual)
    data.num_edges_undirected = int(num_edges_actual)
    data.avg_degree = float(avg_degree_actual)
    data.density = float(density_actual)
    data.size_bucket = compute_size_bucket(num_nodes_actual)
    data.clustering = float(clustering)
    data.assortativity = float(assortativity)
    data.dataset_name = dataset_name.lower()
    data.original_x = original_x
    return data


def _load_planetoid_graph(dataset_name: str, root: str) -> tuple[nx.Graph, torch.Tensor]:
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    return graph, data.x.detach().cpu()


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    reverse_edges = edges[:, [1, 0]]
    undirected_edges = np.concatenate([edges, reverse_edges], axis=0)
    return torch.from_numpy(undirected_edges.T).long()
