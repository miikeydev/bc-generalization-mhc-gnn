from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from .features import build_node_features

try:
    import networkit as nk
    NETWORKIT_AVAILABLE = True
except ImportError:
    NETWORKIT_AVAILABLE = False


def build_real_graph_data(
    dataset_name: str,
    feature_mode: str,
    lap_pe_dim: int,
    random_feature_dim: int,
    rng_seed: int,
    root: str = "data/real_graphs",
    feature_config: dict | None = None,
    bc_backend: str = "networkx",
    bc_mode: str = "exact",
) -> Data:
    graph, original_x = _load_planetoid_graph(dataset_name=dataset_name, root=root)
    rng = np.random.default_rng(rng_seed)
    feature_cfg = feature_config or {}
    
    x = build_node_features(
        graph=graph,
        mode=feature_mode,
        lap_pe_dim=lap_pe_dim,
        random_feature_dim=random_feature_dim,
        rng=rng,
        feature_config=feature_cfg,
    )
    y_raw = _compute_exact_bc(graph, bc_backend=bc_backend, bc_mode=bc_mode)

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


def _compute_exact_bc(graph: nx.Graph, bc_backend: str = "networkx", bc_mode: str = "exact") -> np.ndarray:
    backend = str(bc_backend).lower()
    
    if backend == "auto":
        backend = "networkit" if NETWORKIT_AVAILABLE and graph.number_of_nodes() > 10000 else "networkx"
    
    if backend == "networkit" and NETWORKIT_AVAILABLE:
        try:
            G = nk.nxadapter.nx2nk(graph)
            bc = nk.centrality.Betweenness(G, normalized=True).run().scores()
            return np.array(bc, dtype=np.float32)
        except Exception:
            pass
    
    bc = nx.betweenness_centrality(graph, normalized=True)
    return np.array([bc[node] for node in graph.nodes()], dtype=np.float32)


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    reverse_edges = edges[:, [1, 0]]
    undirected_edges = np.concatenate([edges, reverse_edges], axis=0)
    return torch.from_numpy(undirected_edges.T).long()
