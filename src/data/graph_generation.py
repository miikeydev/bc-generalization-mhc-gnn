from __future__ import annotations

import math

import networkx as nx
import numpy as np


SUPPORTED_FAMILIES = {"er", "ba", "sbm", "ws", "rgg"}


def generate_connected_graph(
    family: str,
    num_nodes: int,
    average_degree: int,
    seed: int,
    max_attempts: int = 12,
) -> nx.Graph:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported graph family: {family}")

    rng = np.random.default_rng(seed)
    last_graph: nx.Graph | None = None

    for _ in range(max_attempts):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        graph = _sample_graph(family, num_nodes, average_degree, graph_seed)
        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph = nx.convert_node_labels_to_integers(graph)

        if graph.number_of_nodes() == 0:
            continue

        if nx.is_connected(graph):
            return graph

        largest_component = max(nx.connected_components(graph), key=len)
        candidate = graph.subgraph(largest_component).copy()
        candidate = nx.convert_node_labels_to_integers(candidate)
        last_graph = candidate

        if candidate.number_of_nodes() >= max(8, int(0.9 * num_nodes)):
            return candidate

    if last_graph is not None:
        return last_graph

    raise RuntimeError("Failed to generate a non-empty graph")


def _sample_graph(family: str, num_nodes: int, average_degree: int, seed: int) -> nx.Graph:
    if family == "er":
        p = min(1.0, max(0.0, average_degree / max(1, num_nodes - 1)))
        return nx.erdos_renyi_graph(n=num_nodes, p=p, seed=seed)

    if family == "ba":
        m = max(1, min(num_nodes - 1, int(round(average_degree / 2))))
        return nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)

    if family == "sbm":
        num_blocks = 4
        sizes = [num_nodes // num_blocks] * num_blocks
        for idx in range(num_nodes % num_blocks):
            sizes[idx] += 1

        base_p = average_degree / max(1, num_nodes - 1)
        p_in = min(0.9, max(0.01, 3.0 * base_p))
        p_out = min(0.2, max(0.0005, base_p / 3.0))

        probs = np.full((num_blocks, num_blocks), p_out, dtype=float)
        np.fill_diagonal(probs, p_in)
        return nx.stochastic_block_model(sizes=sizes, p=probs.tolist(), seed=seed)

    if family == "ws":
        k = int(round(average_degree))
        k = min(num_nodes - 1, max(2, k))
        if k % 2 == 1:
            k = max(2, k - 1)
        return nx.watts_strogatz_graph(n=num_nodes, k=k, p=0.2, seed=seed)

    radius = math.sqrt(average_degree / max(1.0, (num_nodes - 1) * math.pi))
    return nx.random_geometric_graph(n=num_nodes, radius=radius, seed=seed)
