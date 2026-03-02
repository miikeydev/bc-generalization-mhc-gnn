from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def build_node_features(
    graph: nx.Graph,
    mode: str,
    lap_pe_dim: int,
    random_feature_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "structural_only":
        degree = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32).reshape(-1, 1)
        log_degree = np.log1p(degree)
        lap_pe = compute_laplacian_positional_encoding(graph, lap_pe_dim)
        return np.concatenate([degree, log_degree, lap_pe], axis=1).astype(np.float32)

    if mode == "random":
        num_nodes = graph.number_of_nodes()
        return rng.normal(0.0, 1.0, size=(num_nodes, random_feature_dim)).astype(np.float32)

    if mode == "none":
        num_nodes = graph.number_of_nodes()
        return np.ones((num_nodes, 1), dtype=np.float32)

    raise ValueError(f"Unsupported feature mode: {mode}")


def compute_laplacian_positional_encoding(graph: nx.Graph, lap_pe_dim: int) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if lap_pe_dim <= 0 or num_nodes <= 1:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    target_dim = min(lap_pe_dim, num_nodes - 1)
    num_eigs = min(num_nodes - 1, target_dim + 1)
    laplacian = nx.normalized_laplacian_matrix(graph).astype(np.float64)

    try:
        _, eigvecs = eigsh(laplacian, k=num_eigs, which="SM")
    except Exception:
        eigvals_dense, eigvecs_dense = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(eigvals_dense)
        eigvecs = eigvecs_dense[:, order][:, :num_eigs]

    if eigvecs.shape[1] > 1:
        lap_pe = eigvecs[:, 1:]
    else:
        lap_pe = np.zeros((num_nodes, 0), dtype=np.float64)

    if lap_pe.shape[1] < target_dim:
        pad = np.zeros((num_nodes, target_dim - lap_pe.shape[1]), dtype=np.float64)
        lap_pe = np.concatenate([lap_pe, pad], axis=1)

    return lap_pe[:, :target_dim].astype(np.float32)
