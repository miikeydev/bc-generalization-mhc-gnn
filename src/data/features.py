from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def build_node_features(
    graph: nx.Graph,
    mode: str,
    lap_pe_dim: int = 0,
    random_feature_dim: int = 16,
    rng: np.random.Generator | None = None,
    feature_config: dict | None = None,
) -> np.ndarray:
    feature_cfg = feature_config or {}
    normalized_mode = _normalize_feature_mode(mode)
    
    if normalized_mode == "degree_only":
        return _build_degree_only(graph)
    
    if normalized_mode == "degree_plus_rwpe":
        rwpe_dim = feature_cfg.get("rwpe_dim", 8)
        rwpe_steps = feature_cfg.get("rwpe_steps", 5)
        return _build_degree_plus_rwpe(graph, rwpe_dim, rwpe_steps)
    
    if normalized_mode == "degree_plus_ppr":
        ppr_dim = feature_cfg.get("ppr_dim", 8)
        return _build_degree_plus_ppr(graph, ppr_dim)
    
    if normalized_mode == "structural_only":
        degree = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32).reshape(-1, 1)
        log_degree = np.log1p(degree)
        lap_pe = compute_laplacian_positional_encoding(graph, lap_pe_dim)
        return np.concatenate([degree, log_degree, lap_pe], axis=1).astype(np.float32)

    if normalized_mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        num_nodes = graph.number_of_nodes()
        return rng.normal(0.0, 1.0, size=(num_nodes, random_feature_dim)).astype(np.float32)

    if normalized_mode == "none":
        num_nodes = graph.number_of_nodes()
        return np.ones((num_nodes, 1), dtype=np.float32)

    raise ValueError(f"Unsupported feature mode: {normalized_mode}")


def _normalize_feature_mode(mode: str) -> str:
    mode_lower = str(mode).lower().strip()
    if mode_lower == "structural_only":
        return "structural_only"
    if mode_lower in {"degree_only", "degree"}:
        return "degree_only"
    if mode_lower in {"degree_plus_rwpe", "degree_rwpe"}:
        return "degree_plus_rwpe"
    if mode_lower in {"degree_plus_ppr", "degree_ppr"}:
        return "degree_plus_ppr"
    if mode_lower in {"random", "gaussian"}:
        return "random"
    if mode_lower in {"none", "constant", "ones"}:
        return "none"
    return mode_lower


def _build_degree_only(graph: nx.Graph) -> np.ndarray:
    degree = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32).reshape(-1, 1)
    log_degree = np.log1p(degree)
    return np.concatenate([degree, log_degree], axis=1).astype(np.float32)


def _build_degree_plus_rwpe(graph: nx.Graph, rwpe_dim: int, rwpe_steps: int) -> np.ndarray:
    degree = _build_degree_only(graph)
    rwpe = _compute_random_walk_pe(graph, rwpe_dim, rwpe_steps)
    return np.concatenate([degree, rwpe], axis=1).astype(np.float32)


def _build_degree_plus_ppr(graph: nx.Graph, ppr_dim: int) -> np.ndarray:
    degree = _build_degree_only(graph)
    ppr = _compute_personalized_pagerank_pe(graph, ppr_dim)
    return np.concatenate([degree, ppr], axis=1).astype(np.float32)


def _compute_random_walk_pe(graph: nx.Graph, rwpe_dim: int, steps: int) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return np.zeros((0, rwpe_dim), dtype=np.float32)
    
    A = nx.adjacency_matrix(graph).astype(np.float32).T
    d = np.array(A.sum(axis=0)).flatten()
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    P = A.multiply(d_inv.reshape(-1, 1))
    
    pe_list = []
    for node_idx in range(num_nodes):
        walk_vec = np.zeros(steps + 1, dtype=np.float32)
        walk_vec[0] = 1.0
        current = np.zeros(num_nodes, dtype=np.float32)
        current[node_idx] = 1.0
        for step in range(1, steps + 1):
            current = P.T.dot(current)
            walk_vec[step] = current[node_idx]
        pe_list.append(walk_vec)
    
    pe = np.array(pe_list, dtype=np.float32)
    if pe.shape[1] > rwpe_dim:
        pe = pe[:, :rwpe_dim]
    elif pe.shape[1] < rwpe_dim:
        pad = np.zeros((num_nodes, rwpe_dim - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)
    return pe


def _compute_personalized_pagerank_pe(graph: nx.Graph, ppr_dim: int) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return np.zeros((0, ppr_dim), dtype=np.float32)
    
    ppr_alpha = 0.15
    ppr_max_iter = 100
    pe_list = []
    
    for node_idx in range(num_nodes):
        personalization = {i: (1.0 if i == node_idx else 0.0) for i in range(num_nodes)}
        try:
            ppr_scores = nx.pagerank(
                graph,
                alpha=ppr_alpha,
                personalization=personalization,
                max_iter=ppr_max_iter,
                tol=1e-6,
            )
            ppr_vec = np.array([ppr_scores.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)
        except Exception:
            ppr_vec = np.ones(num_nodes, dtype=np.float32) / num_nodes
        pe_list.append(ppr_vec)
    
    pe = np.array(pe_list, dtype=np.float32)
    if pe.shape[1] > ppr_dim:
        pe = pe[:, :ppr_dim]
    elif pe.shape[1] < ppr_dim:
        pad = np.zeros((num_nodes, ppr_dim - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)
    return pe


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
