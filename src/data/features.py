from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .protocol import normalize_feature_mode


def build_node_features(
    graph: nx.Graph,
    mode: str,
    lap_pe_dim: int = 0,
    random_feature_dim: int = 16,
    rng: np.random.Generator | None = None,
    feature_config: dict | None = None,
) -> np.ndarray:
    feature_cfg = feature_config or {}
    normalized_mode = normalize_feature_mode(mode)

    if normalized_mode == "degree_only":
        return _build_degree_only(graph)

    if normalized_mode == "degree_plus_rwpe":
        rwpe_dim = int(feature_cfg.get("rwpe_dim", 8))
        rwpe_steps = int(feature_cfg.get("rwpe_steps", 5))
        return _build_degree_plus_rwpe(graph, rwpe_dim, rwpe_steps)

    if normalized_mode == "degree_plus_ppr":
        ppr_dim = int(feature_cfg.get("ppr_dim", 8))
        ppr_alpha = float(feature_cfg.get("ppr_alpha", 0.15))
        ppr_steps = int(feature_cfg.get("ppr_steps", 8))
        return _build_degree_plus_ppr(graph, ppr_dim, ppr_alpha, ppr_steps)

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


def _build_degree_only(graph: nx.Graph) -> np.ndarray:
    degree = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32).reshape(-1, 1)
    log_degree = np.log1p(degree)
    return np.concatenate([degree, log_degree], axis=1).astype(np.float32)


def _build_degree_plus_rwpe(graph: nx.Graph, rwpe_dim: int, rwpe_steps: int) -> np.ndarray:
    degree = _build_degree_only(graph)
    rwpe = _compute_random_walk_pe(graph, rwpe_dim, rwpe_steps)
    return np.concatenate([degree, rwpe], axis=1).astype(np.float32)


def _build_degree_plus_ppr(graph: nx.Graph, ppr_dim: int, ppr_alpha: float, ppr_steps: int) -> np.ndarray:
    degree = _build_degree_only(graph)
    ppr = _compute_personalized_pagerank_pe(graph, ppr_dim, ppr_alpha, ppr_steps)
    return np.concatenate([degree, ppr], axis=1).astype(np.float32)


def _compute_random_walk_pe(graph: nx.Graph, rwpe_dim: int, steps: int) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0 or rwpe_dim <= 0:
        return np.zeros((0, rwpe_dim), dtype=np.float32)

    transition = _build_transition_matrix(graph)
    anchors = _build_anchor_matrix(graph, rwpe_dim)
    features = anchors
    num_steps = max(1, steps)
    for _ in range(num_steps):
        features = transition.T @ features
    return np.asarray(features, dtype=np.float32)


def _compute_personalized_pagerank_pe(
    graph: nx.Graph,
    ppr_dim: int,
    alpha: float,
    steps: int,
) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0 or ppr_dim <= 0:
        return np.zeros((0, ppr_dim), dtype=np.float32)

    transition = _build_transition_matrix(graph)
    anchors = _build_anchor_matrix(graph, ppr_dim)
    features = anchors.copy()
    restart = anchors.copy()
    alpha_clamped = min(max(alpha, 1e-4), 0.9999)

    for _ in range(max(1, steps)):
        features = alpha_clamped * restart + (1.0 - alpha_clamped) * (transition.T @ features)

    return np.asarray(features, dtype=np.float32)


def _build_transition_matrix(graph: nx.Graph) -> sparse.csr_matrix:
    adjacency = nx.to_scipy_sparse_array(graph, format="csr", dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sparse.diags(inv_degree, format="csr") @ adjacency


def _build_anchor_matrix(graph: nx.Graph, num_anchors: int) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    anchors = np.zeros((num_nodes, num_anchors), dtype=np.float32)
    if num_nodes == 0 or num_anchors <= 0:
        return anchors

    degrees = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32)
    ordered_nodes = np.argsort(-degrees, kind="stable")
    selected = ordered_nodes[: min(num_nodes, num_anchors)]
    for column, node_idx in enumerate(selected):
        anchors[int(node_idx), column] = 1.0
    return anchors


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
