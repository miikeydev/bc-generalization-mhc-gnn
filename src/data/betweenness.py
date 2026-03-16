from __future__ import annotations

import networkx as nx
import numpy as np

try:
    import networkit as nk

    NETWORKIT_AVAILABLE = True
except ImportError:
    NETWORKIT_AVAILABLE = False


AUTO_NETWORKIT_THRESHOLD = 10_000


def compute_betweenness_centrality(
    graph: nx.Graph,
    bc_backend: str = "networkx",
    bc_mode: str = "exact",
    bc_approximation_k: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    backend = str(bc_backend).lower()
    mode = str(bc_mode).lower()

    if backend == "auto":
        backend = "networkit" if NETWORKIT_AVAILABLE and graph.number_of_nodes() > AUTO_NETWORKIT_THRESHOLD else "networkx"

    if backend == "networkit" and NETWORKIT_AVAILABLE:
        try:
            return _compute_bc_networkit(
                graph=graph,
                mode=mode,
                bc_approximation_k=bc_approximation_k,
                seed=seed,
            )
        except Exception:
            pass

    return _compute_bc_networkx(
        graph=graph,
        mode=mode,
        bc_approximation_k=bc_approximation_k,
        seed=seed,
    )


def _compute_bc_networkx(
    graph: nx.Graph,
    mode: str,
    bc_approximation_k: int | None,
    seed: int,
) -> np.ndarray:
    if mode == "approx" and bc_approximation_k is not None:
        bc = nx.betweenness_centrality(graph, normalized=True, k=bc_approximation_k, seed=seed)
    else:
        bc = nx.betweenness_centrality(graph, normalized=True)
    return np.array([bc[node] for node in graph.nodes()], dtype=np.float32)


def _compute_bc_networkit(
    graph: nx.Graph,
    mode: str,
    bc_approximation_k: int | None,
    seed: int,
) -> np.ndarray:
    G = nk.nxadapter.nx2nk(graph)
    if mode == "approx":
        sample_count = _resolve_networkit_sample_count(graph.number_of_nodes(), bc_approximation_k)
        estimator = _build_networkit_estimator(G, sample_count, seed)
        estimator.run()
        return np.array(estimator.scores(), dtype=np.float32)

    exact = nk.centrality.Betweenness(G, normalized=True)
    exact.run()
    return np.array(exact.scores(), dtype=np.float32)


def _build_networkit_estimator(graph, sample_count: int, seed: int):
    if hasattr(nk.centrality, "EstimateBetweenness"):
        try:
            return nk.centrality.EstimateBetweenness(
                graph,
                sample_count,
                normalized=True,
                parallel=True,
                seed=seed,
            )
        except TypeError:
            return nk.centrality.EstimateBetweenness(graph, sample_count, True, True)
    raise ValueError("NetworKit approximation backend is unavailable in this installation")


def _resolve_networkit_sample_count(num_nodes: int, bc_approximation_k: int | None) -> int:
    if bc_approximation_k is not None:
        return max(1, min(int(bc_approximation_k), num_nodes))
    return max(32, min(num_nodes, int(np.sqrt(max(1, num_nodes)) * 8)))
