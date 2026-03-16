from __future__ import annotations

import networkx as nx
import numpy as np

try:
    import networkit as nk

    NETWORKIT_AVAILABLE = True
except ImportError:
    NETWORKIT_AVAILABLE = False

def compute_betweenness_centrality(
    graph: nx.Graph,
    bc_backend: str = "networkit",
    seed: int = 0,
) -> np.ndarray:
    backend = str(bc_backend).lower()

    if backend != "networkit":
        raise ValueError(f"Unsupported betweenness backend: {bc_backend}")
    if not NETWORKIT_AVAILABLE:
        raise RuntimeError("NetworKit is required for betweenness centrality but is not installed")

    return _compute_bc_networkit(graph=graph, seed=seed)


def _compute_bc_networkit(
    graph: nx.Graph,
    seed: int,
) -> np.ndarray:
    G = nk.nxadapter.nx2nk(graph)
    exact = nk.centrality.Betweenness(G, normalized=True)
    exact.run()
    return np.array(exact.scores(), dtype=np.float32)
