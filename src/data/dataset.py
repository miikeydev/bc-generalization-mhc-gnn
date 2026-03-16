from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .betweenness import compute_betweenness_centrality
from .features import build_node_features
from .graph_generation import generate_connected_graph
from .protocol import compute_size_bucket, normalize_data_config


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test_id: Dataset
    test_ood: Dataset
    train_label_mean: float
    train_label_std: float


class GraphListDataset(Dataset):
    def __init__(
        self,
        data_list: list[Data],
        feature_mode: str,
        lap_pe_dim: int,
        apply_train_lap_pe_sign_flip: bool,
    ) -> None:
        self.data_list = data_list
        self.feature_mode = feature_mode
        self.lap_pe_dim = lap_pe_dim
        self.apply_train_lap_pe_sign_flip = apply_train_lap_pe_sign_flip

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        data = self.data_list[idx].clone()
        if self.apply_train_lap_pe_sign_flip and self.feature_mode == "structural_only" and self.lap_pe_dim > 0:
            start = data.x.shape[1] - self.lap_pe_dim
            if start >= 0:
                signs = torch.randint(0, 2, (self.lap_pe_dim,), device=data.x.device, dtype=torch.int64)
                signs = signs.float().mul(2.0).sub(1.0)
                data.x[:, start:] = data.x[:, start:] * signs
        return data


def build_inductive_datasets(config: dict, seed: int) -> DatasetBundle:
    data_cfg = normalize_data_config(config["data"])

    train_items = _generate_split(
        families=data_cfg["train_families"],
        count=data_cfg["train_graphs"],
        num_nodes=data_cfg["train_num_nodes"],
        average_degree=data_cfg["average_degree"],
        feature_mode=data_cfg["feature_mode"],
        lap_pe_dim=data_cfg["lap_pe_dim"],
        random_feature_dim=data_cfg["random_feature_dim"],
        feature_config=data_cfg.get("feature_config", {}),
        bc_backend=data_cfg.get("bc_backend", "networkx"),
        bc_mode=data_cfg.get("bc_mode", "exact"),
        bc_approximation_k=data_cfg.get("bc_approximation_k"),
        seed=seed + 101,
    )

    val_items = _generate_split(
        families=data_cfg["train_families"],
        count=data_cfg["val_graphs"],
        num_nodes=data_cfg["train_num_nodes"],
        average_degree=data_cfg["average_degree"],
        feature_mode=data_cfg["feature_mode"],
        lap_pe_dim=data_cfg["lap_pe_dim"],
        random_feature_dim=data_cfg["random_feature_dim"],
        feature_config=data_cfg.get("feature_config", {}),
        bc_backend=data_cfg.get("bc_backend", "networkx"),
        bc_mode=data_cfg.get("bc_mode", "exact"),
        bc_approximation_k=data_cfg.get("bc_approximation_k"),
        seed=seed + 202,
    )

    test_id_items = _generate_split(
        families=data_cfg["train_families"],
        count=data_cfg["test_graphs_id"],
        num_nodes=data_cfg["test_num_nodes"],
        average_degree=data_cfg["average_degree"],
        feature_mode=data_cfg["feature_mode"],
        lap_pe_dim=data_cfg["lap_pe_dim"],
        random_feature_dim=data_cfg["random_feature_dim"],
        feature_config=data_cfg.get("feature_config", {}),
        bc_backend=data_cfg.get("bc_backend", "networkx"),
        bc_mode=data_cfg.get("bc_mode", "exact"),
        bc_approximation_k=data_cfg.get("bc_approximation_k"),
        seed=seed + 303,
    )

    ood_families = data_cfg.get("ood_families", [])
    if len(ood_families) > 0 and data_cfg["test_graphs_ood"] > 0:
        test_ood_items = _generate_split(
            families=ood_families,
            count=data_cfg["test_graphs_ood"],
            num_nodes=data_cfg["test_num_nodes"],
            average_degree=data_cfg["average_degree"],
            feature_mode=data_cfg["feature_mode"],
            lap_pe_dim=data_cfg["lap_pe_dim"],
            random_feature_dim=data_cfg["random_feature_dim"],
            feature_config=data_cfg.get("feature_config", {}),
            bc_backend=data_cfg.get("bc_backend", "networkx"),
            bc_mode=data_cfg.get("bc_mode", "exact"),
            bc_approximation_k=data_cfg.get("bc_approximation_k"),
            seed=seed + 404,
        )
    else:
        test_ood_items = []

    train_label_values = np.concatenate([item["y_log"] for item in train_items], axis=0)
    train_label_mean = float(train_label_values.mean())
    train_label_std = float(train_label_values.std() + 1e-8)

    train_data = _to_pyg_data(train_items, train_label_mean, train_label_std)
    val_data = _to_pyg_data(val_items, train_label_mean, train_label_std)
    test_id_data = _to_pyg_data(test_id_items, train_label_mean, train_label_std)
    test_ood_data = _to_pyg_data(test_ood_items, train_label_mean, train_label_std)

    return DatasetBundle(
        train=GraphListDataset(
            train_data,
            feature_mode=data_cfg["feature_mode"],
            lap_pe_dim=data_cfg["lap_pe_dim"],
            apply_train_lap_pe_sign_flip=bool(data_cfg.get("train_sign_flip_lappe", False)),
        ),
        val=GraphListDataset(
            val_data,
            feature_mode=data_cfg["feature_mode"],
            lap_pe_dim=data_cfg["lap_pe_dim"],
            apply_train_lap_pe_sign_flip=False,
        ),
        test_id=GraphListDataset(
            test_id_data,
            feature_mode=data_cfg["feature_mode"],
            lap_pe_dim=data_cfg["lap_pe_dim"],
            apply_train_lap_pe_sign_flip=False,
        ),
        test_ood=GraphListDataset(
            test_ood_data,
            feature_mode=data_cfg["feature_mode"],
            lap_pe_dim=data_cfg["lap_pe_dim"],
            apply_train_lap_pe_sign_flip=False,
        ),
        train_label_mean=train_label_mean,
        train_label_std=train_label_std,
    )


def _generate_split(
    families: Iterable[str],
    count: int,
    num_nodes: int,
    average_degree: int,
    feature_mode: str,
    lap_pe_dim: int,
    random_feature_dim: int,
    feature_config: dict,
    bc_backend: str,
    bc_mode: str,
    bc_approximation_k: int | None,
    seed: int,
) -> list[dict]:
    families_list = list(families)
    rng = np.random.default_rng(seed)
    items: list[dict] = []

    for _ in range(count):
        family = families_list[int(rng.integers(0, len(families_list)))]
        graph_seed = int(rng.integers(0, 2**31 - 1))
        graph = generate_connected_graph(
            family=family,
            num_nodes=num_nodes,
            average_degree=average_degree,
            seed=graph_seed,
        )

        raw_bc = compute_betweenness_centrality(
            graph,
            bc_backend=bc_backend,
            bc_mode=bc_mode,
            bc_approximation_k=bc_approximation_k,
            seed=graph_seed,
        )
        y_log = np.log1p(raw_bc).astype(np.float32)

        feature_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        x = build_node_features(
            graph=graph,
            mode=feature_mode,
            lap_pe_dim=lap_pe_dim,
            random_feature_dim=random_feature_dim,
            rng=feature_rng,
            feature_config=feature_config,
        )

        edge_index = _edge_index_from_graph(graph)

        num_nodes_actual = graph.number_of_nodes()
        num_edges_actual = graph.number_of_edges()
        avg_degree_actual = 2.0 * num_edges_actual / max(1, num_nodes_actual)
        density_actual = 2.0 * num_edges_actual / max(1, num_nodes_actual * (num_nodes_actual - 1))

        size_bucket = compute_size_bucket(num_nodes_actual)

        items.append(
            {
                "x": x,
                "edge_index": edge_index,
                "y_raw": raw_bc.astype(np.float32),
                "y_log": y_log,
                "family": family,
                "num_nodes": num_nodes_actual,
                "num_edges": num_edges_actual,
                "avg_degree": avg_degree_actual,
                "density": density_actual,
                "size_bucket": size_bucket,
            }
        )

    return items


def _to_pyg_data(items: list[dict], mean: float, std: float) -> list[Data]:
    data_list: list[Data] = []
    for item in items:
        y_norm = (item["y_log"] - mean) / std
        data = Data(
            x=torch.from_numpy(item["x"]).float(),
            edge_index=torch.from_numpy(item["edge_index"]).long(),
            y=torch.from_numpy(y_norm).float(),
        )
        data.y_raw = torch.from_numpy(item["y_raw"]).float()
        data.y_log = torch.from_numpy(item["y_log"]).float()
        data.family = item["family"]
        data.num_nodes_graph = item.get("num_nodes", 0)
        data.num_edges_graph = item.get("num_edges", 0)
        data.avg_degree = item.get("avg_degree", 0.0)
        data.density = item.get("density", 0.0)
        data.size_bucket = item.get("size_bucket", "unknown")
        data_list.append(data)
    return data_list


def _edge_index_from_graph(graph: nx.Graph) -> np.ndarray:
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    reverse_edges = edges[:, [1, 0]]
    undirected_edges = np.concatenate([edges, reverse_edges], axis=0)
    return undirected_edges.T
