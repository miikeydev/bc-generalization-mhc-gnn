from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch_geometric.nn import APPNP as APPNPConv
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv, JumpingKnowledge, SAGEConv


def sinkhorn_knopp(
    raw_matrix: torch.Tensor,
    tau: float,
    num_iters: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    tau_safe = max(float(tau), eps)
    matrix = torch.exp((raw_matrix / tau_safe).clamp(min=-20.0, max=20.0)).clamp_min(eps)
    for _ in range(int(num_iters)):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def build_permutation_bank(
    n_streams: int,
    max_permutations: int | None,
    seed: int,
) -> torch.Tensor:
    if max_permutations is None and n_streams > 6:
        raise ValueError(
            "mhc_lite with n_streams > 6 requires mhc_lite_max_permutations to avoid factorial blow-up"
        )

    permutations = list(itertools.permutations(range(n_streams)))

    if max_permutations is not None and max_permutations < len(permutations):
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(permutations), generator=generator)[:max_permutations]
        permutations = [permutations[int(idx)] for idx in indices]

    mats = []
    for perm in permutations:
        mat = torch.zeros(n_streams, n_streams, dtype=torch.float32)
        for row_idx, col_idx in enumerate(perm):
            mat[row_idx, col_idx] = 1.0
        mats.append(mat)

    return torch.stack(mats, dim=0)


class HyperConnectionMappings(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_streams: int,
        variant: str,
        use_dynamic: bool,
        use_static: bool,
        init_alpha: float,
        sinkhorn_tau: float,
        sinkhorn_iters: int,
        mhc_lite_max_permutations: int | None,
        mhc_lite_permutation_seed: int,
        rms_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.variant = variant
        self.use_dynamic = use_dynamic
        self.use_static = use_static
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters
        self.rms_eps = rms_eps

        if not use_dynamic and not use_static:
            raise ValueError("At least one of use_dynamic/use_static must be True")

        if use_dynamic:
            self.pre_dynamic = nn.Linear(hidden_dim, 1, bias=False)
            self.post_dynamic = nn.Linear(hidden_dim, 1, bias=False)

        if use_static:
            self.pre_static = nn.Parameter(torch.full((1, n_streams), 1.0 / n_streams))
            self.post_static = nn.Parameter(torch.full((1, n_streams), 1.0 / n_streams))

        self.alpha_pre = nn.Parameter(torch.tensor(float(init_alpha)))
        self.alpha_post = nn.Parameter(torch.tensor(float(init_alpha)))

        if variant in {"hc", "mhc"}:
            if use_dynamic:
                self.res_dynamic = nn.Linear(hidden_dim, n_streams * n_streams, bias=False)
            if use_static:
                static_res = torch.eye(n_streams, dtype=torch.float32)
                self.res_static = nn.Parameter(static_res)
            self.alpha_res = nn.Parameter(torch.tensor(float(init_alpha)))
        elif variant == "mhc_lite":
            perm_bank = build_permutation_bank(
                n_streams=n_streams,
                max_permutations=mhc_lite_max_permutations,
                seed=mhc_lite_permutation_seed,
            )
            self.register_buffer("permutation_bank", perm_bank)
            num_perms = int(perm_bank.shape[0])

            if use_dynamic:
                self.perm_dynamic = nn.Linear(hidden_dim, num_perms, bias=False)
            if use_static:
                self.perm_static = nn.Parameter(torch.zeros(num_perms))

            self.alpha_perm = nn.Parameter(torch.tensor(float(init_alpha)))
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self._rms_norm(x_streams)
        pooled = x_norm.mean(dim=1)

        pre_logits = torch.zeros(
            pooled.shape[0],
            1,
            self.n_streams,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        if self.use_dynamic:
            pre_logits = pre_logits + self.alpha_pre * self.pre_dynamic(pooled).unsqueeze(-1)
        if self.use_static:
            pre_logits = pre_logits + self.pre_static.unsqueeze(0)
        h_pre = torch.sigmoid(pre_logits)

        post_logits = torch.zeros(
            pooled.shape[0],
            1,
            self.n_streams,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        if self.use_dynamic:
            post_logits = post_logits + self.alpha_post * self.post_dynamic(pooled).unsqueeze(-1)
        if self.use_static:
            post_logits = post_logits + self.post_static.unsqueeze(0)
        h_post = 2.0 * torch.sigmoid(post_logits)

        if self.variant == "mhc_lite":
            logits = torch.zeros(
                pooled.shape[0],
                self.permutation_bank.shape[0],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            if self.use_dynamic:
                logits = logits + self.alpha_perm * self.perm_dynamic(pooled)
            if self.use_static:
                logits = logits + self.perm_static.unsqueeze(0)
            weights = torch.softmax(logits, dim=-1)
            h_res = torch.einsum("np,pij->nij", weights, self.permutation_bank.to(pooled.dtype))
            return h_pre, h_post, h_res

        raw_res = torch.zeros(
            pooled.shape[0],
            self.n_streams,
            self.n_streams,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        if self.use_dynamic:
            dynamic = self.res_dynamic(pooled).view(pooled.shape[0], self.n_streams, self.n_streams)
            raw_res = raw_res + self.alpha_res * dynamic
        if self.use_static:
            raw_res = raw_res + self.res_static.unsqueeze(0)

        if self.variant == "mhc":
            h_res = sinkhorn_knopp(
                raw_matrix=raw_res,
                tau=self.sinkhorn_tau,
                num_iters=self.sinkhorn_iters,
            )
        else:
            h_res = raw_res

        return h_pre, h_post, h_res

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.rms_eps)
        return x / rms


class HyperConnectionGNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        n_streams: int,
        variant: str,
        gnn_type: str,
        use_dynamic: bool,
        use_static: bool,
        init_alpha: float,
        sinkhorn_tau: float,
        sinkhorn_iters: int,
        mhc_lite_max_permutations: int | None,
        mhc_lite_permutation_seed: int,
        gcnii_alpha: float,
        gcnii_theta: float,
        appnp_alpha: float,
        appnp_k: int,
        jk_mode: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.gnn_type = gnn_type.lower()
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.dropout = nn.Dropout(dropout)

        self.input_expand = nn.Linear(input_dim, n_streams * hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.hyper_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            self.gnn_layers.append(
                build_conv_layer(
                    gnn_type=self.gnn_type,
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    layer_index=layer_idx,
                    gcnii_alpha=gcnii_alpha,
                    gcnii_theta=gcnii_theta,
                    appnp_alpha=appnp_alpha,
                    appnp_k=appnp_k,
                )
            )
            self.hyper_layers.append(
                HyperConnectionMappings(
                    hidden_dim=hidden_dim,
                    n_streams=n_streams,
                    variant=variant,
                    use_dynamic=use_dynamic,
                    use_static=use_static,
                    init_alpha=init_alpha,
                    sinkhorn_tau=sinkhorn_tau,
                    sinkhorn_iters=sinkhorn_iters,
                    mhc_lite_max_permutations=mhc_lite_max_permutations,
                    mhc_lite_permutation_seed=mhc_lite_permutation_seed,
                )
            )

        self.jk = None
        readout_dim = hidden_dim
        if self.gnn_type == "jknet":
            if jk_mode == "lstm":
                self.jk = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
            else:
                self.jk = JumpingKnowledge(mode=jk_mode)
            readout_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim

        self.readout = nn.Linear(readout_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        streams = self.input_expand(x).view(num_nodes, self.n_streams, self.hidden_dim)
        gcnii_x0 = streams.mean(dim=1)
        representations: list[torch.Tensor] = []

        for gnn, hyper in zip(self.gnn_layers, self.hyper_layers):
            h_pre, h_post, h_res = hyper(streams)
            aggregated = torch.bmm(h_pre, streams).squeeze(1)
            if isinstance(gnn, GCNIIMessageLayer):
                message = gnn(aggregated, edge_index, gcnii_x0)
            else:
                message = gnn(aggregated, edge_index)
            message = torch.relu(message)
            message = self.dropout(message)
            if self.jk is not None:
                representations.append(message)

            residual = torch.bmm(h_res, streams)
            expanded = h_post.transpose(1, 2) * message.unsqueeze(1)
            streams = residual + expanded

        if self.jk is not None:
            out = self.jk(representations)
        else:
            out = streams.mean(dim=1)
        return self.readout(out).squeeze(-1)


class GCNIIMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layer_index: int,
        alpha: float,
        theta: float,
    ) -> None:
        super().__init__()
        self.conv = GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=layer_index + 1, shared_weights=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.conv(x, x0, edge_index)


class APPNPMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        alpha: float,
        k: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.prop = APPNPConv(K=k, alpha=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        return self.prop(hidden, edge_index)


def build_conv_layer(
    gnn_type: str,
    in_dim: int,
    out_dim: int,
    layer_index: int,
    gcnii_alpha: float,
    gcnii_theta: float,
    appnp_alpha: float,
    appnp_k: int,
):
    key = gnn_type.lower()
    if key == "gcn":
        return GCNConv(in_dim, out_dim)
    if key == "sage":
        return SAGEConv(in_dim, out_dim)
    if key == "gat":
        return GATConv(in_dim, out_dim, heads=1, concat=False)
    if key == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINConv(mlp)
    if key == "gcnii":
        return GCNIIMessageLayer(
            hidden_dim=in_dim,
            layer_index=layer_index,
            alpha=gcnii_alpha,
            theta=gcnii_theta,
        )
    if key == "appnp":
        return APPNPMessageLayer(
            hidden_dim=in_dim,
            alpha=appnp_alpha,
            k=appnp_k,
        )
    if key == "jknet":
        return GCNConv(in_dim, out_dim)
    raise ValueError(f"Unsupported gnn_type: {gnn_type}")


def factorial_bound_for_streams(n_streams: int) -> int:
    return math.factorial(n_streams)
