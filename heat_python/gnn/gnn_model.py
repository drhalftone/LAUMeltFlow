"""Message-passing GNN for the heat-shield state surrogate.

Adapted from whip/gnn/bead_mpgnn.py (NOT the Sod flux model): same
NodeEncoder/EdgeEncoder/MessageMLP/NodeUpdate core and per-NODE delta output,
because the heat shield is a 1D chain whose surrogate predicts the next
per-cell state. The bead's relative-neighbor edge features map onto the
heat-shield's gradient-driven conduction.

Node features (absolute per-cell state):  [T, rho, rho_i..., porosity]
Edge features (relative neighbor + geom):  [dT, drho, drho_i..., dporosity, dx]
Output (per-cell delta of independent DOF): [dT, d_rho_i...]  (rho, porosity
    derived from rho_i; the inert species is ~constant)

K=1 message passing matches the solver's nearest-neighbor conduction stencil;
K>1 widens the effective stencil (set n_message_passes=2 to try).
"""

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection (same as bead/Sod)."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = (nn.Linear(in_dim, hidden_dim)
                     if in_dim != hidden_dim else nn.Identity())
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x + self.mlp(self.norm(x))


class HeatMPGNN(nn.Module):
    """Per-cell state-delta predictor for the 1D heat shield. Weight-shared
    across cells, so it generalizes to any mesh length."""

    def __init__(self, node_dim, edge_dim, output_dim,
                 hidden_dim=64, n_message_passes=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        self.node_encoder = ResMLPBlock(node_dim, hidden_dim)
        self.edge_encoder = ResMLPBlock(edge_dim, hidden_dim)

        self.msg_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim, hidden_dim)
            for _ in range(n_message_passes)
        ])
        self.node_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 3, hidden_dim)   # [self | left | right]
            for _ in range(n_message_passes)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        # Normalization buffers (set from data, used at inference).
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))
        self.register_buffer("y_std", torch.ones(output_dim))

    def forward(self, node_feat, left_edge, right_edge, has_left, has_right):
        """Per-cell forward (training mode). Inputs are normalized.

        node_feat (B, node_dim); left_edge/right_edge (B, edge_dim);
        has_left/has_right (B,) bool. Returns (B, output_dim) normalized delta.
        """
        h_node = self.node_encoder(node_feat)
        h_left = self.edge_encoder(left_edge)
        h_right = self.edge_encoder(right_edge)
        zeros = torch.zeros_like(h_node)

        for k in range(self.n_message_passes):
            left_msg = self.msg_mlps[k](h_left)
            right_msg = self.msg_mlps[k](h_right)
            left_msg = torch.where(has_left.unsqueeze(-1), left_msg, zeros)
            right_msg = torch.where(has_right.unsqueeze(-1), right_msg, zeros)
            h_node = self.node_mlps[k](
                torch.cat([h_node, left_msg, right_msg], dim=-1))

        return self.output_head(h_node)

    @torch.no_grad()
    def step_mesh(self, node_state, dx, out_idx):
        """One surrogate step over a full mesh of any length (rollout mode).

        node_state (m, node_dim) absolute per-cell state INCLUDING the two
            ghost cells (indices 0 and m-1) that carry the boundary forcing.
        dx (m,) cell widths. out_idx: indices of node features that the model
            predicts a delta for (maps output_dim -> node_dim columns).

        Returns the updated interior state (n, node_dim); ghosts are left for
        the caller to re-impose from the BC. Only interior cells (1..m-2) are
        advanced.
        """
        device = node_state.device
        m = node_state.shape[0]
        interior = torch.arange(1, m - 1, device=device)

        self_f = node_state[interior]
        left_f = node_state[interior - 1]
        right_f = node_state[interior + 1]
        dx_int = dx[interior]

        left_edge = torch.cat([left_f - self_f, dx_int.unsqueeze(-1)], dim=-1)
        right_edge = torch.cat([right_f - self_f, dx_int.unsqueeze(-1)], dim=-1)
        has = torch.ones(interior.shape[0], dtype=torch.bool, device=device)

        nf = (self_f - self.node_mean) / self.node_std
        le = (left_edge - self.edge_mean) / self.edge_std
        re = (right_edge - self.edge_mean) / self.edge_std

        delta = self.forward(nf, le, re, has, has) * self.y_std + self.y_mean

        new_interior = self_f.clone()
        new_interior[:, out_idx] = self_f[:, out_idx] + delta
        return new_interior
