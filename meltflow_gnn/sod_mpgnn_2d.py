"""Message-passing GNN for learning 2D Roe flux on the Sod tube.

Port of the bead/Sod-1D MPGNN to 2D. Each training sample is a 5-cell
stencil: center C with neighbors W, E, S, N. The model predicts the 4
face fluxes around C: F_w, F_e, G_s, G_n.

Architecture:
  Node features (4): [rho, u, v, p]   for each of 5 cells
  Edge features (3): [dx, normal_x, normal_y]   for each of 4 edges
  Output        (4): [F_rho, F_rhou, F_rhov, F_E]   for each of 4 edges

Message passing (K=1, sum aggregation):
  1. Encode 5 nodes -> h_n (shape (B, 5, H))
  2. Encode 4 edges -> h_e (shape (B, 4, H))
  3. For each edge (C, neighbor): msg = MsgMLP([h_neighbor | h_e])
  4. Center aggregates: m_C = sum of 4 incoming messages
  5. Each neighbor aggregates: m_N = msg from C (degenerate: single edge)
  6. Node update: h_n_new = NodeMLP([h_n | m_n])
  7. Per-edge flux head: flux = FluxHead([h_C_new | h_neighbor_new | h_e])

Cell ordering convention (matches grid_sampler_2d output):
    Index:   0   1   2   3   4
    Cell:    W   C   E   S   N

Edge ordering convention:
    Index:   0       1       2       3
    Edge:    C-W     C-E     C-S     C-N
    Normal: (-1,0)  (+1,0)  (0,-1)  (0,+1)

Flux ordering (output):
    F_w (interface C-W), F_e (interface C-E),
    G_s (interface C-S), G_n (interface C-N)
"""

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x + self.mlp(self.norm(x))


# Index conventions for the 5-cell stencil
IDX_W, IDX_C, IDX_E, IDX_S, IDX_N = 0, 1, 2, 3, 4
NEIGHBOR_IDX = [IDX_W, IDX_E, IDX_S, IDX_N]  # in edge order


class SodMPGNN_2D(nn.Module):
    """Message-passing GNN over a 5-cell 2D stencil predicting 4 face fluxes."""

    def __init__(self, node_dim=4, edge_dim=3, output_dim=4,
                 hidden_dim=32, n_message_passes=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        self.node_encoder = ResMLPBlock(node_dim, hidden_dim)
        self.edge_encoder = ResMLPBlock(edge_dim, hidden_dim)

        # Message MLP per round: [h_neighbor | h_edge] -> message
        self.msg_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 2, hidden_dim)
            for _ in range(n_message_passes)
        ])
        # Node update per round: [h_self | aggregated_messages] -> hidden
        self.node_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 2, hidden_dim)
            for _ in range(n_message_passes)
        ])

        # Per-edge flux head: [h_C | h_neighbor | h_edge] -> flux
        self.flux_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Normalization buffers
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))
        self.register_buffer("y_std", torch.ones(output_dim))

    def forward_stencil(self, nodes, edges):
        """Predict 4 fluxes from a 5-cell stencil.

        Args:
            nodes: (B, 5, node_dim) — five cells (W, C, E, S, N)
            edges: (B, 4, edge_dim) — four edges (C-W, C-E, C-S, C-N)

        Returns:
            (B, 4, output_dim) — fluxes at the four faces, normalized space.
        """
        B = nodes.shape[0]
        H = self.hidden_dim
        device = nodes.device

        h_n = self.node_encoder(nodes.reshape(B * 5, -1)).reshape(B, 5, H)
        h_e = self.edge_encoder(edges.reshape(B * 4, -1)).reshape(B, 4, H)

        for k in range(self.n_message_passes):
            # Messages flow on each of the 4 edges, both directions.
            # We need:
            #   - For C: aggregate over messages from {W, E, S, N}
            #   - For each neighbor n: message from C (single source)

            # Build [h_neighbor | h_edge] for each of 4 edges -> messages into C
            neighbor_hs = h_n[:, NEIGHBOR_IDX, :]  # (B, 4, H)
            msg_into_C_inp = torch.cat([neighbor_hs, h_e], dim=-1)  # (B, 4, 2H)
            msg_into_C = self.msg_mlps[k](
                msg_into_C_inp.reshape(B * 4, -1)
            ).reshape(B, 4, H)
            agg_C = msg_into_C.sum(dim=1)  # (B, H)

            # Build [h_C | h_edge] for each of 4 edges -> message into each neighbor
            h_C_rep = h_n[:, IDX_C:IDX_C + 1, :].expand(-1, 4, -1)  # (B, 4, H)
            msg_into_N_inp = torch.cat([h_C_rep, h_e], dim=-1)  # (B, 4, 2H)
            msg_into_N = self.msg_mlps[k](
                msg_into_N_inp.reshape(B * 4, -1)
            ).reshape(B, 4, H)  # (B, 4, H)  one per neighbor

            # Node update
            #  Center: [h_C | agg_C]
            h_C_new = self.node_mlps[k](
                torch.cat([h_n[:, IDX_C, :], agg_C], dim=-1)
            )
            #  Neighbors: [h_neighbor | msg_into_N]
            neighbor_update_inp = torch.cat([neighbor_hs, msg_into_N], dim=-1)  # (B, 4, 2H)
            h_neighbors_new = self.node_mlps[k](
                neighbor_update_inp.reshape(B * 4, -1)
            ).reshape(B, 4, H)

            # Write back
            h_n_new = torch.zeros_like(h_n)
            h_n_new[:, IDX_C, :] = h_C_new
            for slot, idx in enumerate(NEIGHBOR_IDX):
                h_n_new[:, idx, :] = h_neighbors_new[:, slot, :]
            h_n = h_n_new

        # Per-edge flux head
        h_C_final = h_n[:, IDX_C:IDX_C + 1, :].expand(-1, 4, -1)  # (B, 4, H)
        h_neighbors_final = h_n[:, NEIGHBOR_IDX, :]  # (B, 4, H)
        flux_inp = torch.cat([h_C_final, h_neighbors_final, h_e], dim=-1)  # (B, 4, 3H)
        flux_norm = self.flux_head(flux_inp.reshape(B * 4, -1)).reshape(B, 4, -1)
        return flux_norm  # (B, 4, output_dim)
