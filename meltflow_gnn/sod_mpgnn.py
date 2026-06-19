"""Message-passing GNN for learning Roe flux on a 1D shock tube.

A port of whip/gnn/bead_mpgnn.py adapted for the Sod problem. Same
NodeEncoder + EdgeEncoder + MessageMLP + NodeUpdate structure, but with
the output head attached to *edges* instead of nodes — the Sod problem
predicts numerical flux at cell interfaces, not state deltas at cells.

Training data: pairs (L, R) of cell states with one interface between
them. At training time each cell has only one neighbor (the other cell
in the pair); the ghost-masking mechanism handles this naturally.

At inference time the model is applied across an N-cell mesh: each
interior cell has both neighbors, each boundary cell has one. The same
weights handle both regimes because of the message-aggregation symmetry.

Node features (3): [rho, u, p]
Edge features (1): [dx]
Output       (3): [F_rho, F_rhou, F_E]  per interface

K = 1 message passing — matches Roe's local stencil.
"""

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection. Same as bead_mpgnn."""

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


class SodMPGNN(nn.Module):
    """Message-passing GNN that predicts Roe flux at cell interfaces.

    Architecture mirrors BeadMPGNN: separate node and edge encoders,
    K rounds of message passing, then an output head — but the head
    produces per-edge flux rather than per-node deltas.
    """

    def __init__(self, node_dim=3, edge_dim=1, output_dim=3,
                 hidden_dim=32, n_message_passes=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        self.node_encoder = ResMLPBlock(node_dim, hidden_dim)
        self.edge_encoder = ResMLPBlock(edge_dim, hidden_dim)

        # Message MLPs (per round)
        self.msg_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim, hidden_dim)
            for _ in range(n_message_passes)
        ])

        # Node update MLPs (per round): [self | left_msg | right_msg] -> hidden
        self.node_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 3, hidden_dim)
            for _ in range(n_message_passes)
        ])

        # Edge output head: [h_L | h_R | edge_h] -> flux
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

    def forward_pair(self, node_L, node_R, edge_feat):
        """Predict flux at the interface between two cells (training mode).

        Args:
            node_L:    (B, 3) [rho, u, p] left cell
            node_R:    (B, 3) [rho, u, p] right cell
            edge_feat: (B, 1) [dx] at the interface

        Returns:
            (B, 3) flux [F_rho, F_rhou, F_E] — normalized space.
        """
        h_L = self.node_encoder(node_L)
        h_R = self.node_encoder(node_R)
        h_e = self.edge_encoder(edge_feat)

        # Message-passing rounds. For 2-cell graphs:
        #   L has only one neighbor (R), R has only one neighbor (L).
        # The "left" and "right" slots in the bead model become "no neighbor"
        # and "the other cell" depending on which side we're at.
        zeros = torch.zeros_like(h_L)

        for k in range(self.n_message_passes):
            # Message from R to L (via the interface edge)
            msg_RtoL = self.msg_mlps[k](h_e)
            # Message from L to R (via the same interface edge)
            msg_LtoR = self.msg_mlps[k](h_e)

            # L's update: self + (no-left) + (right=R-message)
            h_L_new = self.node_mlps[k](torch.cat([h_L, zeros, msg_RtoL], dim=-1))
            # R's update: self + (left=L-message) + (no-right)
            h_R_new = self.node_mlps[k](torch.cat([h_R, msg_LtoR, zeros], dim=-1))

            h_L = h_L_new
            h_R = h_R_new

        # Edge output head
        flux_norm = self.flux_head(torch.cat([h_L, h_R, h_e], dim=-1))
        return flux_norm

    def predict_flux_at_interfaces(self, state, dx):
        """Predict flux at every interior interface of an N-cell mesh (inference).

        For real Sod rollout: cells have neighbors on both sides (except
        at the boundary). For now we predict only interior interfaces and
        leave boundary fluxes to be handled by ghost cells.

        Args:
            state: (N, 3) primitive variables [rho, u, p] for N cells
            dx: scalar cell width (or (N-1,) per-interface)

        Returns:
            flux: (N-1, 3) flux at each interface i+1/2 between cell i and i+1.
        """
        device = state.device
        N = state.shape[0]
        # Normalize
        node_norm = (state - self.node_mean) / self.node_std

        if torch.is_tensor(dx):
            edge_raw = dx.view(-1, 1) if dx.dim() == 1 else dx.unsqueeze(-1)
            if edge_raw.shape[0] == 1:
                edge_raw = edge_raw.expand(N - 1, 1)
        else:
            edge_raw = torch.full((N - 1, 1), float(dx), device=device)
        edge_norm = (edge_raw - self.edge_mean) / self.edge_std

        # Pair-wise prediction: for each interface i, use cells (i, i+1)
        node_L = node_norm[:-1]  # (N-1, 3)
        node_R = node_norm[1:]   # (N-1, 3)

        flux_norm = self.forward_pair(node_L, node_R, edge_norm)
        flux = flux_norm * self.y_std + self.y_mean
        return flux
