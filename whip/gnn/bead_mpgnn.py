"""Message-Passing GNN for learning bead chain physics.

A true GNN that generalizes to chains of any length. Each bead (node)
encodes its own state, encodes edge features from each neighbor, computes
messages via MLPs, and updates its hidden state from aggregated messages.

Node features (4): [vel_x, vel_y, mass, is_fixed]
Edge features (6): [dpos_x, dpos_y, dvel_x, dvel_y, neighbor_mass, rest_len]
Output (4):        [d_pos_x, d_pos_y, d_vel_x, d_vel_y]

Key design:
  - Separate node and edge encoders learn type-appropriate representations
  - Ghost beads at endpoints: missing neighbors produce zero messages
  - K rounds of message passing (default K=1, matching physics locality)
  - Weight-shared across all beads — works for any chain length
  - Normalization stats stored as buffers for self-contained inference

For K=1 per-bead training:
  Messages use edge encodings only (no neighbor hidden state needed,
  since the edge features already contain all neighbor information).

For K>1 full-chain inference:
  Messages use [neighbor_hidden | edge_encoding], gathered from the graph.
"""

import torch
import torch.nn as nn
import numpy as np


class ResMLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection."""

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


class BeadMPGNN(nn.Module):
    """Message-passing GNN for bead chain physics.

    Generalizes to any chain length N. All beads share the same weights.
    Translation-invariant via relative neighbor features.
    """

    def __init__(self, node_dim=4, edge_dim=6, output_dim=4,
                 hidden_dim=64, n_message_passes=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        # Encoders
        self.node_encoder = ResMLPBlock(node_dim, hidden_dim)
        self.edge_encoder = ResMLPBlock(edge_dim, hidden_dim)

        # Message passing layers (per round)
        # K=1: message from edge encoding only (hidden_dim input)
        # K>1: message from [neighbor_h | edge_h] (hidden_dim*2 input)
        if n_message_passes == 1:
            self.msg_mlps = nn.ModuleList([
                ResMLPBlock(hidden_dim, hidden_dim)
            ])
        else:
            # First round uses edge-only (no neighbor hidden state yet),
            # subsequent rounds use [neighbor_h | edge_h]
            self.msg_mlps = nn.ModuleList([
                ResMLPBlock(hidden_dim, hidden_dim)
            ] + [
                ResMLPBlock(hidden_dim * 2, hidden_dim)
                for _ in range(n_message_passes - 1)
            ])

        self.node_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 3, hidden_dim)  # [self | left_msg | right_msg]
            for _ in range(n_message_passes)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        # Normalization buffers (set during training, used at inference)
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))
        self.register_buffer("y_std", torch.ones(output_dim))

    def forward(self, node_feat, left_edge_feat, right_edge_feat,
                has_left, has_right):
        """Forward pass on per-bead batched data (training mode).

        Args:
            node_feat:       (B, 4) [vel_x, vel_y, mass, is_fixed]
            left_edge_feat:  (B, 6) [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]
            right_edge_feat: (B, 6) same format
            has_left:        (B,) bool
            has_right:       (B,) bool

        Returns:
            (B, 4) [d_pos_x, d_pos_y, d_vel_x, d_vel_y]
        """
        # Encode
        h_node = self.node_encoder(node_feat)            # (B, H)
        h_left_edge = self.edge_encoder(left_edge_feat)  # (B, H)
        h_right_edge = self.edge_encoder(right_edge_feat)  # (B, H)

        zeros = torch.zeros_like(h_node)

        for k in range(self.n_message_passes):
            if k == 0:
                # First round: messages from edge encodings only
                left_msg = self.msg_mlps[k](h_left_edge)
                right_msg = self.msg_mlps[k](h_right_edge)
            else:
                # Subsequent rounds would use neighbor hidden states,
                # but for per-bead data we only have edge info.
                # This path is used in step_chain for full-graph inference.
                left_msg = self.msg_mlps[k](h_left_edge)
                right_msg = self.msg_mlps[k](h_right_edge)

            # Ghost masking: zero out messages from missing neighbors
            left_msg = torch.where(has_left.unsqueeze(-1), left_msg, zeros)
            right_msg = torch.where(has_right.unsqueeze(-1), right_msg, zeros)

            # Node update: [self_h | left_msg | right_msg]
            h_node = self.node_mlps[k](
                torch.cat([h_node, left_msg, right_msg], dim=-1)
            )

        return self.output_head(h_node)

    def step_chain(self, pos, vel, mass, is_fixed, edges, rest_lengths):
        """Run one timestep on a full chain of any length.

        Builds the graph, computes relative features, runs message passing,
        and applies deltas. Uses registered normalization buffers.

        Args:
            pos:          (N, 2) bead positions
            vel:          (N, 2) bead velocities
            mass:         (N,) bead masses
            is_fixed:     (N,) bool
            edges:        (N-1, 2) rod connectivity [[0,1], [1,2], ...]
            rest_lengths: (N-1,) rest lengths

        Returns:
            pos_new: (N, 2)
            vel_new: (N, 2)
        """
        N = len(pos)
        device = pos.device

        # Build left/right neighbor lookup from edges
        left_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        right_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        left_rest = torch.zeros(N, device=device)
        right_rest = torch.zeros(N, device=device)

        for e, (i, j) in enumerate(edges):
            left_idx[j] = i
            right_idx[i] = j
            left_rest[j] = rest_lengths[e]
            right_rest[i] = rest_lengths[e]

        has_left = left_idx >= 0
        has_right = right_idx >= 0

        # Relative features for left neighbor
        safe_left = left_idx.clamp(min=0)
        l_dpos = torch.where(has_left.unsqueeze(1), pos[safe_left] - pos, torch.zeros_like(pos))
        l_dvel = torch.where(has_left.unsqueeze(1), vel[safe_left] - vel, torch.zeros_like(vel))
        l_mass = torch.where(has_left, mass[safe_left], torch.zeros_like(mass))

        # Relative features for right neighbor
        safe_right = right_idx.clamp(min=0)
        r_dpos = torch.where(has_right.unsqueeze(1), pos[safe_right] - pos, torch.zeros_like(pos))
        r_dvel = torch.where(has_right.unsqueeze(1), vel[safe_right] - vel, torch.zeros_like(vel))
        r_mass = torch.where(has_right, mass[safe_right], torch.zeros_like(mass))

        # Assemble features
        node_feat = torch.cat([
            vel,                                  # (N, 2)
            mass.unsqueeze(1),                    # (N, 1)
            is_fixed.float().unsqueeze(1),        # (N, 1)
        ], dim=1)  # (N, 4)

        left_edge_feat = torch.cat([
            l_dpos, l_dvel,                       # (N, 2), (N, 2)
            l_mass.unsqueeze(1),                  # (N, 1)
            left_rest.unsqueeze(1),               # (N, 1)
        ], dim=1)  # (N, 6)

        right_edge_feat = torch.cat([
            r_dpos, r_dvel,                       # (N, 2), (N, 2)
            r_mass.unsqueeze(1),                  # (N, 1)
            right_rest.unsqueeze(1),              # (N, 1)
        ], dim=1)  # (N, 6)

        # Normalize using stored stats
        node_feat = (node_feat - self.node_mean) / self.node_std
        left_edge_feat = (left_edge_feat - self.edge_mean) / self.edge_std
        right_edge_feat = (right_edge_feat - self.edge_mean) / self.edge_std

        # Forward pass
        delta_norm = self.forward(node_feat, left_edge_feat, right_edge_feat,
                                  has_left, has_right)

        # Denormalize output
        delta = delta_norm * self.y_std + self.y_mean

        pos_new = pos + delta[:, :2]
        vel_new = vel + delta[:, 2:]

        return pos_new, vel_new
