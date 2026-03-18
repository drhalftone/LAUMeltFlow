"""Per-bead message-passing GNN for learning bead chain physics.

Each bead gathers relative features from its left and right neighbors,
concatenates with its own state, and runs through a shared MLP to predict
state deltas (residual learning).

Input features (16):
  Self:  [vel_x, vel_y, mass, is_fixed]                          (4)
  Left:  [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]       (6)
  Right: [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]       (6)

Output (4): [d_pos_x, d_pos_y, d_vel_x, d_vel_y]

At inference time on a full chain:
  pos_new = pos + d_pos
  vel_new = vel + d_vel
"""

import torch
import torch.nn as nn
import numpy as np


class ResMLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection.

    Projects input to hidden_dim if needed, then applies:
        x_proj + MLP(LayerNorm(x_proj))
    """

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


class BeadGNN(nn.Module):
    """Per-bead MLP with relative features for learning physics.

    All 16 beads share the same weights. The network is translation-
    invariant because inputs use relative neighbor positions/velocities.
    """

    def __init__(self, input_dim=16, output_dim=4, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [ResMLPBlock(input_dim, hidden_dim)]
        for _ in range(n_layers - 1):
            layers.append(ResMLPBlock(hidden_dim, hidden_dim))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass on per-bead feature vectors.

        Args:
            x: (B, 16) per-bead input features
               OR (B, N_beads, 16) batched chain input

        Returns:
            (B, 4) or (B, N_beads, 4) state deltas
        """
        return self.head(self.backbone(x))

    def step_chain(self, pos, vel, mass, is_fixed, edges, rest_lengths):
        """Run one timestep on a full chain.

        Builds relative features for all beads, runs the shared MLP,
        and applies deltas to get the new state.

        Args:
            pos: (N, 2) bead positions
            vel: (N, 2) bead velocities
            mass: (N,) bead masses
            is_fixed: (N,) bool
            edges: (N-1, 2) rod connectivity [[0,1], [1,2], ...]
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
            # Edge i-j: j's left neighbor is i, i's right neighbor is j
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
        l_rest = left_rest

        # Relative features for right neighbor
        safe_right = right_idx.clamp(min=0)
        r_dpos = torch.where(has_right.unsqueeze(1), pos[safe_right] - pos, torch.zeros_like(pos))
        r_dvel = torch.where(has_right.unsqueeze(1), vel[safe_right] - vel, torch.zeros_like(vel))
        r_mass = torch.where(has_right, mass[safe_right], torch.zeros_like(mass))
        r_rest = right_rest

        # Assemble input: [vel, mass, is_fixed, l_dpos, l_dvel, l_mass, l_rest,
        #                   r_dpos, r_dvel, r_mass, r_rest]
        x = torch.cat([
            vel,                                    # (N, 2)
            mass.unsqueeze(1),                      # (N, 1)
            is_fixed.float().unsqueeze(1),          # (N, 1)
            l_dpos, l_dvel,                         # (N, 2), (N, 2)
            l_mass.unsqueeze(1),                    # (N, 1)
            l_rest.unsqueeze(1),                    # (N, 1)
            r_dpos, r_dvel,                         # (N, 2), (N, 2)
            r_mass.unsqueeze(1),                    # (N, 1)
            r_rest.unsqueeze(1),                    # (N, 1)
        ], dim=1)  # (N, 16)

        # Run MLP
        delta = self.forward(x)  # (N, 4)

        pos_new = pos + delta[:, :2]
        vel_new = vel + delta[:, 2:]

        return pos_new, vel_new
