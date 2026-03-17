"""U-Net GNN model for learning SHAKE constraint projection.

Architecture follows a binary tree with skip connections:

ENCODE (up-pass):
  Level 0: each bead concatenates [self | left_neighbor | right_neighbor] → MLP → h
  Level 1-4: each node concatenates [left_child_h | right_child_h] → MLP → h
  Each node saves its output as a skip connection.

DECODE (down-pass):
  Level 3-1: each node concatenates [parent_output | own_skip] → MLP → d
  Level 0: each bead concatenates [parent_output | own_skip] → MLP → correction (4)

All nodes at the same level share the same MLP weights.
Each MLP block uses LayerNorm + residual connections for stable gradient flow.
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


class BeadChainUNet(nn.Module):

    def __init__(self, state_dim=7, output_dim=4, hidden_dim=64, n_beads=16):
        super().__init__()
        self.n_beads = n_beads
        self.hidden_dim = hidden_dim
        self.n_levels = int(np.log2(n_beads))  # 4 for 16 beads

        # --- Input normalization (learned) ---
        self.input_norm = nn.LayerNorm(state_dim)

        # --- Encoder ---
        # Level 0: bead takes [self | left | right] = 3 * state_dim
        self.enc_level0 = ResMLPBlock(3 * state_dim, hidden_dim)

        # Levels 1-4: node takes [left_child | right_child] = 2 * hidden_dim
        self.enc_levels = nn.ModuleList([
            ResMLPBlock(2 * hidden_dim, hidden_dim)
            for _ in range(self.n_levels)
        ])

        # --- Decoder ---
        # Levels 3-1: node takes [parent_output | own_skip] = 2 * hidden_dim
        self.dec_levels = nn.ModuleList([
            ResMLPBlock(2 * hidden_dim, hidden_dim)
            for _ in range(self.n_levels - 1)
        ])

        # Level 0 decoder: bead takes [parent_output | own_skip] → correction
        self.dec_level0 = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, bead_states, chain_adj, tree_children):
        """
        Fully batched forward pass — no per-sample loops.

        Args:
            bead_states: (B, 16, 7) per-bead features
            chain_adj: (16, 2) per-bead chain neighbor indices, -1 for missing
            tree_children: dict level -> list of (left, right) index pairs

        Returns:
            corrections: (B, 16, 4) [dx, dy, dvx, dvy]
        """
        B, N, D = bead_states.shape

        # Normalize inputs
        bead_states = self.input_norm(bead_states)

        # ===== ENCODE (up-pass) =====

        # Level 0: gather [self | left_neighbor | right_neighbor]
        # Clamp -1 indices to 0 for gathering, then zero-mask
        left_idx = chain_adj[:, 0].clamp(min=0)   # (16,)
        right_idx = chain_adj[:, 1].clamp(min=0)   # (16,)
        left_mask = (chain_adj[:, 0] >= 0).float().unsqueeze(0).unsqueeze(-1)   # (1, 16, 1)
        right_mask = (chain_adj[:, 1] >= 0).float().unsqueeze(0).unsqueeze(-1)  # (1, 16, 1)

        self_feat = bead_states                                          # (B, 16, 7)
        left_feat = bead_states[:, left_idx, :] * left_mask              # (B, 16, 7)
        right_feat = bead_states[:, right_idx, :] * right_mask           # (B, 16, 7)

        enc_input = torch.cat([self_feat, left_feat, right_feat], dim=-1)  # (B, 16, 21)
        h = self.enc_level0(enc_input)  # (B, 16, H)

        skips = [h]

        # Levels 1-4: pair children via indexing
        for level in range(self.n_levels):
            pairs = tree_children[level]  # [(0,1), (2,3), ...]
            left_indices = [l for l, r in pairs]
            right_indices = [r for l, r in pairs]
            left_h = h[:, left_indices, :]    # (B, n, H)
            right_h = h[:, right_indices, :]  # (B, n, H)
            h = self.enc_levels[level](torch.cat([left_h, right_h], dim=-1))  # (B, n, H)
            skips.append(h)

        # h is now (B, 1, H) — root

        # ===== DECODE (down-pass) =====

        # From root down to level 1
        for i, level in enumerate(range(self.n_levels - 1, 0, -1)):
            parent_expanded = h.repeat_interleave(2, dim=1)  # (B, 2n, H)
            skip = skips[level]                               # (B, 2n, H)
            h = self.dec_levels[i](torch.cat([parent_expanded, skip], dim=-1))

        # Level 1 → Level 0 (beads)
        parent_expanded = h.repeat_interleave(2, dim=1)  # (B, 16, H)
        skip = skips[0]                                    # (B, 16, H)
        corrections = self.dec_level0(torch.cat([parent_expanded, skip], dim=-1))  # (B, 16, 4)

        return corrections


def build_chain_adj(n_beads):
    """Build per-bead chain adjacency: (n_beads, 2) with -1 for missing."""
    adj = np.full((n_beads, 2), -1, dtype=np.int64)
    for i in range(n_beads):
        if i > 0:
            adj[i, 0] = i - 1
        if i < n_beads - 1:
            adj[i, 1] = i + 1
    return adj


def build_tree_children(n_beads):
    """Build tree pairing for each level: level -> [(left, right), ...]

    Indices are relative to that level's node list.
    Level 0 has n_beads nodes, level 1 has n_beads/2, etc.
    """
    children = {}
    n_nodes = n_beads
    for level in range(int(np.log2(n_beads))):
        children[level] = [(i, i + 1) for i in range(0, n_nodes, 2)]
        n_nodes = len(children[level])
    return children
