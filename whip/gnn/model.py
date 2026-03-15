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
"""

import torch
import torch.nn as nn
import numpy as np


class BeadChainUNet(nn.Module):

    def __init__(self, state_dim=7, output_dim=4, hidden_dim=32, n_beads=16):
        super().__init__()
        self.n_beads = n_beads
        self.hidden_dim = hidden_dim
        self.n_levels = int(np.log2(n_beads))  # 4 for 16 beads

        # --- Encoder ---
        # Level 0: bead takes [self | left | right] = 3 * state_dim
        self.enc_level0 = nn.Sequential(
            nn.Linear(3 * state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Levels 1-4: node takes [left_child | right_child] = 2 * hidden_dim
        self.enc_levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(self.n_levels)
        ])

        # --- Decoder ---
        # Levels 3-1: node takes [parent_output | own_skip] = 2 * hidden_dim
        self.dec_levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(self.n_levels - 1)
        ])

        # Level 0 decoder: bead takes [parent_output | own_skip] → correction
        self.dec_level0 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, bead_states, chain_adj, tree_children):
        """
        Args:
            bead_states: (B, 16, 7) per-bead features
            chain_adj: (16, 2) per-bead chain neighbor indices, -1 for missing
            tree_children: dict level -> list of (left, right) index pairs

        Returns:
            corrections: (B, 16, 4) [dx, dy, dvx, dvy]
        """
        B = bead_states.shape[0]
        results = []
        for b in range(B):
            results.append(self._forward_single(bead_states[b], chain_adj, tree_children))
        return torch.stack(results)

    def _forward_single(self, bead_states, chain_adj, tree_children):
        device = bead_states.device
        N = self.n_beads

        # ===== ENCODE (up-pass) =====

        # Level 0: [self | left_neighbor | right_neighbor]
        self_feat = bead_states                          # (16, 7)
        left_feat = torch.zeros_like(self_feat)          # (16, 7)
        right_feat = torch.zeros_like(self_feat)         # (16, 7)

        for i in range(N):
            if chain_adj[i, 0] >= 0:
                left_feat[i] = bead_states[chain_adj[i, 0]]
            if chain_adj[i, 1] >= 0:
                right_feat[i] = bead_states[chain_adj[i, 1]]

        enc_input = torch.cat([self_feat, left_feat, right_feat], dim=-1)  # (16, 21)
        h = self.enc_level0(enc_input)  # (16, H)

        skips = [h]  # save for decoder

        # Levels 1-4: pair consecutive nodes
        for level in range(self.n_levels):
            pairs = tree_children[level]  # [(0,1), (2,3), ...]
            left_h = torch.stack([h[l] for l, r in pairs])   # (n, H)
            right_h = torch.stack([h[r] for l, r in pairs])  # (n, H)
            h = self.enc_levels[level](torch.cat([left_h, right_h], dim=-1))  # (n, H)
            skips.append(h)

        # h is now (1, H) — root

        # ===== DECODE (down-pass) =====

        # From root down to level 1
        for i, level in enumerate(range(self.n_levels - 1, 0, -1)):
            # Each parent sends its output to both its children
            parent_expanded = h.repeat_interleave(2, dim=0)  # (2n, H)
            skip = skips[level]                               # (2n, H)
            h = self.dec_levels[i](torch.cat([parent_expanded, skip], dim=-1))

        # Level 1 → Level 0 (beads)
        parent_expanded = h.repeat_interleave(2, dim=0)  # (16, H)
        skip = skips[0]                                    # (16, H)
        corrections = self.dec_level0(torch.cat([parent_expanded, skip], dim=-1))  # (16, 4)

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
