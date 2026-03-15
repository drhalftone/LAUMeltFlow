"""U-Net GNN model for learning SHAKE constraint projection.

Binary tree message passing:
  - Chain edges (level 0): bead-to-bead local forces
  - Up-pass (levels 0->4): children send to parents
  - Down-pass (levels 4->0): parents send corrections to children
  - Output: SHAKE corrections for bead nodes only
"""

import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    """Compute edge messages from concatenated [sender, receiver] features."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_src, x_dst):
        # x_src, x_dst: (E, H)
        return self.net(torch.cat([x_src, x_dst], dim=-1))


class MessagePassingLayer(nn.Module):
    """One round of message passing on a set of edges.

    For each edge (src -> dst), computes a message from [src_feat, dst_feat]
    and aggregates (sums) messages at each destination node.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = EdgeMLP(hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, n_nodes):
        """
        Args:
            x: (N, H) node features
            edge_index: (2, E) edges [src, dst]
            n_nodes: total number of nodes
        Returns:
            x_updated: (N, H) updated node features
        """
        src, dst = edge_index  # (E,), (E,)
        x_src = x[src]  # (E, H)
        x_dst = x[dst]  # (E, H)

        # Compute messages
        messages = self.edge_mlp(x_src, x_dst)  # (E, H)

        # Aggregate at destinations (sum)
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        agg.index_add_(0, dst, messages)

        # Update: combine old features with aggregated messages
        x_updated = x + self.update_mlp(torch.cat([x, agg], dim=-1))

        return x_updated


class BeadChainUNet(nn.Module):
    """U-Net GNN for bead chain SHAKE correction.

    Architecture:
        1. Embed node features (7) -> hidden_dim
        2. Chain message passing (bead-to-bead)
        3. Up-pass: 4 levels of child->parent messages
        4. Down-pass: 4 levels of parent->child messages (with skip)
        5. Decode bead features -> corrections (4)
    """

    def __init__(self, input_dim=7, output_dim=4, hidden_dim=64, n_beads=16):
        super().__init__()
        self.n_beads = n_beads
        self.hidden_dim = hidden_dim

        # Node embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Chain-level message passing (level 0)
        self.chain_mp = MessagePassingLayer(hidden_dim)

        # Up-pass: one layer per tree level
        self.up_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(4)
        ])

        # Down-pass: one layer per tree level
        # Input is hidden_dim * 2 (current + skip from up-pass)
        self.down_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(4)
        ])

        # Skip connection fusion (combine up-pass state with down-pass state)
        self.skip_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            ) for _ in range(4)
        ])

        # Output decoder (bead nodes only)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, chain_edges, tree_edge_index, node_levels, n_total):
        """
        Args:
            x: (B, 31, 7) node features
            chain_edges: (2, 30) bead-to-bead edges
            tree_edge_index: (2, 60) parent-child edges (bidirectional)
            node_levels: (31,) level per node
            n_total: int, total nodes (31)
        Returns:
            corrections: (B, 16, 4) predicted SHAKE corrections
        """
        B = x.shape[0]
        device = x.device
        corrections = []

        for b in range(B):
            h = self._forward_single(
                x[b], chain_edges, tree_edge_index, node_levels, n_total
            )
            corrections.append(h)

        return torch.stack(corrections)

    def _forward_single(self, x, chain_edges, tree_edge_index, node_levels, n_total):
        """Forward pass for a single sample."""

        # 1. Embed all nodes
        h = self.encoder(x)  # (31, H)

        # 2. Chain message passing (bead-to-bead edges only)
        h = self.chain_mp(h, chain_edges, n_total)

        # 3. Up-pass: children -> parents, level by level
        #    Build per-level edge sets (child -> parent only)
        up_edges_per_level = []
        down_edges_per_level = []
        for level in range(1, 5):
            # Find parent nodes at this level
            parent_mask = (node_levels == level)
            parent_ids = torch.where(parent_mask)[0]

            # Extract child->parent edges from tree_edge_index
            src, dst = tree_edge_index
            # child->parent: src is at level-1, dst is at level
            mask = (node_levels[src] == level - 1) & (node_levels[dst] == level)
            up_edges = torch.stack([src[mask], dst[mask]])
            up_edges_per_level.append(up_edges)

            # parent->child: src is at level, dst is at level-1
            mask = (node_levels[src] == level) & (node_levels[dst] == level - 1)
            down_edges = torch.stack([src[mask], dst[mask]])
            down_edges_per_level.append(down_edges)

        skip_states = []
        for i, up_edges in enumerate(up_edges_per_level):
            skip_states.append(h.clone())  # save for skip connection
            h = self.up_layers[i](h, up_edges, n_total)

        # 4. Down-pass: parents -> children, level by level (reverse order)
        for i in range(3, -1, -1):
            down_edges = down_edges_per_level[i]
            h = self.down_layers[i](h, down_edges, n_total)
            # Fuse with skip connection from up-pass
            h_skip = skip_states[i]
            h = self.skip_fuse[i](torch.cat([h, h_skip], dim=-1))

        # 5. Decode bead nodes only
        h_beads = h[:self.n_beads]  # (16, H)
        return self.decoder(h_beads)  # (16, 4)
