"""Dataset for paired input/target .npy files (pix2pix-style)."""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


class BeadChainDataset(Dataset):
    """Load paired input/target .npy files matched by numeric suffix."""

    def __init__(self, data_dir="data"):
        input_dir = os.path.join(data_dir, "input")
        target_dir = os.path.join(data_dir, "target")

        # Find paired files by numeric suffix
        input_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
        target_files = sorted(f for f in os.listdir(target_dir) if f.endswith(".npy"))

        input_ids = {self._extract_id(f): f for f in input_files}
        target_ids = {self._extract_id(f): f for f in target_files}
        common = sorted(set(input_ids) & set(target_ids))

        self.pairs = [
            (os.path.join(input_dir, input_ids[k]),
             os.path.join(target_dir, target_ids[k]))
            for k in common
        ]

        # Load graph structure (shared across all frames)
        graph = np.load(os.path.join(data_dir, "graph.npz"))
        self.chain_edges = torch.from_numpy(graph["chain_edges"]).long()   # (2, 30)
        self.tree_edges = torch.from_numpy(graph["tree_edges"]).long()     # (60, 2)
        self.rest_lengths = torch.from_numpy(graph["rest_lengths"]).float()
        self.node_levels = torch.from_numpy(graph["node_levels"]).long()
        self.n_beads = int(graph["n_beads"])
        self.n_total = int(graph["n_total_nodes"])
        self.n_levels = int(graph["n_levels"])

        # Build tree edges as (2, E) format, grouped by level
        tree_src = self.tree_edges[:, 0]
        tree_dst = self.tree_edges[:, 1]
        self.tree_edge_index = torch.stack([tree_src, tree_dst])  # (2, 60)

        print(f"Loaded {len(self.pairs)} paired frames from {data_dir}")

    @staticmethod
    def _extract_id(filename):
        match = re.search(r'(\d+)', os.path.splitext(filename)[0])
        return match.group(1) if match else filename

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        x = torch.from_numpy(np.load(input_path)).float()   # (31, 7)
        y = torch.from_numpy(np.load(target_path)).float()   # (16, 4)
        return x, y
