"""
Sparse Quadtree for 2D Particle Positions.

Builds a quadtree from particle positions and tracks only active voxels
(those containing particles in their subtree). Provides mappings for
scatter/gather operations in the H2GNN.
"""

import torch
from typing import Tuple, List, Dict, Optional


class Quadtree:
    """
    Sparse quadtree for hierarchical particle aggregation.

    The quadtree has max_depth+1 levels (0 to max_depth):
    - Level 0: root (1 voxel)
    - Level 1: up to 4 voxels (2x2)
    - Level k: up to 4^k voxels (2^k x 2^k)
    - Level max_depth: leaf voxels (2^max_depth x 2^max_depth)

    Only "active" voxels are tracked - those containing particles in subtree.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        max_depth: int = 4,
        domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    ):
        """
        Build quadtree from particle positions.

        Args:
            positions: (N, 2) tensor of (x, y) positions
            max_depth: number of subdivision levels (4 = 16x16 leaf grid)
            domain: (xmin, xmax, ymin, ymax) bounding box
        """
        self.max_depth = max_depth
        self.domain = domain
        self.device = positions.device
        self.n_particles = positions.shape[0]

        xmin, xmax, ymin, ymax = domain
        self.domain_size = (xmax - xmin, ymax - ymin)

        # Compute leaf voxel indices for each particle
        # Leaf grid is 2^max_depth x 2^max_depth
        n_cells = 2 ** max_depth
        cell_size_x = (xmax - xmin) / n_cells
        cell_size_y = (ymax - ymin) / n_cells

        # Map positions to cell indices
        ix = ((positions[:, 0] - xmin) / cell_size_x).long().clamp(0, n_cells - 1)
        iy = ((positions[:, 1] - ymin) / cell_size_y).long().clamp(0, n_cells - 1)

        # Linear index in leaf grid (row-major)
        leaf_indices = iy * n_cells + ix  # (N,)

        # Build sparse structure bottom-up
        self._build_sparse_structure(leaf_indices, n_cells)

    def _build_sparse_structure(self, leaf_indices: torch.Tensor, n_cells: int):
        """Build active voxel lists and parent-child mappings."""

        # Find unique active leaves and map particles to them
        active_leaves, inverse = torch.unique(leaf_indices, return_inverse=True)
        self.particle_to_leaf = inverse  # (N,) maps particle -> local leaf index

        # Store active voxel info per level
        # active_voxels[level] = tensor of global voxel indices that are active
        # local_index[level] = dict mapping global index -> local index (0, 1, 2, ...)
        self.active_voxels: Dict[int, torch.Tensor] = {}
        self.local_index: Dict[int, Dict[int, int]] = {}

        # Level max_depth (leaves)
        self.active_voxels[self.max_depth] = active_leaves
        self.local_index[self.max_depth] = {
            int(v): i for i, v in enumerate(active_leaves.tolist())
        }

        # Build parent-child mappings
        # child_to_parent[level] = (num_active_at_level,) tensor
        #   maps local child index -> local parent index
        # parent_to_children[level] = (num_active_at_level,) tensor
        #   maps local child index -> local parent index (used for gather)
        self.child_to_parent: Dict[int, torch.Tensor] = {}
        self.parent_to_children: Dict[int, torch.Tensor] = {}

        # Propagate up from leaves to root
        current_active = active_leaves
        current_n_cells = n_cells

        for level in range(self.max_depth, 0, -1):
            # Parent grid is half the size
            parent_n_cells = current_n_cells // 2

            # Compute parent indices for active children
            # Child (ix, iy) -> Parent (ix//2, iy//2)
            child_iy = current_active // current_n_cells
            child_ix = current_active % current_n_cells
            parent_ix = child_ix // 2
            parent_iy = child_iy // 2
            parent_global = parent_iy * parent_n_cells + parent_ix

            # Find unique active parents
            active_parents, child_to_parent_idx = torch.unique(
                parent_global, return_inverse=True
            )

            # Store mappings
            self.active_voxels[level - 1] = active_parents
            self.local_index[level - 1] = {
                int(v): i for i, v in enumerate(active_parents.tolist())
            }
            self.child_to_parent[level] = child_to_parent_idx
            self.parent_to_children[level] = child_to_parent_idx  # Same tensor, used for gather

            # Move up
            current_active = active_parents
            current_n_cells = parent_n_cells

        # Compute voxel centers for each level
        self._compute_voxel_centers()

    def _compute_voxel_centers(self):
        """Compute center positions for all active voxels."""
        xmin, xmax, ymin, ymax = self.domain

        self.voxel_centers: Dict[int, torch.Tensor] = {}

        for level in range(self.max_depth + 1):
            n_cells = 2 ** level
            cell_size_x = (xmax - xmin) / n_cells
            cell_size_y = (ymax - ymin) / n_cells

            active = self.active_voxels[level]
            iy = active // n_cells
            ix = active % n_cells

            cx = xmin + (ix.float() + 0.5) * cell_size_x
            cy = ymin + (iy.float() + 0.5) * cell_size_y

            self.voxel_centers[level] = torch.stack([cx, cy], dim=-1)

    def num_active(self, level: int) -> int:
        """Return number of active voxels at given level."""
        return len(self.active_voxels[level])

    def get_particle_to_leaf(self) -> torch.Tensor:
        """Return (N,) tensor mapping particle_idx -> local leaf index."""
        return self.particle_to_leaf

    def get_child_to_parent(self, level: int) -> torch.Tensor:
        """
        Return mapping from children at `level` to parents at `level-1`.

        Returns (num_active_at_level,) tensor where entry i is the local
        index of the parent of child i.
        """
        return self.child_to_parent[level]

    def get_parent_to_children(self, level: int) -> torch.Tensor:
        """
        Return mapping for gather operation: broadcast parent to children.

        Returns (num_active_at_level,) tensor where entry i is the local
        index of the parent of child i. Use this to index parent tensor
        to broadcast to children.
        """
        return self.parent_to_children[level]

    def get_voxel_centers(self, level: int) -> torch.Tensor:
        """Return (num_active, 2) voxel center positions at given level."""
        return self.voxel_centers[level]

    def get_active_voxels(self, level: int) -> torch.Tensor:
        """Return global indices of active voxels at given level."""
        return self.active_voxels[level]


def test_quadtree():
    """Test quadtree construction with random particles."""
    torch.manual_seed(42)

    # Generate random particles
    n_particles = 50
    positions = torch.rand(n_particles, 2)

    # Build quadtree
    qt = Quadtree(positions, max_depth=4)

    print(f"Quadtree with {n_particles} particles, max_depth=4")
    print(f"Domain: {qt.domain}")
    print()

    for level in range(qt.max_depth + 1):
        n_active = qt.num_active(level)
        n_possible = 4 ** level
        print(f"Level {level}: {n_active}/{n_possible} active voxels "
              f"({100*n_active/n_possible:.1f}%)")

    print()
    print(f"particle_to_leaf shape: {qt.particle_to_leaf.shape}")
    print(f"particle_to_leaf unique values: {qt.particle_to_leaf.unique().shape[0]}")

    # Verify parent-child consistency
    for level in range(qt.max_depth, 0, -1):
        c2p = qt.get_child_to_parent(level)
        n_children = qt.num_active(level)
        n_parents = qt.num_active(level - 1)
        print(f"Level {level} -> {level-1}: {n_children} children -> {n_parents} parents")
        assert c2p.max() < n_parents, f"Invalid parent index at level {level}"

    print("\nQuadtree test passed!")


if __name__ == "__main__":
    test_quadtree()
