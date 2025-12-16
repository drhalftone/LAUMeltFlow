"""
Hierarchical Hypergraph Neural Network (H2GNN) for 2D Electrostatics.

Learns to compute electric field at particle locations from particle
configurations using a quadtree-based hierarchical message passing scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quadtree import Quadtree


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter sum aggregation.

    Args:
        src: (N, D) source tensor
        index: (N,) indices into output tensor
        dim_size: size of output dimension 0

    Returns:
        out: (dim_size, D) tensor where out[i] = sum of src[index == i]
    """
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class H2GNN(nn.Module):
    """
    Hierarchical Hypergraph Neural Network for 2D electrostatics.

    Architecture:
    - Encoder: particles -> leaf voxels -> ... -> root (scatter/aggregate)
    - Decoder: root -> ... -> leaf voxels -> particles (broadcast/gather)
    - Skip connections at each level
    - Relative position encoding: particle position relative to voxel center

    Weight sharing: same MLP for all voxels at each level.
    Sparse processing: only active voxels are computed.
    """

    def __init__(
        self,
        max_depth: int = 4,
        particle_dim: int = 3,      # (x, y, q)
        hidden_dim: int = 64,
        output_dim: int = 2,        # (Ex, Ey)
        n_mlp_layers: int = 2
    ):
        super().__init__()
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        self.particle_dim = particle_dim

        # Encoder: particles -> hidden
        # Input: (x, y, q, rel_x, rel_y) = particle_dim + 2 for relative position
        self.particle_encoder = MLP(
            particle_dim + 2, hidden_dim, hidden_dim, n_mlp_layers
        )

        # Encoder MLPs (one per level, shared across voxels at that level)
        # Process aggregated child states at each level
        self.encoders = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim, n_mlp_layers)
            for _ in range(max_depth)
        ])

        # Decoder MLPs (one per level)
        # Input: parent_hidden (broadcast) + skip_hidden (from encoder)
        self.decoders = nn.ModuleList([
            MLP(hidden_dim * 2, hidden_dim, hidden_dim, n_mlp_layers)
            for _ in range(max_depth)
        ])

        # Decoder: leaf voxel states -> particle E field
        # Input: leaf_hidden + particle_features + relative_position
        self.particle_decoder = MLP(
            hidden_dim + particle_dim + 2, output_dim, hidden_dim, n_mlp_layers
        )

    def forward(
        self,
        particles: torch.Tensor,
        quadtree: Quadtree
    ) -> torch.Tensor:
        """
        Forward pass: compute E field at each particle.

        Args:
            particles: (N, 3) tensor of (x, y, q)
            quadtree: Quadtree object with sparse connectivity

        Returns:
            E: (N, 2) electric field at each particle
        """
        max_depth = self.max_depth

        # ============== RELATIVE POSITION ENCODING ==============

        # Get particle positions and leaf voxel centers
        positions = particles[:, :2]  # (N, 2)
        particle_to_leaf = quadtree.get_particle_to_leaf()
        leaf_centers = quadtree.get_voxel_centers(max_depth)  # (n_leaves, 2)

        # Compute relative position: particle pos - voxel center
        particle_leaf_centers = leaf_centers[particle_to_leaf]  # (N, 2)
        relative_pos = positions - particle_leaf_centers  # (N, 2)

        # Augment particle features with relative position
        particles_augmented = torch.cat([particles, relative_pos], dim=-1)  # (N, 5)

        # ============== ENCODER (scatter up) ==============

        # Particles -> hidden features (with relative position)
        h_particles = self.particle_encoder(particles_augmented)  # (N, hidden)

        # Scatter particles to leaf voxels (sum aggregation)
        n_leaves = quadtree.num_active(max_depth)
        h_leaves = scatter_sum(h_particles, particle_to_leaf, n_leaves)

        # Store encoder states for skip connections
        encoder_states = {max_depth: h_leaves}

        # Propagate up: leaves -> root
        h = h_leaves
        for level in range(max_depth, 0, -1):
            # Scatter children to parents (sum aggregation)
            child_to_parent = quadtree.get_child_to_parent(level)
            n_parents = quadtree.num_active(level - 1)
            h_aggregated = scatter_sum(h, child_to_parent, n_parents)

            # Apply level-specific MLP
            h = self.encoders[level - 1](h_aggregated)

            # Store for skip connection
            encoder_states[level - 1] = h

        # h is now root state: (1, hidden)

        # ============== DECODER (gather down) ==============

        # Propagate down: root -> leaves
        for level in range(1, max_depth + 1):
            # Gather: broadcast parent state to children
            # parent_to_children[i] = index of parent for child i
            parent_to_children = quadtree.get_parent_to_children(level)
            h_broadcast = h[parent_to_children]  # (num_children, hidden)

            # Skip connection: concat with encoder state at this level
            h_skip = encoder_states[level]
            h_combined = torch.cat([h_broadcast, h_skip], dim=-1)

            # Apply level-specific decoder MLP
            h = self.decoders[level - 1](h_combined)

        # h is now leaf states: (n_leaves, hidden)

        # Gather: broadcast leaf state to particles
        h_to_particles = h[particle_to_leaf]  # (N, hidden)

        # Concat with augmented particle features (includes relative position)
        h_final = torch.cat([h_to_particles, particles_augmented], dim=-1)

        # Output E field
        E = self.particle_decoder(h_final)  # (N, 2)

        return E


def test_h2gnn():
    """Test H2GNN forward pass."""
    torch.manual_seed(42)

    # Generate random particles
    n_particles = 50
    positions = torch.rand(n_particles, 2)
    charges = torch.rand(n_particles) * 2 - 1
    particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)

    # Build quadtree
    quadtree = Quadtree(positions, max_depth=4)

    # Create model
    model = H2GNN(max_depth=4, hidden_dim=32)

    # Forward pass
    E = model(particles, quadtree)

    print(f"Input particles: {particles.shape}")
    print(f"Output E field: {E.shape}")
    print(f"E range: [{E.min().item():.4f}, {E.max().item():.4f}]")

    # Test backward pass
    loss = E.pow(2).mean()
    loss.backward()
    print(f"Backward pass successful, loss = {loss.item():.6f}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nH2GNN test passed!")


if __name__ == "__main__":
    test_h2gnn()
