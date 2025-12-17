"""
Hierarchical Hypergraph Neural Network with Attention (H2GNN-Attention).

Modification of H2GNN that uses:
1. Fixed-size padded voxels (k particles per voxel, padded with q=0)
2. Attention-based aggregation within voxels instead of scatter_sum
3. Proper masking for padded particles

Key insight: zero-charge particles don't contribute to physics, but we need
to mask attention so they don't influence the learned representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .quadtree import Quadtree


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for fixed-size sets."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) boolean mask, True = valid, False = padding

        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Apply mask: padded positions should not be attended to
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out


class AttentionBlock(nn.Module):
    """Transformer block: attention + feedforward with residuals."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VoxelPadding:
    """Utility for padding voxels to fixed size."""

    @staticmethod
    def pad_particles_to_voxels(
        particles: torch.Tensor,
        voxel_ids: torch.Tensor,
        n_voxels: int,
        k_per_voxel: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad particles into fixed-size voxels.

        Args:
            particles: (N, D) particle features (e.g., x, y, q)
            voxel_ids: (N,) voxel index for each particle
            n_voxels: total number of voxels
            k_per_voxel: fixed number of particles per voxel

        Returns:
            padded: (n_voxels, k_per_voxel, D) padded particle features
            mask: (n_voxels, k_per_voxel) boolean mask, True = real particle
            inverse: (N,) indices to unpad: original_particle[i] = padded[voxel_ids[i], inverse[i]]
        """
        N, D = particles.shape
        device = particles.device

        # Initialize padded tensor and mask
        padded = torch.zeros(n_voxels, k_per_voxel, D, device=device)
        mask = torch.zeros(n_voxels, k_per_voxel, dtype=torch.bool, device=device)
        inverse = torch.zeros(N, dtype=torch.long, device=device)

        # Count particles per voxel
        counts = torch.zeros(n_voxels, dtype=torch.long, device=device)

        # Fill padded tensor
        for i in range(N):
            v = voxel_ids[i].item()
            idx = counts[v].item()
            if idx < k_per_voxel:
                padded[v, idx] = particles[i]
                mask[v, idx] = True
                inverse[i] = idx
            counts[v] += 1

        return padded, mask, inverse

    @staticmethod
    def pad_particles_to_voxels_vectorized(
        particles: torch.Tensor,
        voxel_ids: torch.Tensor,
        n_voxels: int,
        k_per_voxel: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized version of pad_particles_to_voxels (faster for large N).
        """
        N, D = particles.shape
        device = particles.device

        # Sort particles by voxel
        sorted_indices = torch.argsort(voxel_ids)
        sorted_voxels = voxel_ids[sorted_indices]
        sorted_particles = particles[sorted_indices]

        # Count particles per voxel
        unique_voxels, counts = torch.unique(sorted_voxels, return_counts=True)

        # Initialize outputs
        padded = torch.zeros(n_voxels, k_per_voxel, D, device=device)
        mask = torch.zeros(n_voxels, k_per_voxel, dtype=torch.bool, device=device)
        inverse = torch.zeros(N, dtype=torch.long, device=device)

        # Fill each voxel
        ptr = 0
        for v, count in zip(unique_voxels.tolist(), counts.tolist()):
            n_fill = min(count, k_per_voxel)
            padded[v, :n_fill] = sorted_particles[ptr:ptr + n_fill]
            mask[v, :n_fill] = True

            # Track inverse mapping
            for local_idx in range(n_fill):
                orig_idx = sorted_indices[ptr + local_idx].item()
                inverse[orig_idx] = local_idx

            ptr += count

        return padded, mask, inverse


class H2GNNAttention(nn.Module):
    """
    Hierarchical GNN with attention-based aggregation.

    Key differences from original H2GNN:
    1. Particles are padded to fixed size per voxel
    2. Attention is used within voxels instead of scatter_sum
    3. Aggregation to parent uses attention-weighted pooling
    """

    def __init__(
        self,
        max_depth: int = 4,
        particle_dim: int = 3,      # (x, y, q)
        hidden_dim: int = 64,
        output_dim: int = 2,        # (Ex, Ey)
        n_heads: int = 4,
        n_attn_layers: int = 2,
        k_per_voxel: int = 16,      # max particles per leaf voxel
        dropout: float = 0.0
    ):
        super().__init__()
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        self.particle_dim = particle_dim
        self.k_per_voxel = k_per_voxel

        # Particle encoder: (x, y, q, rel_x, rel_y) -> hidden
        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention blocks for within-voxel processing at leaf level
        self.leaf_attention = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_attn_layers)
        ])

        # Pooling attention: aggregate particles to voxel representation
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = MultiHeadAttention(hidden_dim, n_heads, dropout)

        # Encoder MLPs (process aggregated children at each level)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(max_depth)
        ])

        # Decoder MLPs (parent + skip -> children)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(max_depth)
        ])

        # Broadcast attention: expand voxel state back to particles
        self.broadcast_attn = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_attn_layers)
        ])

        # Output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim + particle_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
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
            quadtree: Quadtree object

        Returns:
            E: (N, 2) electric field at each particle
        """
        N = particles.shape[0]
        device = particles.device
        max_depth = self.max_depth

        # ============== PADDING & RELATIVE POSITIONS ==============

        # Get particle-to-leaf mapping
        particle_to_leaf = quadtree.get_particle_to_leaf()
        n_leaves = quadtree.num_active(max_depth)
        leaf_centers = quadtree.get_voxel_centers(max_depth)

        # Compute relative positions
        positions = particles[:, :2]
        particle_leaf_centers = leaf_centers[particle_to_leaf]
        relative_pos = positions - particle_leaf_centers
        particles_augmented = torch.cat([particles, relative_pos], dim=-1)  # (N, 5)

        # Pad particles to fixed-size voxels
        padded, mask, inverse_idx = VoxelPadding.pad_particles_to_voxels_vectorized(
            particles_augmented, particle_to_leaf, n_leaves, self.k_per_voxel
        )
        # padded: (n_leaves, k, 5), mask: (n_leaves, k)

        # ============== PARTICLE ENCODING ==============

        # Encode particles
        h = self.particle_encoder(padded)  # (n_leaves, k, hidden)

        # Apply attention within each voxel
        for attn_block in self.leaf_attention:
            h = attn_block(h, mask)

        # Store particle-level features for later
        h_particles_in_voxels = h  # (n_leaves, k, hidden)

        # ============== POOL TO VOXEL REPRESENTATION ==============

        # Attention pooling: use learnable query to aggregate particles
        # Query attends to all particles in voxel
        batch_query = self.pool_query.expand(n_leaves, -1, -1)  # (n_leaves, 1, hidden)
        h_with_query = torch.cat([batch_query, h], dim=1)  # (n_leaves, 1+k, hidden)
        mask_with_query = torch.cat([
            torch.ones(n_leaves, 1, dtype=torch.bool, device=device),
            mask
        ], dim=1)

        h_pooled = self.pool_attn(h_with_query, mask_with_query)
        h_voxels = h_pooled[:, 0, :]  # Take query output: (n_leaves, hidden)

        # ============== ENCODER (scatter up hierarchy) ==============

        encoder_states = {max_depth: h_voxels}

        h = h_voxels
        for level in range(max_depth, 0, -1):
            # Get parent mapping
            child_to_parent = quadtree.get_child_to_parent(level)
            n_parents = quadtree.num_active(level - 1)

            # Aggregate children to parents (using mean for simplicity)
            # TODO: could use attention here too for richer aggregation
            h_aggregated = torch.zeros(n_parents, self.hidden_dim, device=device)
            h_aggregated.scatter_add_(0, child_to_parent.unsqueeze(-1).expand_as(h), h)

            # Count children per parent for mean
            counts = torch.zeros(n_parents, device=device)
            counts.scatter_add_(0, child_to_parent, torch.ones(h.shape[0], device=device))
            h_aggregated = h_aggregated / counts.unsqueeze(-1).clamp(min=1)

            # Apply encoder MLP
            h = self.encoders[level - 1](h_aggregated)
            encoder_states[level - 1] = h

        # h is now root state: (1, hidden)

        # ============== DECODER (broadcast down hierarchy) ==============

        for level in range(1, max_depth + 1):
            # Broadcast parent to children
            parent_to_children = quadtree.get_parent_to_children(level)
            h_broadcast = h[parent_to_children]

            # Skip connection
            h_skip = encoder_states[level]
            h_combined = torch.cat([h_broadcast, h_skip], dim=-1)

            # Decode
            h = self.decoders[level - 1](h_combined)

        # h is now leaf voxel states: (n_leaves, hidden)

        # ============== BROADCAST TO PARTICLES ==============

        # Expand voxel state and combine with particle features
        h_voxel_expanded = h.unsqueeze(1).expand(-1, self.k_per_voxel, -1)  # (n_leaves, k, hidden)

        # Add voxel context to particle features
        h_particles = h_particles_in_voxels + h_voxel_expanded

        # Apply broadcast attention
        for attn_block in self.broadcast_attn:
            h_particles = attn_block(h_particles, mask)

        # ============== OUTPUT ==============

        # Gather back to original particle order
        h_output = torch.zeros(N, self.hidden_dim, device=device)
        for i in range(N):
            v = particle_to_leaf[i].item()
            local_idx = inverse_idx[i].item()
            h_output[i] = h_particles[v, local_idx]

        # Concat with original features and decode
        h_final = torch.cat([h_output, particles_augmented], dim=-1)
        E = self.output_decoder(h_final)

        return E


def test_h2gnn_attention():
    """Test H2GNN-Attention forward pass."""
    torch.manual_seed(42)

    # Generate random particles
    n_particles = 100
    positions = torch.rand(n_particles, 2)
    charges = torch.rand(n_particles) * 2 - 1
    particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)

    # Build quadtree
    quadtree = Quadtree(positions, max_depth=4)

    # Create model
    model = H2GNNAttention(
        max_depth=4,
        hidden_dim=64,
        n_heads=4,
        k_per_voxel=16
    )

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

    # Compare with original H2GNN
    from .h2gnn import H2GNN
    original = H2GNN(max_depth=4, hidden_dim=64)
    n_params_original = sum(p.numel() for p in original.parameters())
    print(f"Original H2GNN parameters: {n_params_original:,}")

    print("\nH2GNN-Attention test passed!")


if __name__ == "__main__":
    test_h2gnn_attention()
