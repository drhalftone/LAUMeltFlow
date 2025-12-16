"""
Hierarchical Hypergraph Neural Network (H2GNN) for 2D Electrostatics.

This package implements a hierarchical hypergraph neural network that learns
to compute electric fields from charged particle configurations. Particles
are nodes, quadtree voxels are hyperedges, with weight sharing per level
and sparse processing of only active voxels.

Key components:
- Quadtree: Sparse spatial hierarchy for particles
- H2GNN: Encoder-decoder model with scatter/gather operations
- Dataset: Ground truth E field from Coulomb's law
"""

from .quadtree import Quadtree
from .h2gnn import H2GNN, MLP, scatter_sum
from .dataset import ElectrostaticDataset, generate_sample, compute_coulomb_field_2d

__all__ = [
    'Quadtree',
    'H2GNN',
    'MLP',
    'scatter_sum',
    'ElectrostaticDataset',
    'generate_sample',
    'compute_coulomb_field_2d',
]
