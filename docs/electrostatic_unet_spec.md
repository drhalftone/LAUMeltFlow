# Hierarchical Hypergraph Neural Network for Electromagnetic Particle Simulation

## Overview

This project implements a hierarchical hypergraph neural network (H²GNN) that learns electromagnetic interactions between charged particles. The architecture uses a quadtree/octree spatial hierarchy where particles are nodes and voxels at each level serve as hyperedges, enabling efficient multi-scale message passing with sparse processing of only occupied regions.

**Current implementation**: 2D electrostatics with quadtree (extensible to 3D octree).

## Implementation Status

### Completed (v1.0)

- **Quadtree**: Sparse 2D spatial hierarchy, only tracks active voxels
- **H2GNN Model**: Encoder-decoder with weight-shared MLPs per level
- **Dataset**: Vectorized 2D Coulomb field computation
- **Training**: Full training loop with validation and checkpointing
- **Evaluation**: Visualization and metrics

### Files

```
electrostatic_unet/
├── __init__.py        # Package exports
├── quadtree.py        # Sparse quadtree data structure
├── h2gnn.py           # H²GNN model architecture
├── dataset.py         # Data generation with Coulomb field
├── train.py           # Training loop
└── evaluate.py        # Visualization and metrics
```

## Quick Start

**Train:**
```bash
python -m electrostatic_unet.train --epochs 100 --n-train 1000
```

**Evaluate:**
```bash
python -m electrostatic_unet.evaluate --checkpoint checkpoints/h2gnn_best.pt
```

**Test modules:**
```python
from electrostatic_unet.quadtree import test_quadtree
from electrostatic_unet.h2gnn import test_h2gnn
from electrostatic_unet.dataset import test_dataset

test_quadtree()
test_h2gnn()
test_dataset()
```

## Motivation

We aim to build a neural network-based particle simulation system where:

1. Charged particles exist as graph nodes with state (position, charge, ...)
2. A quadtree/octree partitions space into a hierarchy of voxels
3. Voxels act as hyperedges connecting the particles (or child voxels) they contain
4. Message passing propagates information up and down the hierarchy
5. Particles receive electric field predictions based on local and global information

This approach offers several advantages over traditional methods:
- **Adaptive resolution**: Quadtree provides fine detail where particles cluster
- **Sparse processing**: Empty voxels are skipped entirely
- **Learned physics**: Network learns electromagnetic interactions from data
- **End-to-end differentiable**: Enables gradient-based optimization

## Physics Background

For static charged particles, the electric field is governed by Coulomb's law.

**2D Coulomb's law** (current implementation):
$$\mathbf{E}_i = \sum_{j \neq i} \frac{q_j (\mathbf{r}_i - \mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^2}$$

**3D Coulomb's law** (future):
$$\mathbf{E}_i = \sum_{j \neq i} \frac{q_j (\mathbf{r}_i - \mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$$

The hierarchical structure naturally captures the multi-scale nature of these interactions: nearby particles interact strongly (fine voxels), while distant particles contribute through aggregated effects (coarse voxels).

## Architecture

### Current Configuration

- **Dimensions**: 2D (quadtree) - 3D octree planned
- **Particles**: 10-100 per sample
- **Quadtree depth**: 4 levels (16×16 leaf resolution)
- **Input**: N particles with (x, y, q)
- **Output**: N electric field vectors (Ex, Ey)

### Nodes: Charged Particles

Each particle is a node with feature vector:
- Position: $(x, y)$ [or $(x, y, z)$ in 3D]
- Charge: $q$
- (Future) Velocity, acceleration, mass

### Hyperedges: Quadtree Voxels

The spatial domain is partitioned by a quadtree. Each voxel at each level acts as a hyperedge:

```
Level 0 (root):     [────────────]     1 voxel covering entire domain
                          ↓
Level 1:            [──────][──────]   4 child voxels (2²)
                       ↓        ↓
Level 2:            [──][──] [──][──]  16 voxels (4²)
                     ↓   ↓
Level 3 (leaves):   [·][·]             64 voxels (8²) - finest level
                     ↓
                   p1,p2               Particles as nodes
```

**Key properties**:
- Leaf voxels connect 0, 1, or more particles (hyperedge)
- Parent voxels connect their 4 child voxels (hyperedge) [8 in 3D]
- Empty voxels (no particles in subtree) are not processed

### Sparse Processing

Only "active" voxels are processed:
- A leaf voxel is active if it contains ≥1 particle
- A non-leaf voxel is active if it has ≥1 active child

This gives O(N log N) complexity for N particles, similar to Barnes-Hut or FMM methods, rather than O(V) for V total voxels.

### Weight Sharing

Each level has its own MLP weights, shared across all voxels at that level:
- **Translation invariant**: Same processing regardless of voxel position
- **Parameter efficient**: Only need L networks for L levels
- **Scale-appropriate**: Each level learns operations suited to its spatial scale

```
Level 0 (root):     [MLP_0]           ← one network for root
                       ↓
Level 1:            [MLP_1] [MLP_1]   ← same network, all level-1 voxels
                       ↓       ↓
Level 2:            [MLP_2] [MLP_2]   ← same network, all level-2 voxels
                       ↓
Level 3 (leaves):   [MLP_3] [MLP_3]   ← same network, all leaf voxels
```

### Encoder (Scatter / Up-Pass)

Information flows from particles up through the hierarchy:

1. **Particle → Hidden**: MLP encodes particle features (x, y, q, rel_x, rel_y) → hidden state
2. **Particles → Leaf Voxel**: Scatter with sum aggregation (preserves particle count information)
3. **Child → Parent**: Scatter children to parent voxels with sum aggregation
4. **Level MLP**: Apply level-specific MLP to aggregated state
5. Repeat until root

```python
# Relative position encoding
relative_pos = positions - leaf_centers[particle_to_leaf]
particles_augmented = concat(particles, relative_pos)  # (N, 5)

h_particles = particle_encoder(particles_augmented)    # (N, hidden)
h_leaves = scatter_sum(h_particles, particle_to_leaf)  # (n_leaves, hidden)

for level in range(max_depth, 0, -1):
    h = scatter_sum(h, child_to_parent[level])
    h = encoders[level-1](h)
```

**Why sum aggregation?** Mean aggregation loses information about how many particles contributed. Sum preserves this, which is physically meaningful: the total field contribution from a voxel depends on both the charges and the number of particles.

### Decoder (Gather / Down-Pass)

Information flows from root back down to particles:

1. **Parent → Child**: Gather (broadcast) parent state to children
2. **Skip Connection**: Concatenate with encoder state at same level
3. **Level MLP**: Apply level-specific decoder MLP
4. Repeat until leaf voxels
5. **Leaf → Particle**: Gather leaf state to particles
6. **Output MLP**: Predict E field from (leaf_hidden, particle_features)

```python
for level in range(1, max_depth + 1):
    h_broadcast = h[parent_to_children[level]]
    h = decoders[level-1](concat(h_broadcast, encoder_states[level]))

h_to_particles = h[particle_to_leaf]
E = particle_decoder(concat(h_to_particles, particles))  # (N, 2)
```

### Skip Connections

Like a U-Net, each level has skip connections between encoder and decoder:
- Encoder state at level L is concatenated with decoder input at level L
- Final particle output includes original (x, y, q) features
- Preserves fine-grained spatial information through the bottleneck

## Data Generation

Training data uses analytical Coulomb field computed via brute-force O(N²) pairwise summation:

**Input**: N random particles in [0, 1]² with charges in [-1, 1]
**Output**: Ground truth E field at each particle location

```python
# Vectorized 2D Coulomb field - O(N²) computation
r = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2) displacement vectors
r_mag = torch.norm(r, dim=-1)  # (N, N) distances

# E_i = Σ_{j≠i} q_j * r_ij / |r_ij|²
E_contributions = charges * r / r_mag**2  # (N, N, 2)
E = E_contributions.sum(dim=1)  # (N, 2) - sum over all j
```

The key insight is that ground truth computation is O(N²), but the trained network approximates this with O(N log N) complexity using the hierarchical structure—similar to how Barnes-Hut or Fast Multipole Methods work, but with learned rather than analytical approximations.

## Training

**Training Loop**:
1. Generate random particle configuration (positions + charges)
2. Compute ground truth E field using O(N²) Coulomb summation
3. Build quadtree from particle positions
4. Forward pass through H²GNN → predicted E field
5. Compute loss = MSE(E_pred, E_true)
6. Backpropagate and update weights

**Hyperparameters**:
- **Loss**: MSE between predicted and ground truth E field
- **Optimizer**: Adam with learning rate ~1e-4
- **Scheduler**: ReduceLROnPlateau (halve LR on plateau)
- **Checkpointing**: Best model and periodic saves

**Goal**: Train the network to approximate the expensive O(N²) Coulomb computation using the efficient O(N log N) hierarchical structure. Once trained, inference is fast.

## Evaluation

1. **MSE/RMSE**: Mean squared error on E field
2. **MAE**: Mean absolute error
3. **Relative Error**: Error normalized by field magnitude
4. **Visualization**: Side-by-side ground truth vs predicted E vectors

## Comparison: H²GNN vs Convolutional U-Net

| Aspect | Convolutional U-Net | H²GNN (This Project) |
|--------|---------------------|----------------------|
| Input | Fixed grid (charge density) | Variable particle set |
| Layers | 2D/3D convolutions | Hypergraph message passing |
| Processing | Dense (all grid cells) | Sparse (active voxels only) |
| Resolution | Fixed grid size | Adaptive quadtree/octree |
| Output | Field on grid | E field at particles |
| Complexity | O(grid size) | O(N log N) for N particles |

## Future Extensions

1. **3D Octree**: Extend to 3D with 8 children per voxel
2. **Moving charges**: Add velocity, acceleration; predict trajectories
3. **Magnetic fields**: Include Biot-Savart and Lorentz force
4. **Adaptive refinement**: Dynamic octree depth based on particle density
5. **Hybrid methods**: Combine with analytical short-range forces
6. **Attention aggregation**: Add attention-weighted pooling alongside sum

## Success Criteria

1. E field prediction accuracy significantly better than baseline
2. Generalization to unseen particle counts and configurations
3. Computational efficiency: sparse processing scales with O(N log N)
4. Extensibility to 3D and dynamic simulations
