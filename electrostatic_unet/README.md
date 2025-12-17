# Electrostatic U-Net: Hierarchical Graph Neural Networks

This module implements hierarchical graph neural networks for learning electromagnetic field predictions from charged particle configurations.

## Models

### H2GNN (Original)

Hierarchical Hypergraph Neural Network using scatter/sum aggregation.

- **Aggregation**: `scatter_sum` (fast but lossy)
- **Parameters**: ~92k
- **Complexity**: O(N log N)

### H2GNN-Attention (New)

Improved architecture with attention-based aggregation and fixed-size padded voxels.

- **Aggregation**: Multi-head attention within voxels
- **Parameters**: ~243k
- **Complexity**: O(N log N) with larger constant factor
- **Key innovation**: Zero-charge padding enables fixed-size attention

**Key improvements over original:**
1. Attention preserves more information than sum aggregation
2. Particles within a voxel can attend to each other
3. Proper masking ensures padded particles don't influence outputs
4. Richer learned aggregation through attention pooling

## Quick Start

### Test the models

```python
# Test original H2GNN
python -m electrostatic_unet.h2gnn

# Test H2GNN-Attention
python -m electrostatic_unet.h2gnn_attention
```

### Train original H2GNN

```bash
python -m electrostatic_unet.train \
    --epochs 200 \
    --n-train 1000 \
    --hidden-dim 64
```

### Compare H2GNN vs H2GNN-Attention

```bash
# Windows
compare_h2gnn.bat

# Or directly
python -m electrostatic_unet.train_comparison \
    --epochs 200 \
    --n-train 500 \
    --hidden-dim 64 \
    --k-per-voxel 16 \
    --device cuda
```

## Architecture

### H2GNN (Original)

```
Particles → Encode → scatter_sum to voxels → Hierarchy up → Root
                                                              ↓
Output ← Decode ← gather from voxels ← Hierarchy down ← Root
```

### H2GNN-Attention

```
Particles → Encode → Pad to fixed K → Self-attention → Attention pool → Hierarchy up → Root
                                                                                         ↓
Output ← Decode ← Attention broadcast ← Gather from voxels ← Hierarchy down ← Root
```

**Key difference**: Fixed-size padded voxels enable attention operations:
- Zero-charge particles (q=0) are added to fill voxels to size K
- Attention masks prevent padded particles from influencing real ones
- Physics is preserved: q=0 particles contribute nothing to the field

## Files

```
electrostatic_unet/
├── __init__.py              # Package exports
├── quadtree.py              # Sparse quadtree data structure
├── h2gnn.py                 # Original H2GNN (scatter_sum)
├── h2gnn_attention.py       # H2GNN-Attention (attention aggregation)
├── dataset.py               # Data generation (Coulomb field)
├── train.py                 # Training loop for H2GNN
├── train_comparison.py      # Compare H2GNN vs H2GNN-Attention
├── evaluate.py              # Visualization and metrics
└── README.md                # This file
```

## Hyperparameters

### H2GNN

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 4 | Quadtree depth (16x16 leaf grid) |
| `hidden_dim` | 64 | Hidden dimension for MLPs |
| `n_mlp_layers` | 2 | Layers per MLP |

### H2GNN-Attention

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 4 | Quadtree depth |
| `hidden_dim` | 64 | Hidden dimension |
| `n_heads` | 4 | Attention heads |
| `n_attn_layers` | 2 | Attention blocks at leaf/broadcast |
| `k_per_voxel` | 16 | Fixed particles per voxel (padding size) |
| `dropout` | 0.0 | Dropout rate |

## Comparison Results

After training, check:
- `checkpoints/comparison/comparison_plot.png` - Training curves
- `checkpoints/comparison/comparison_results.json` - Loss data
- `checkpoints/comparison/H2GNN_best.pt` - Best H2GNN model
- `checkpoints/comparison/H2GNN-Attention_best.pt` - Best attention model

## References

The attention-based approach draws from:

- **Set Transformer** (Lee et al., 2019) - Attention-based set pooling
- **PointNet++** (Qi et al., 2017) - Hierarchical point cloud processing
- **Graph Attention Networks** (Veličković et al., 2018) - Attention for graphs
- **Multipole Graph Neural Operator** (Li et al., 2020) - Hierarchical physics

## Physics Background

The models learn to predict electric fields from Coulomb's law:

$$\mathbf{E}_i = \sum_{j \neq i} \frac{q_j (\mathbf{r}_i - \mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^2}$$

Ground truth is computed via O(N²) brute force. The hierarchical structure enables O(N log N) inference, similar to Barnes-Hut or Fast Multipole Methods but with learned approximations.
