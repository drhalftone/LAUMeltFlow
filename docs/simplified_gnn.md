# Simplified GNN for Flux Learning

**Date:** December 12, 2025

## Background

The original GNN (`meltflow_gnn/model.py`) was designed for general unstructured meshes with multiphase flows. For the 1D Sod shock tube problem (single-phase, uniform mesh), it was over-engineered.

This document describes a simplified GNN architecture that:
1. Retains the message-passing structure for future extensibility
2. Uses the same 6-input FluxMLP as our uniformly-trained model
3. Can directly load pre-trained weights without retraining

## Original vs Simplified Architecture

| Component | Original GNN | Simplified GNN |
|-----------|--------------|----------------|
| Node features | 5 (ρ, u, p, φ, x) | 3 (ρ, u, p) |
| Edge features | 2 (dx, normal) | 0 (uniform mesh) |
| FluxMLP inputs | 12 | 6 |
| Antisymmetric mode | Yes (optional) | Removed |
| ResidualBlocks | Yes | Removed |
| Parameters | ~270K | ~266K |

### Why the Original Had Extra Features

| Feature | Purpose | Why Unnecessary for Sod |
|---------|---------|------------------------|
| φ (level set) | Distinguish phases in multiphase flow | Single phase |
| x (position) | Spatially-varying behavior | Uniform physics |
| dx (edge attr) | Variable cell sizes | Uniform mesh |
| normal (edge attr) | Flux direction in 2D/3D | 1D problem |
| Antisymmetric | Guarantee F_ij = -F_ji | Degraded training (per GNN report) |

## Simplified Architecture

### SimpleFluxMLP

```
Input:  [ρ_L, u_L, p_L, ρ_R, u_R, p_R]  (6 features)
        ↓
        Linear(6, 256) + GELU
        ↓
        Linear(256, 256) + GELU  (×4 more layers)
        ↓
        Linear(256, 3)
        ↓
Output: [flux_ρ, flux_ρu, flux_E]  (3 fluxes)
```

This is identical to the standalone MLP trained with uniform sampling.

### SimpleFluxGNN (Message Passing)

```python
class SimpleFluxGNN(MessagePassing):
    def message(self, x_i, x_j):
        # x_j = left state, x_i = right state
        flux = self.flux_net(x_j, x_i)
        return flux / dx
```

The GNN computes flux at each edge via message passing, then aggregates (sums) at each node to get the flux divergence.

### SimpleEulerGNN (Full Model)

```python
def forward(self, x, edge_index, dt, dx):
    div_flux = self.flux_layer(x, edge_index, dx)
    return x - dt * div_flux
```

## Loading Pre-trained Weights

The key advantage: **no retraining required**.

```python
from meltflow_gnn.model_simple import SimpleEulerGNN, load_uniform_weights

# Create model with same architecture as trained MLP
model = SimpleEulerGNN(hidden_dim=256, n_layers=5)

# Load weights from uniformly-trained MLP
load_uniform_weights(model, "flux_model_cuda.npz")

# Now the GNN uses the pre-trained flux function
```

### Weight Mapping

The `load_uniform_weights()` function maps NumPy weights to PyTorch:

| NumPy (train_uniform.py) | PyTorch (model_simple.py) |
|--------------------------|---------------------------|
| `W0` (6, 256) | `mlp[0].weight` (256, 6).T |
| `b0` (256,) | `mlp[0].bias` (256,) |
| `W1` (256, 256) | `mlp[2].weight` (256, 256).T |
| ... | ... |
| `W5` (256, 3) | `mlp[10].weight` (3, 256).T |
| `b5` (3,) | `mlp[10].bias` (3,) |

Note: PyTorch stores weights as (out_features, in_features), so we transpose.

## Usage Example

```python
import torch
from meltflow_gnn.model_simple import SimpleEulerGNN, load_uniform_weights, create_1d_graph

# Create and load model
model = SimpleEulerGNN(hidden_dim=256, n_layers=5)
load_uniform_weights(model, "flux_model_cuda.npz")

# Create 1D mesh graph
n_cells = 100
edge_index = create_1d_graph(n_cells)

# Initial condition (Sod shock tube)
x = torch.zeros(n_cells, 3)
x[:50] = torch.tensor([1.0, 0.0, 100000.0])      # Left: high density/pressure
x[50:] = torch.tensor([0.125, 0.0, 10000.0])     # Right: low density/pressure

# Time step
dt, dx = 1e-6, 0.01
x_new = model(x, edge_index, dt, dx)
```

## When to Use Each Model

| Use Case | Recommended Model |
|----------|-------------------|
| 1D uniform mesh, single phase | `SimpleEulerGNN` or standalone MLP |
| 2D/3D unstructured mesh | Original `EulerGNN` |
| Multiphase flows (need φ) | Original `EulerGNN` |
| Variable mesh spacing | Original `EulerGNN` |
| Training from scratch | Standalone MLP + uniform sampling |
| Deploying in GNN framework | `SimpleEulerGNN` with loaded weights |

## File Locations

| File | Description |
|------|-------------|
| `meltflow_gnn/model.py` | Original GNN (12-input FluxMLP) |
| `meltflow_gnn/model_simple.py` | Simplified GNN (6-input FluxMLP) |
| `meltflow_gnn/train_uniform_cuda.py` | Training script for uniform MLP |
| `flux_model_cuda.npz` | Pre-trained weights (compatible with both) |
| `docs/uniform_sampling_experiment.md` | Training experiment details |

## Key Insight

The simplified GNN demonstrates that **for structured problems, the graph structure is mostly organizational**. The actual learning happens in the FluxMLP, which only needs local information (left and right states).

However, keeping the GNN wrapper provides:
1. **Clean abstraction**: Flux computation is separate from mesh traversal
2. **Future extensibility**: Same code works on irregular meshes
3. **Framework compatibility**: Integrates with PyTorch Geometric ecosystem
