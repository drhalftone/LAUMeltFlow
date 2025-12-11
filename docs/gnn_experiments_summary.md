# GNN Experiments Summary

This document summarizes the experiments conducted to improve the GNN flux predictor for the 1D Sod shock tube problem.

## Error Analysis

### Error Distribution by Region

Analysis of the final timestep errors reveals that the largest errors are **not at the shock**, but at the **right boundary**:

| Region | X Range | RMS Density | RMS Velocity | RMS Pressure |
|--------|---------|-------------|--------------|--------------|
| Expansion fan | x < 0.3 | 1.35e-02 | 5 | 1,824 |
| Contact/Shock | 0.3 - 0.7 | 6.13e-02 | 22 | 6,922 |
| **Right boundary** | x >= 0.7 | **7.08e-02** | **95** | **7,460** |

Top 5 error locations are at x = 1.00, 0.99, 0.98, 0.97, and 0.41 (contact discontinuity).

### Root Cause: Training Data Imbalance

At x = 1.00 (right boundary), the final state is:
- **MeltFlow**: rho=0.099, u=-105, p=6,389 (rapid expansion)
- **GNN**: rho=0.282, u=293, p=28,640 (much more moderate)

The GNN cannot follow the steep downward trajectory because **extreme states are underrepresented in training data**:

| Condition | Samples | % of Right Region Data |
|-----------|---------|------------------------|
| rho < 0.15 | 12,059 | 69% (adequate) |
| **u < 0** (negative velocity) | **315** | **1.8%** (rare) |
| **p < 10,000** | **296** | **1.7%** (rare) |

The final boundary state (rho~0.1, u~-100, p~6000) is a combination that appears only in the **last few timesteps** of the simulation. The GNN is essentially **extrapolating into unseen territory**.

### Key Insight

The problem is not Gibbs phenomenon at the shock - it's that:
1. Neural networks are biased toward smooth interpolation
2. Extreme boundary states (negative velocity, low pressure) are rare in training
3. The GNN learned "average" behavior, not extreme tails

---

## Baseline Configuration

- **Grid**: 100 nodes (dx = 0.01)
- **Architecture**: 256 hidden units, 5 layers, residual blocks with LayerNorm
- **Training**: 500 epochs, MSE loss, no conservation constraints
- **Validation loss**: ~0.000055-0.000109
- **Final RMS errors**: Density ~5e-02, Velocity ~50, Pressure ~5,000-6,000

## Experiments

### 1. Conservation Constraints

Tested multiple approaches to enforce physical conservation laws. **All failed.**

| Approach | Val Loss | Result |
|----------|----------|--------|
| Hard antisymmetric (F_ij = -F_ji) | 0.00384 | Simulation exploded |
| Soft conservation (weight=0.1) | 0.000095 | Errors doubled |
| Soft conservation (weight=0.01) | 0.000085 | Unstable |
| Global conservation (weight=0.01) | 0.000052 | Unstable |
| Global conservation (weight=0.001) | 0.000086 | Unstable |

**Conclusion**: The Roe flux is not antisymmetric by design. Conservation emerges naturally from the finite volume update structure. Adding constraints degrades performance.

### 2. Residual Connections + LayerNorm

Added residual blocks with LayerNorm and GELU activation to the FluxMLP.

| Metric | Baseline | Residual+LayerNorm |
|--------|----------|-------------------|
| Val Loss | 0.000055 | 0.000081 |
| Train Loss | 0.000057 | 0.000028 |
| Density RMS | 5e-02 | 5.16e-02 |
| Velocity RMS | 48 | 53.8 |
| Pressure RMS | 5,500 | 4,350 |

**Conclusion**: Mixed results. Lower training loss suggests better fitting, but higher validation loss indicates slight overfitting. Pressure improved, velocity slightly worse. Training still plateaus around epoch 150.

### 3. Finer Grid (400 Nodes)

Attempted to improve accuracy by using 4x finer spatial resolution (dx = 0.0025, 400 nodes).

| Metric | 100 Nodes | 400 Nodes |
|--------|-----------|-----------|
| Training samples | 563 | 224 |
| Val Loss | 0.000069 | 0.000106 |
| Simulation | Stable | **Unstable** |

**Conclusion**: **Failed due to Gibbs phenomenon.**

The Gibbs phenomenon causes oscillations near discontinuities (shocks) that don't diminish with finer grids - they just become sharper and more localized. With 400 nodes:

1. **More edges around the shock** = more samples with extreme flux gradients
2. **Fewer total training samples** due to CFL-constrained timestep (dt ~ dx)
3. **Harder learning task** - the GNN must learn more "outlier" flux values near the discontinuity
4. The shock is a true discontinuity - finer resolution doesn't help the GNN generalize

This is a fundamental limitation: mesh refinement near shocks doesn't improve GNN training because the underlying physics has a discontinuity.

## Key Findings

1. **Standard MSE loss works best** - no conservation constraints needed
2. **Architecture changes provide marginal improvement** - the bottleneck is data, not model capacity
3. **Finer grids don't help** - Gibbs phenomenon makes shock regions harder to learn
4. **Training plateaus early** (~epoch 150) regardless of architecture

## Recommendations

For future improvement, consider:

1. **More diverse training data** - different initial conditions, shock strengths, positions
2. **Shock-aware features** - add gradient magnitude as a node feature
3. **Multi-scale training** - train on coarse grid, fine-tune on finer regions away from shocks
4. **Flux limiting** - apply TVD/ENO-style limiters to GNN-predicted fluxes
5. **Separate shock treatment** - use classical Roe flux near detected discontinuities, GNN elsewhere

## Current Best Model

- **Config**: `in_1Dsod1fl` (100 nodes)
- **Architecture**: ResidualBlock + LayerNorm, 256 hidden, 5 layers
- **File**: `flux_model.pt`
- **Performance**: Stable simulation, RMS errors (density ~5e-02, velocity ~54, pressure ~6,000)
