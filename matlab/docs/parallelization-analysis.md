# Parallelization Analysis for GNN Replacement

## Overview

This document analyzes the for loops in the MeltFlow MATLAB codebase to determine if they can be parallelized, with the goal of understanding the computational structure for a Graph Neural Network (GNN) replacement.

**Conclusion: Most loops CAN be parallelized.** The code structure maps naturally to GNN message-passing.

---

## Loop-by-Loop Analysis

### 1. `roe_perfect.m` - Roe Solver

| Loop | Location | Parallelizable? | Reason |
|------|----------|-----------------|--------|
| Flux calculation | lines 16-22 | **YES** | Each `F_iph(:,i)` is independent - reads `W(:,i)` and `W(:,i+1)`, writes to different index |
| Update step | lines 23-47 | **YES** | Each `W(k,i)` update only reads from pre-computed `F_iph` array (no write conflicts) |
| 2D x-sweep | lines 53-87 | **Already `parfor`** | Parallelizes over `j` (y-slices) |
| 2D y-sweep | lines 91-125 | **Already `parfor`** | Parallelizes over `i` (x-slices) |

**Key insight**: The update formula:
```matlab
W(k,i) = W(k,i) - dt/dx*(F_iph(k,i_r)-F_iph(k,i_l))
```
Only reads from `F_iph` (computed in previous loop) and writes to a unique location. No data dependency between iterations.

---

### 2. `advc.m` - Level Set Advection

| Loop | Location | Parallelizable? | Reason |
|------|----------|-----------------|--------|
| 1D update | lines 15-23 | **YES** | Reads from `In` (stored copy), writes to `I(i)` - no overlap |
| 2D update | lines 28-52 | **YES** | Reads from `In` (stored copy), writes to `I(i,j)` - no overlap |

**Key insight**: The code stores `In = I` before the loop, then reads from `In` and writes to `I`. This is the classic pattern for safe parallelization.

---

### 3. `ghost_GFM.m` - Ghost Fluid Method

| Loop | Location | Parallelizable? | Reason |
|------|----------|-----------------|--------|
| Entropy computation | lines 24-28, 50-54 | **YES** | Independent reads from `U`, writes to unique `s(i)` |
| Entropy inversion | lines 30-34, 56-60 | **YES** | Independent reads from `s`, writes to unique `UU(1,1,i)` |
| Copy loop | lines 74-82 | **YES** | Each iteration writes to different indices |
| 2D versions | lines 95-142 | **YES** | Same pattern, just nested loops |

**Exception**: The `extrp()` call (lines 29, 55) is a synchronization barrier - all entropy values must be computed before extrapolation. But the loops before/after `extrp()` are independently parallelizable.

---

### 4. `godunov.m` - Called Per-Node

This function is called per node and has no global state - it's already embarrassingly parallel at the caller level.

---

## Implications for GNN Replacement

### Why This Structure is Ideal for GNNs

1. **Message-passing structure**: The flux computation (`F_iph`) is essentially a message between neighboring nodes - exactly what GNNs do naturally.

2. **Local stencil operations**: Each node update depends only on:
   - Its own state `W(:,i)`
   - Neighbor states `W(:,i-1)`, `W(:,i+1)`
   - This maps perfectly to GNN edge aggregation

3. **No sequential dependencies within a timestep**: All nodes can be updated simultaneously given the fluxes.

4. **The computation graph**:
   ```
   Node states W^n → Compute fluxes F (neighbor interactions) → Update W^(n+1)
   ```
   This is exactly a single GNN layer.

---

## Recommended GNN Architecture

```
Input:
  - Node features: [rho, rho*u, (rho*v), E, phi, x, (y)]
  - Edge features: [dx, boundary_type]

GNN Layer ≈ One timestep:
  1. Edge function: Learn flux F_iph from neighboring node states
  2. Aggregation: Sum incoming/outgoing fluxes
  3. Node update: New state from old state + aggregated flux
```

The solver is already structured as a graph neural network - the GNN replaces the physics-based edge function (Roe flux) with a learned one.

---

## Summary Table

| Function | Total Loops | Parallelizable | Already Parallel |
|----------|-------------|----------------|------------------|
| `roe_perfect.m` | 6 | 4 | 2 (parfor) |
| `advc.m` | 2 | 2 | 0 |
| `ghost_GFM.m` | 10 | 10 | 0 |
| `godunov.m` | 2 | N/A (per-node) | N/A |

All node update loops can be executed in parallel, making this codebase well-suited for GPU acceleration or GNN replacement.
