# MeltFlow MATLAB Codebase - Chat Summary

## Overview

This document summarizes a technical discussion about the MeltFlow MATLAB codebase, covering code analysis, improvements made, and exploration of replacing the solver with a Graph Neural Network (GNN).

---

## 1. Codebase Analysis

### What the Code Does

MeltFlow is a **Ghost Fluid Method (GFM) solver** for multi-phase compressible/incompressible flow. It solves the Euler equations for two-fluid systems in 1D and 2D.

**Applications:**
- Shock tubes with multiple fluids
- Droplet dynamics and oscillations
- Bubble and cavitation dynamics
- Gas-liquid interface interactions

### Core Components

| Component | Files | Purpose |
|-----------|-------|---------|
| Main Solver | `main.m` | Time-stepping loop, orchestrates simulation |
| Ghost Fluid Method | `ghost_GFM.m`, `real_GFM.m` | Handles interface conditions between fluids |
| Roe Solver | `roe_perfect.m`, `roe_flux1D/2D.m` | Compressible flow flux computation |
| Level Set | `advc.m`, `reinit_fast.m` | Tracks and maintains interface position |
| Extrapolation | `extrp.m`, `extrp_vel.m` | Extends fields across interface |

### Numerical Methods

- **Roe's approximate Riemann solver** for compressible flows
- **Level set method** for interface tracking
- **WENO (5th-order)** finite differences for reinitialization
- **Godunov upwind scheme** for advection
- **CFL-based adaptive timestepping**

### Project Structure

```
meltflow-matlab/
├── main.m              # Entry point
├── functions/          # 34 solver functions
├── input/              # Test case configurations
├── visualization/      # Plotting functions
├── exact/              # Exact solutions for validation
├── data/               # Output data files
└── info/               # Documentation
```

---

## 2. Improvements Made

### 2.1 README Created

A comprehensive `README.md` was created covering:
- Project overview and features
- Quick start guide
- Project structure
- Available test cases
- Configuration parameters
- Numerical methods documentation
- References to foundational papers (Fedkiw, Roe)

### 2.2 Debug Flag Added

**Problem:** Figures 99 and 100 were appearing during simulation - leftover debug code in `ghost_GFM.m`.

**Solution:** Added a `flg_dbg` parameter to enable/disable debug figures.

**Files modified:**
- `functions/defaults.m` - Added default `flg_dbg = 0`
- `functions/key_call.m` - Added to parameter vector (position 39)
- `functions/prm_call.m` - Added to parameter unpacking
- `functions/ghost_GFM.m` - Wrapped debug figures with `if (flg_dbg)`
- `info/variables.txt` - Updated parameter documentation

**Usage:** Add `flg_dbg = 1;` to input file to enable debug figures.

### 2.3 Plot Labels and Titles Added

**Problem:** The four subplots in figure 1 lacked titles and proper axis labels.

**Solution:** Updated `functions/plt_setup.m` to add:

| Subplot | Title | Y-axis Label |
|---------|-------|--------------|
| 1 | Temperature | T [K] |
| 2 | Velocity | u [m/s] |
| 3 | Pressure | p [Pa] |
| 4 | Level Set | φ [m] |

**Note:** The first subplot was previously mislabeled as density (ρ) but actually displays temperature calculated via `T = p/(R*ρ)`.

---

## 3. Understanding the Simulation Output

### Default Test Case: 1D Sod Shock Tube (`in_1Dsod1fl`)

A tube filled with gas divided by a diaphragm at x = 0.5:
- **Left side (x < 0.5):** High pressure gas (ρ=1, p=10⁵ Pa)
- **Right side (x > 0.5):** Low pressure gas (ρ=0.125, p=10⁴ Pa)

At t=0, the diaphragm is removed, creating three waves:
1. **Expansion fan** - moves left, smoothly drops pressure
2. **Contact discontinuity** - moves right, separates gases
3. **Shock wave** - moves right fastest, sharp jump in properties

This is a standard CFD validation case with an exact analytical solution.

---

## 4. GNN Replacement Feasibility

### Can the Solver Be Replaced with a Graph Neural Network?

**Short answer:** Yes, with caveats.

### What Would Work Well

| Aspect | Why GNNs Fit |
|--------|--------------|
| Grid structure | Mesh naturally maps to a graph |
| Local physics | Message passing mimics flux propagation |
| Speed | 100-1000x faster inference reported |
| 1D/2D scale | Tractable problem size |

### Key Challenges

1. **Shock waves & discontinuities** - GNNs tend to smear sharp gradients
2. **Conservation laws** - No guarantee of mass/momentum/energy conservation
3. **Multi-phase interface** - Ghost Fluid Method jump conditions are hard to learn
4. **Generalization** - Needs diverse training data

### Possible Approaches

| Approach | Description |
|----------|-------------|
| Full replacement | Train GNN to predict U(t+dt) from U(t) |
| Hybrid | GNN for smooth regions, traditional solver near shocks |
| Neural operator | FNO or DeepONet to learn solution operator |

### Feasibility Assessment

| Use Case | Feasibility |
|----------|-------------|
| Single-fluid Sod shock tube | High |
| Fixed two-fluid problems | Medium |
| General GFM replacement | Hard (active research) |

---

## 5. Spatial Discretization → Graph Mapping

### 1D Grid Structure

```matlab
x = x_min:dx:x_max    →    [0, 0.01, 0.02, ..., 1.0]
                            node 1  node 2  ...  node n
```

**Data at each node:**
```matlab
U(k,i) = [ρ, u, p]   % Primitive variables
W(k,i) = [ρ, ρu, E]  % Conserved variables
phi(i) = level set   % Interface distance
```

### 2D Grid Structure

```
j=n(2)  ●───●───●───●───●
        │   │   │   │   │
j=2     ●───●───●───●───●
        │   │   │   │   │
j=1     ●───●───●───●───●
       i=1 i=2 i=3 ... i=n(1)

U(k,i,j) = [ρ, u, v, p] at node (i,j)
```

### Roe Solver Update (1D)

```
F_{i+1/2} = flux between node i and node i+1

Update: W(i) = W(i) - dt/dx * (F_{i+1/2} - F_{i-1/2})
```

### Direct Graph Mapping

| MATLAB Code | Graph Equivalent |
|-------------|------------------|
| Grid point `i` | Node `i` |
| `U(:,i)` = [ρ,u,p] | Node features |
| `phi(i)` | Additional node feature |
| `x(i)` | Node position encoding |
| Neighbor `i±1` | Edge connections |
| `dx` | Edge feature |
| `roe_flux1D(W_L, W_R)` | Message function |
| `W(i) -= dt/dx*(F_r - F_l)` | Node update function |

### Key Insight

The Roe solver is already doing **local message passing** - it computes physics-based messages (fluxes) between neighbors. A GNN would learn to approximate those flux computations.

```
GNN message passing ≈ Flux computation

message(node_i, node_j) ≈ roe_flux(W_i, W_j)
update(node_i, messages) ≈ W_i - dt/dx * (F_right - F_left)
```

---

## References

- Fedkiw, R. P., et al. (1999). A non-oscillatory Eulerian approach to interfaces in multimaterial flows (the ghost fluid method). *Journal of Computational Physics*, 152(2), 457-492.
- Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43(2), 357-372.
- Sanchez-Gonzalez, A., et al. (2020). Learning to Simulate Complex Physics with Graph Networks. *ICML*.
