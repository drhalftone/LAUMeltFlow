# Galerkin Methods Analysis for GNN-Based Fluid Solver

## Overview

This document analyzes Galerkin methods as an alternative numerical framework for solving the compressible Euler equations, with a focus on Discontinuous Galerkin (DG) as the foundation for a Graph Neural Network (GNN) replacement.

**Recommendation: Option B - Implement DG, learn the numerical flux with a GNN.**

---

## What Are Galerkin Methods?

Galerkin methods approximate PDE solutions by projecting onto a finite-dimensional function space.

### Core Idea

1. **Approximate the solution** as a linear combination of basis functions:
   ```
   u(x,t) ≈ Σ uⱼ(t) φⱼ(x)
   ```

2. **Require the residual to be orthogonal** to the test space (weak form):
   ```
   ∫ R(u) · ψᵢ dx = 0   for all test functions ψᵢ
   ```

### Main Variants

| Method | Basis Functions | Characteristics |
|--------|-----------------|-----------------|
| **Finite Element (FEM)** | Piecewise polynomials (local support) | Good for complex geometries |
| **Spectral Galerkin** | Global polynomials, Fourier modes | High accuracy for smooth solutions |
| **Discontinuous Galerkin (DG)** | Piecewise polynomials, discontinuous across elements | Good for hyperbolic PDEs with shocks |
| **Petrov-Galerkin** | Different test and trial spaces | Can add stabilization |

---

## Current Approach: Finite Volume + Roe Flux

The MeltFlow solver uses finite volume method with Roe's approximate Riemann solver for the compressible Euler equations:

```
∂W/∂t + ∂F(W)/∂x = 0

where W = [ρ, ρu, E]ᵀ (conserved variables)
```

Update formula:
```
Wᵢⁿ⁺¹ = Wᵢⁿ - (dt/dx) * [F_{i+1/2} - F_{i-1/2}]
```

**Characteristics:**
- Roe flux approximates the Riemann problem at cell interfaces
- Naturally conservative
- Handles shocks via upwinding
- First-order accurate

---

## Why Discontinuous Galerkin?

### Comparison of Galerkin Methods for Hyperbolic PDEs

**1. Standard Galerkin (Continuous FEM)**
- ❌ **Not suitable** for hyperbolic problems with shocks
- Produces spurious oscillations (Gibbs phenomenon)
- No built-in upwinding

**2. Discontinuous Galerkin (DG)**
- ✅ **Ideal candidate** - designed for hyperbolic conservation laws
- Allows discontinuities at element boundaries (good for shocks)
- Uses numerical fluxes at interfaces (similar to Roe!)
- Higher-order accuracy with compact stencils
- Easily parallelizable

**3. Streamline Upwind Petrov-Galerkin (SUPG)**
- ✅ Adds stabilization for convection-dominated problems
- Still continuous, but handles advection better
- More complex implementation

---

## Discontinuous Galerkin vs. Roe Solver

| Aspect | Roe (Current) | Discontinuous Galerkin |
|--------|---------------|------------------------|
| Order of accuracy | 1st order | Arbitrary (p-refinement) |
| Shock handling | Upwind flux | Numerical flux + limiters |
| Parallelization | Good | Excellent (local operations) |
| Implementation complexity | Moderate | Higher |
| GNN compatibility | Good | **Excellent** |
| Mesh flexibility | Structured only | Unstructured supported |
| hp-adaptivity | No | Yes |

---

## DG Method Structure

### Mathematical Formulation

For conservation law `∂u/∂t + ∇·F(u) = 0`, the DG weak form on element K:

```
∫_K (∂u/∂t) φ dx - ∫_K F(u)·∇φ dx + ∫_∂K F*·n φ ds = 0
```

Where:
- `φ` = test/basis functions (polynomials of degree p)
- `F*` = numerical flux at element interfaces
- `n` = outward normal vector

### Computational Structure

```
For each element K:
  1. Volume integral:  ∫_K F(u)·∇φ dx      (local to element)
  2. Surface flux:     ∫_∂K F*·n φ ds      (neighbor interaction)
  3. Update solution coefficients
```

### Why DG is Naturally Graph-Structured

The DG structure maps perfectly to a graph:

| DG Component | Graph Equivalent |
|--------------|------------------|
| Elements | Nodes |
| Element DOFs | Node features |
| Shared faces | Edges |
| Numerical flux F* | Edge messages |
| Surface integral | Message aggregation |
| Volume integral + update | Node update |

This is **more naturally a graph** than finite volumes because:
- Multiple degrees of freedom per element (richer node features)
- Face-based communication (explicit edge structure)
- Local operations dominate (sparse computation)

---

## Recommended Approach: Option B

### Strategy: Implement DG, Learn the Numerical Flux

**Core idea:** Use DG framework but replace the analytical numerical flux with a learned GNN.

```
Traditional DG:  F* = RoeFlux(u⁻, u⁺, n)  or  LaxFriedrichs(u⁻, u⁺, n)
GNN-DG:          F* = GNN(u⁻, u⁺, n, edge_features)
```

### Architecture

```
Input (per element):
  - Node features: [ρ, ρu, (ρv), E, φ, x, (y)] × (p+1) DOFs
  - Edge features: [face_normal, face_area, dx]
  - Neighbor features: adjacent element states

GNN Layer ≈ One DG timestep:
  1. Edge function:  F* = MLP(u⁻, u⁺, n)           # Learn numerical flux
  2. Aggregation:    ∫_∂K F*·n φ ds                # Sum over faces
  3. Volume term:    ∫_K F(u)·∇φ dx                # Can be learned or analytical
  4. Node update:    u^{n+1} = u^n + dt * (volume + surface terms)
```

### Implementation Steps

1. **Implement basic DG solver in Python/PyTorch**
   - Start with 1D Euler equations
   - Use simple numerical flux (Lax-Friedrichs or Roe)
   - Verify against MeltFlow results

2. **Convert to graph structure**
   - Elements → nodes
   - Faces → edges
   - Use PyTorch Geometric or DGL

3. **Replace numerical flux with GNN**
   - Train on DG simulation data
   - Input: left/right states, normal
   - Output: numerical flux vector

4. **Extend to 2D and multi-fluid**
   - Add level set as node feature
   - Handle ghost fluid interface conditions

### Advantages of This Approach

1. **Higher-order accuracy** - DG naturally supports p-refinement
2. **Better generalization** - learns flux function, not specific solutions
3. **Mesh flexibility** - can extend to unstructured meshes
4. **Physical constraints** - can enforce conservation in architecture
5. **Interpretability** - learned flux can be analyzed

---

## Training Data Generation

Use MeltFlow (or new DG solver) to generate training pairs:

```
Input:  (u⁻, u⁺, n) at each face
Output: F* from Roe solver

Training scenarios:
  - Sod shock tube (various initial conditions)
  - Two-fluid Riemann problems
  - Droplet dynamics
  - Random perturbations
```

### Data Augmentation

- Rotational symmetry (2D)
- Galilean invariance (add constant velocity)
- Scaling (different pressure/density ratios)

---

## Key References

### DG Methods
- Cockburn & Shu (1998) - "The Runge-Kutta Discontinuous Galerkin Method for Conservation Laws"
- Hesthaven & Warburton (2008) - "Nodal Discontinuous Galerkin Methods"

### GNN for Physics
- Sanchez-Gonzalez et al. (2020) - "Learning to Simulate Complex Physics with Graph Networks"
- Pfaff et al. (2021) - "Learning Mesh-Based Simulation with Graph Networks" (MeshGraphNets)
- Brandstetter et al. (2022) - "Message Passing Neural PDE Solvers"

### Neural Operators (related)
- Li et al. (2020) - "Fourier Neural Operator for Parametric PDEs"
- Lu et al. (2021) - "DeepONet: Learning nonlinear operators"

---

## Summary

| Approach | Description | Complexity | GNN Fit |
|----------|-------------|------------|---------|
| Option A | Keep Roe, learn flux | Low | Good |
| **Option B** | **DG framework, learn numerical flux** | **Medium** | **Excellent** |
| Option C | Learn entire Galerkin operator | High | Research-level |

**Option B is recommended** because it:
- Provides a principled numerical framework (DG)
- Has natural graph structure for GNN
- Allows higher-order accuracy
- Keeps the learning task focused (just the flux)
- Has clear path to extensions (unstructured, 3D, multi-physics)
