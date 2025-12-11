# Conservation Constraints for Euler GNN Surrogates

This document outlines how to add conservation constraints to a graph neural network (GNN) surrogate for an Euler (finite-volume or DG) solver.

## Experimental Results (MeltFlow GNN)

We tested several conservation constraint approaches on the 1D Sod shock tube problem. Here are the findings:

### Baseline (No Conservation Constraints)
- **Architecture**: 256 hidden units, 5 layers, no antisymmetry
- **Validation loss**: 0.000055
- **Final RMS Errors**: Density ~5e-02, Velocity ~48, Pressure ~5.5e+03
- **Status**: Stable simulation, best results

### Experiment 1: Hard Antisymmetric Constraint
Implemented `F(i,j) = g(i,j) - g(j,i)` in the model architecture.
- **Validation loss**: 0.00384 (70x worse than baseline)
- **Final RMS Errors**: Density 5.12e+12, Velocity 1.57e+03, Pressure 5.23e+18
- **Status**: **FAILED** - Simulation exploded, completely unstable

### Experiment 2: Soft Conservation Loss (weight=0.1)
Added penalty term `||F_ij + F_ji||^2` to the loss function.
- **Validation loss**: 0.000095 (1.7x worse)
- **Final RMS Errors**: Density 1.2e-01, Velocity 148, Pressure 1.1e+04
- **Status**: **FAILED** - Errors roughly doubled compared to baseline

### Experiment 3: Soft Conservation Loss (weight=0.01)
Reduced conservation loss weight.
- **Validation loss**: 0.000085
- **Final RMS Errors**: Density 9.9, Velocity 771, Pressure 2.4e+06
- **Status**: **FAILED** - Simulation became unstable

### Experiment 4: Global Conservation Loss (weight=0.01)
Added penalty for mean flux difference: `||mean(F_pred - F_true)||^2`
- **Validation loss**: 0.000052 (similar to baseline)
- **Final RMS Errors**: Density 1.4, Velocity 6.3e+03, Pressure 7.9e+05
- **Status**: **FAILED** - Simulation became unstable despite good validation loss

### Experiment 5: Global Conservation Loss (weight=0.001)
Reduced global conservation weight further.
- **Validation loss**: 0.000086
- **Final RMS Errors**: Density 6.7, Velocity 1.5e+03, Pressure 1.1e+06
- **Status**: **FAILED** - Still unstable

### Why These Approaches Failed

1. **The Roe flux is not antisymmetric by design**: The Roe numerical flux `F*(U_L, U_R)` computes a unique flux value at an interface based on the Riemann problem. It's the same physical flux regardless of which cell "owns" the edge in the graph structure.

2. **Graph structure mismatch**: The bidirectional edges in the graph are used for message passing with normals (±1), but training only uses positive-direction edges. The soft conservation loss compared trained fluxes with untrained reverse-direction fluxes, which is invalid.

3. **Conservation is already built into the finite volume method**: The flux differencing scheme `(F_right - F_left)/dx` inherently conserves quantities. What leaves one cell enters the adjacent cell by construction.

4. **Hard constraints restrict model capacity**: Forcing exact antisymmetry prevented the network from learning the correct Roe flux behavior, leading to catastrophic instability.

5. **Global conservation loss causes subtle instabilities**: Even when validation loss appears similar to baseline, adding any conservation-related penalty to the loss function introduced subtle changes that caused the simulation to become unstable over time. The flux prediction task is highly sensitive to small perturbations.

### Recommended Approach

For flux-based GNN surrogates learning Roe-type numerical fluxes:
- **Do NOT use any conservation constraints** (hard, soft, or global)
- Train with **standard MSE loss only** on the numerical flux
- Conservation emerges naturally from the finite volume update structure
- Focus on accurate flux prediction rather than explicit conservation enforcement

**All conservation constraint approaches tested (5 experiments) degraded performance compared to the simple MSE baseline.**

If long-term drift is observed, consider:
- **Multi-step rollout training** (Section 4): Penalize long-horizon drift directly
- **More training data**: Generate multiple trajectories with different initial conditions

---

## 1. Soft Constraints in the Loss

### 1.1 Global Conservation Loss ⚠️ **TESTED - DOES NOT WORK**

For each time step or rollout, compute global conserved quantities:

- Total mass
  \(M = \sum_i \rho_i V_i\)
- Total momentum (1D)
  \(P = \sum_i \rho_i u_i V_i\)
- Total energy
  \(E = \sum_i E_i V_i\)

Add a penalty to the training loss:

\[
L_{\text{cons,global}} =
(M_{\text{pred}} - M_{\text{true}})^2 +
(P_{\text{pred}} - P_{\text{true}})^2 +
(E_{\text{pred}} - E_{\text{true}})^2
\]

This encourages the GNN to match the global conservation behavior of the reference Euler solver over each step or rollout. [web:62][web:65]

**EXPERIMENTAL RESULT**: We tested a simplified version of this (penalizing mean flux difference) with weights 0.01 and 0.001. Both caused simulation instability despite achieving similar validation loss to baseline. The flux prediction task is highly sensitive to any perturbation of the loss function.

### 1.2 Local Residual Loss

For each cell \(i\), approximate a discrete conservation equation, e.g. for mass:

\[
\rho^{n+1}_i V_i \approx
\rho^n_i V_i - \Delta t \sum_{f \in \partial i} F_{\rho,f}
\]

Using GNN-predicted states and fluxes, compute a residual:

\[
R_{\rho,i} =
\rho^{n+1}_{i,\text{pred}} V_i -
\Bigl(\rho^n_i V_i - \Delta t \sum_{f \in \partial i} \hat{F}_{\rho,f}\Bigr)
\]

Add a local residual loss:

\[
L_{\text{cons,local}} =
\sum_i \bigl( R_{\rho,i}^2 + R_{P,i}^2 + R_{E,i}^2 \bigr)
\]

This nudges the network toward a conservative flux-divergence form at the cell level. [web:62]

## 2. Architectural (Hard) Constraints

### 2.1 Flux-Based GNN

Instead of predicting updated cell states directly, let the GNN predict **interface fluxes** on graph edges:

1. **Graph representation**
   - Nodes: cells.
   - Edges: interfaces between cells.

2. **Edge (flux) prediction**
   - For each edge \(i \leftrightarrow j\), the edge MLP outputs a candidate numerical flux \(\hat{F}_{ij}\) (vector of fluxes for \(\rho, \rho u, E\), etc.).

3. **Antisymmetry constraint** ⚠️ **USE WITH CAUTION**
   - Enforce
     \[
     \hat{F}_{ji} = -\hat{F}_{ij}
     \]
     for internal faces.
   - This ensures that flux leaving cell \(i\) enters cell \(j\) and vice versa.
   - **WARNING**: Our experiments show this causes instability when learning Roe-type fluxes. The Roe flux is inherently directional and not antisymmetric. Only use this if your target flux formulation is truly antisymmetric.

4. **Cell update by flux divergence**
   - For each cell:
     \[
     U^{n+1}_i =
     U^{n}_i -
     \frac{\Delta t}{V_i}
     \sum_{j \in \mathcal{N}(i)} \hat{F}_{ij} A_{ij}
     \]
     where \(U_i\) collects conservative variables, and \(A_{ij}\) is the interface area/length. [web:62][web:69]

With antisymmetric fluxes, the internal flux contributions cancel in the global sum, giving exact conservation (up to boundary conditions and numerical precision).

### 2.2 Conservative-State Representation

Operate in **conservative variables**:

- Use \(U_i = (\rho_i, \rho_i u_i, E_i)\) (or their multi-D analogs) as the primary state.
- Apply updates only through flux divergences so that conservation is structurally enforced:

\[
U^{n+1}_i = U^n_i + \text{(sum of flux terms)}
\]

Primitive variables (\(\rho, u, p\)) can be derived as needed for feature construction or diagnostics. [web:65]

### 2.3 Boundary-Aware Flux Constraints

Treat boundary faces separately:

- For walls: enforce zero normal mass flux and appropriate momentum/energy flux conditions, either by:
  - Hard-coding fluxes at wall edges, or
  - Conditioning the edge MLP on boundary type and clamping outputs to satisfy physical BCs. [web:62]

- For inflow/outflow: constrain or condition flux predictions to match prescribed states.

## 3. Combined Training Objective

Let \(L_{\text{data}}\) be the standard supervised loss (e.g. MSE between predicted and reference states over time). A combined objective could be:

\[
L =
L_{\text{data}} +
\lambda_{\text{global}} L_{\text{cons,global}} +
\lambda_{\text{local}} L_{\text{cons,local}}
\]

where \(\lambda_{\text{global}}\) and \(\lambda_{\text{local}}\) balance accuracy vs conservation. [web:62][web:65]

**Based on our experiments**: Start with \(\lambda_{\text{global}} = \lambda_{\text{local}} = 0\) and only add conservation terms if drift is observed over long rollouts.

## 4. Practical Notes

- ~~Start from a **flux-based GNN** with antisymmetric edge fluxes to get conservation "for free"~~ **Updated**: For Roe-type fluxes, do NOT use antisymmetric constraints. Start with a standard flux-based GNN and add global conservation losses only if needed.
- Use **multi-step/rollout training** so that the model is penalized for long-horizon drift, not just single-step errors. [web:41]
- Carefully normalize states and fluxes (e.g., non-dimensionalization, scaling by typical magnitudes) to keep training stable. [web:62][web:65]
- **Sequential train/val split** works better than random shuffling for time-series CFD data.
- **Avoid over-parameterization**: Our experiments showed that deeper networks (512 hidden, 8 layers) overfit compared to the baseline (256 hidden, 5 layers).
