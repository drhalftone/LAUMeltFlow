# Conservation Constraints for Euler GNN Surrogates

This document outlines how to add conservation constraints to a graph neural network (GNN) surrogate for an Euler (finite-volume or DG) solver.

## 1. Soft Constraints in the Loss

### 1.1 Global Conservation Loss

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

3. **Antisymmetry constraint**
   - Enforce
     \[
     \hat{F}_{ji} = -\hat{F}_{ij}
     \]
     for internal faces.  
   - This ensures that flux leaving cell \(i\) enters cell \(j\) and vice versa.

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

## 4. Practical Notes

- Start from a **flux-based GNN** with antisymmetric edge fluxes to get conservation “for free,” then add global/local conservation losses for robustness. [web:62][web:69]
- Use **multi-step/rollout training** so that the model is penalized for long-horizon drift, not just single-step errors. [web:41]
- Carefully normalize states and fluxes (e.g., non-dimensionalization, scaling by typical magnitudes) to keep training stable. [web:62][web:65]
