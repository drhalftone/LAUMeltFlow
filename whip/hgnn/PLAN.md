# Hamiltonian Graph Neural Network for Bead Chain

## Goal

Replace the U-Net GNN (which learns SHAKE corrections) with a Hamiltonian GNN
that learns the system's Hamiltonian directly. The Hamiltonian structure
guarantees energy conservation and constraint satisfaction by construction,
rather than learning them from data.

**Phase 1 (this plan):** Conservative system only — gravity + rigid constraints,
no drag or damping. Validate that the HGNN reproduces the correct dynamics.

**Phase 2 (future):** Add a learned drag/dissipation term on top.

---

## Architecture

### What the network learns

A single scalar: the system Hamiltonian H(q, p).

The equations of motion come for free via autograd:
```
dq/dt =  dH/dp    (positions evolve along momentum gradient)
dp/dt = -dH/dq    (momenta evolve against position gradient)
```

### How the GNN produces H

Following the CHNN (Finzi et al., NeurIPS 2020) and HGNN (Bishnoi et al., 2023)
pattern:

1. **Per-node features:** position q_i, momentum p_i = m_i * v_i
2. **Per-edge features:** relative displacement (q_j - q_i), rest length L0
3. **Message passing:** 2-3 rounds of message passing along the chain graph.
   Each node aggregates neighbor messages and updates its hidden state.
4. **Per-node energy:** Each node's final hidden state is mapped to a scalar
   energy contribution e_i via a small MLP.
5. **Total Hamiltonian:** H = sum(e_i)
6. **Dynamics:** dq/dt, dp/dt obtained by differentiating H w.r.t. q and p.

### Constraint enforcement

Following CHNN, rigid rod constraints are enforced exactly via projection,
not learned:

- Constraint: Phi_e = ||q_j - q_i||^2 - L0^2 = 0  for each rod
- Constraint Jacobian: DPhi computed analytically
- Projection: P = I - J @ DPhi @ (DPhi^T @ J @ DPhi)^{-1} @ DPhi^T
- Constrained dynamics: dz/dt = P @ J @ grad(H)

The network only needs to learn H — the constraints are handled by the
projection operator at each integration step.

### Kinetic energy: known form

Following CHNN, we don't learn kinetic energy from scratch. It has a known
quadratic form:

```
T = sum_i  |p_i|^2 / (2 * m_i)
```

We can either:
- (a) Hardcode T with known masses and only learn V(q) with the GNN
- (b) Learn the masses as parameters (log-parameterized for positivity)

Option (a) is simpler and we know the masses, so start there.

### Potential energy: learned by GNN

The GNN learns V(q) — gravitational + any effective potential from the
constrained dynamics. Each node computes a local energy contribution from
its position and its neighbors' positions.

---

## System: Conservative Bead Chain (no drag)

Reuse the existing simulation infrastructure from `whip/simulation.py`:
- `create_chain()` with drag=0, damping=0
- `step_symplectic_euler()` with use_constraints=True
- `compute_energy()` for ground truth energy

System parameters (match existing training data):
- n_nodes = 16 (power of 2)
- total_length = 2.0 m
- total_mass = 0.5 kg
- taper_ratio = 10.0 (tapered mass)
- gravity = 9.81
- dt = 0.0001
- drag = 0.0, damping = 0.0  (conservative!)

---

## Files to Create

### 1. `whip/hgnn/model.py` — HGNN model

```
class HamiltonianGNN(nn.Module):
    - __init__(n_beads, hidden_dim, n_message_passes, mass)
    - forward(q, p) -> scalar H
    - derivatives(q, p) -> (dq_dt, dp_dt)  # via autograd

class ConstrainedHamiltonianDynamics:
    - __init__(hamiltonian_fn, constraint_fn, masses)
    - forward(t, z) -> dz_dt  # projected dynamics
```

Key details:
- GNN operates on chain graph (not binary tree — simpler)
- Message passing: node MLP + edge MLP, 2-3 rounds
- Output: per-node scalar energy, summed to H
- Kinetic energy T hardcoded with known masses
- Network only learns potential energy V(q)
- LayerNorm + residual connections (learned from GNN improvements)

### 2. `whip/hgnn/data.py` — Data generation

Generate conservative (no drag) trajectories:
- Run simulation with drag=0, damping=0, use_constraints=True
- Save trajectories as (q, p) sequences at regular intervals
- Each training sample: a short trajectory chunk (5-10 steps)
- Need: positions, momenta, and timestamps

### 3. `whip/hgnn/train.py` — Training loop

Following CHNN training approach:
- Loss = MAE between predicted and true trajectories over short rollouts
- Forward: integrate initial (q0, p0) for C steps using learned dynamics
- Compare predicted trajectory against ground truth
- Optimizer: AdamW, lr=3e-3, cosine schedule
- Use torchdiffeq or manual leapfrog for integration

### 4. `whip/hgnn/evaluate.py` — Evaluation

Compare HGNN vs ground truth (SHAKE):
- Long rollout (30k+ steps)
- Metrics: position error, rod length violations, energy drift
- Same evaluation framework as whip/gnn/evaluate.py

### 5. `whip/hgnn/train.bat` — Windows launch script

---

## Training Strategy

### Data generation
- Run 50-100 simulations with different initial conditions
  (random perturbations from horizontal rest position)
- Each simulation: 30k steps at dt=0.0001 (3 seconds of sim time)
- Save every 10 steps -> 3000 frames per simulation
- Chunk into subsequences of length C=5 for training

### Loss function
- Integrate from (q0, p0) for C=5 steps using learned constrained dynamics
- L = mean |predicted_trajectory - true_trajectory|
- This trains the Hamiltonian to produce correct dynamics, not just
  correct energies

### What success looks like
- Energy should be conserved to machine precision (by construction)
- Rod lengths should be exactly maintained (by constraint projection)
- Trajectories should match SHAKE ground truth over long rollouts
- If this works, the architecture is validated and we can add drag in Phase 2

---

## Dependencies

- torch (already in whip/gnn/.venv)
- torchdiffeq (for ODE integration — pip install torchdiffeq)
- numpy, matplotlib (already available)

Can reuse the whip/gnn/.venv environment.

---

## Differences from Current U-Net GNN Approach

| Aspect | U-Net GNN (current) | HGNN (new) |
|--------|---------------------|------------|
| Learns | Correction vectors (dx, dv) | Scalar potential energy V(q) |
| Constraints | Learned from data | Enforced exactly via projection |
| Energy | No guarantee | Conserved by construction |
| Architecture | Binary tree U-Net | Message-passing on chain graph |
| Training target | Single-step corrections | Multi-step trajectory rollouts |
| Drag/damping | Baked into corrections | Separate (Phase 2) |

---

## References

- Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)
- Finzi, Wang, Wilson, "Simplifying Hamiltonian and Lagrangian NNs via
  Explicit Constraints" (NeurIPS 2020) — github.com/mfinzi/constrained-hamiltonian-neural-networks
- Bishnoi et al., "Discovering Symbolic Laws with HGNN" (2023) —
  github.com/M3RG-IITD/HGNN
- Sanchez-Gonzalez et al., "Hamiltonian Graph Networks with ODE Integrators"
  (2019) — DeepMind
