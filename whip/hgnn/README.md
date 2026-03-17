# Hamiltonian Graph Neural Network (HGNN) for Bead Chain

Learns the dynamics of a constrained bead chain by learning its Hamiltonian,
rather than learning correction vectors like the U-Net GNN in `whip/gnn/`.

## Architecture

```
                    ┌─────────────────────────────┐
                    │     H(q, p) = T(p) + V(q)   │
                    └──────┬──────────────┬────────┘
                           │              │
                    ┌──────┴──────┐ ┌─────┴──────┐
                    │  T (known)  │ │ V (learned) │
                    │ |p|²/(2m)   │ │   via GNN   │
                    └─────────────┘ └─────────────┘
```

**Kinetic energy T** is hardcoded: `T = sum_i |p_i|^2 / (2 * m_i)` using known
masses. Not learned.

**Potential energy V** is learned by a message-passing GNN (`PotentialEnergyGNN`):

1. Each node gets features: `[position (2) | mass (1) | is_fixed (1)]`
2. Each edge gets features: `[relative_position (2) | distance (1) | rest_length (1)]`
3. **3 rounds of message passing**, where each node explicitly concatenates
   `[self | left_message | right_message]` — preserving directional information
   along the chain (same pattern as the U-Net GNN).
4. Per-node energy readout: `h_node -> scalar e_i`
5. Total: `V = sum(e_i)`

**Equations of motion** via autograd:
```
dq/dt =  dH/dp    (Hamilton's first equation)
dp/dt = -dH/dq    (Hamilton's second equation)
```

**Constraint enforcement** via Lagrange multiplier projection (not learned):
- Each rod has constraint `Phi = ||q_j - q_i||^2 - L0^2 = 0`
- At each integration step, the unconstrained dynamics are projected onto
  the constraint manifold so rod lengths are maintained exactly
- This is the differentiable equivalent of SHAKE

**Integration** uses leapfrog (Stormer-Verlet), a symplectic integrator that
preserves the Hamiltonian structure.

### Graph structure

```
ghost ← [bead 0] — [bead 1] — [bead 2] — ... — [bead 14] — [bead 15] → ghost
          fixed=1   fixed=0    fixed=0              fixed=0     fixed=0
```

**16 real GNN nodes** (beads 0–15), each with features `[q, mass, is_fixed]`.

- **Bead 0** is the fixed anchor (`is_fixed=1`). The GNN sees this flag and
  learns it represents an immovable boundary condition.
- **Bead 0's left** and **bead 15's right** have no physical neighbor. Instead
  of zero-masking, these endpoints receive messages from **ghost beads** —
  synthetic neighbors with mass=0, position equal to the endpoint itself
  (so relative position=0, distance=0, rest length=0). This gives the message
  MLP physically meaningful "nothing here" features rather than arbitrary zeros.
- **Ghost beads are not GNN nodes.** They only exist as message sources for the
  two endpoints. Every node update always sees `[self | left_msg | right_msg]`
  with no special cases or masking.

### Building blocks

All MLPs use `ResMLPBlock`: `x + MLP(LayerNorm(x))` with GELU activations
and a linear projection if the input dimension doesn't match. This gives
residual connections and normalization at every level.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `PotentialEnergyGNN`, `HamiltonianGNN`, `ConstrainedDynamics`, `leapfrog_step`, `integrate_trajectory` |
| `data.py` | Generate conservative trajectories (drag=0, damping=0) from `whip/simulation.py` |
| `train.py` | Train on short trajectory rollouts (MAE loss on predicted vs true trajectories) |
| `architecture.png` | Architecture diagram |
| `PLAN.md` | Full architecture plan and rationale |

## How to run

### 1. Generate training data

```
generate_data.bat
```

Runs 50 simulations with random initial velocity perturbations, no drag or
damping, with SHAKE constraints. Saves chunked (q, p) trajectories to
`data/trajectories.npz`.

Data is completely separate from `whip/gnn/data/` — that data has drag and
damping baked in and cannot be used here.

### 2. Train

```
train.bat
```

Defaults: hidden_dim=64, 3 message-passing rounds, 200 epochs, AdamW lr=3e-3.

Override with e.g.: `train.bat --hidden_dim 128 --epochs 500`

## Training details

- **Loss:** MAE between predicted and ground-truth trajectories over short
  rollouts (5 frames). Both position and momentum errors contribute, with
  momentum weighted by 0.1x.
- **Optimizer:** AdamW with weight_decay=1e-5, cosine LR schedule
- **Gradient clipping:** max norm 1.0
- **Integration:** Each training step integrates 4 leapfrog steps forward,
  each requiring 3 Hamiltonian evaluations (leapfrog half-step pattern).
  Gradients flow back through the full integration chain.

## Comparison with U-Net GNN (`whip/gnn/`)

| | U-Net GNN | HGNN |
|---|---|---|
| **Learns** | Correction vectors (dx, dy, dvx, dvy) | Scalar potential energy V(q) |
| **Constraints** | Learned implicitly from data | Enforced exactly via projection |
| **Energy** | No conservation guarantee | Conserved by construction |
| **Architecture** | Binary tree with U-Net skip connections | Message passing on chain graph |
| **Training target** | Single-step SHAKE corrections | Multi-step trajectory rollouts |
| **Dissipation** | Baked into training data | Not supported (Phase 1) |
| **Message aggregation** | Explicit [self \| left \| right] concat | Explicit [self \| left \| right] concat |
| **Missing neighbors** | Zero-masked | Ghost beads (mass=0, same position) |

## Phase 2 (future)

Add a separate learned dissipation term for drag. The integration step becomes:
1. HGNN computes conservative dynamics (gravity + constraints)
2. Small drag network predicts velocity correction from current state
3. Combine before advancing to next timestep

## References

- Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)
- Finzi et al., "Simplifying Hamiltonian and Lagrangian NNs via Explicit
  Constraints" (NeurIPS 2020) — constraint projection approach
- Bishnoi et al., "Discovering Symbolic Laws with HGNN" (2023) — GNN + Hamiltonian
- Sanchez-Gonzalez et al., "Hamiltonian Graph Networks with ODE Integrators" (2019)
