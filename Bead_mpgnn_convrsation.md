# Bead MPGNN Conversation Summary

This document summarizes the conversation between Grey Goodwin and Claude about the LAUMeltFlow whip simulation project, focused on building a message-passing graph neural network (MPGNN) surrogate for a finite element bead chain.

## Part 1: Understanding the Whip Project

### Project overview
The `whip/` folder is a physics simulation + ML surrogate pipeline for an elastic bead chain (a whip). The goal is to simulate the chain using finite element methods, then train Graph Neural Networks (GNNs) to act as fast surrogates that can replace the expensive physics solver.

### The physics
- 16 mass beads connected by spring rod elements
- Tapered mass (base 10x heavier than tip) — this is what makes a whip crack
- Forces: gravity, Hooke's law springs, viscous damping, velocity drag
- Symplectic Euler integration with `dt = 0.0001s`
- Optional SHAKE constraint projection for rod inextensibility

### SHAKE algorithm
Not an acronym — named by Ryckaert, Ciccotti, Berendsen (1977). The algorithm "shakes" particles back onto the constraint surface after integration:
1. For each rod, compute distance error from rest length
2. Nudge both beads along the rod axis, weighted by inverse mass
3. Iterate until all rods converge
4. Correct velocities to remove radial components

### Three GNN approaches in the project
1. **Per-bead GNN** (bead_model.py) — shared MLP, learns single-step physics
2. **U-Net GNN** (gnn/model.py) — binary-tree hierarchical, learns SHAKE corrections
3. **Hamiltonian GNN** (hgnn/model.py) — learns potential energy V(q), gets dynamics via autograd

## Part 2: Volume Sampling Strategy

### The problem
Trajectory data (recording simulations over time) traces thin curves through state space. The GNN, when it drifts slightly off those paths during rollout, sees states it never trained on and makes bad predictions that compound.

### The solution
Volume sampling uniformly scatters training samples across the full reachable state space. Each sample is one bead in isolation for one timestep — no time evolution between samples, just "given this random state, what happens next?"

### What's in each training sample
**Input (16 features)**:
- Self: `[vel_x, vel_y, mass, is_fixed]` (4)
- Left neighbor: `[dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]` (6)
- Right neighbor: `[dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]` (6)

**Output (4 values)**: `[d_pos_x, d_pos_y, d_vel_x, d_vel_y]`

### Key insight: polar coordinate sampling
Neighbor positions are sampled in polar coordinates with radius near rest length and random angle, ensuring physically realistic neighbor distances.

## Part 3: The Original BeadGNN (was not really a GNN)

The original `BeadGNN` was a **shared MLP**, not a true GNN. It took a flat 16-feature vector and ran it through `nn.Linear` layers. All 16 features were mixed together from the first layer — no notion of "this is self" vs "this is a neighbor" except through the optimizer.

### Limitations
1. Input size was hardcoded for 16-bead chains
2. Could not run on chains of different lengths
3. No structural separation of node vs edge features
4. No weight sharing at the GNN level — just a single MLP applied once per bead

## Part 4: Building the Message-Passing GNN

### Design decision: K=1 message passing
With K=1 round of message passing:
- Each bead's receptive field = immediate neighbors (1 hop)
- Matches the physics: spring forces are local
- Per-bead volume sampling is still valid
- No need to switch to trajectory data

### New architecture: BeadMPGNN

```
Node features (4): [vel_x, vel_y, mass, is_fixed]
Edge features (6): [dpos_x, dpos_y, dvel_x, dvel_y, nbr_mass, rest_len]

NodeEncoder:    ResMLPBlock(4 → 64)
EdgeEncoder:    ResMLPBlock(6 → 64)
MessageMLP:     ResMLPBlock(64 → 64)
NodeUpdateMLP:  ResMLPBlock(192 → 64)   # [self_h | left_msg | right_msg]
OutputHead:     Linear(64 → 4)

Total: 80,324 trainable parameters
```

### Key components
1. **Separate encoders** for nodes and edges — each learns type-appropriate representations
2. **Message computation** per edge via dedicated MLP
3. **Ghost masking**: `torch.where(has_left, msg, zeros)` — structurally enforces "no neighbor → no message"
4. **Structured aggregation**: `[self | left_msg | right_msg]` concatenation preserves directionality
5. **Normalization stats as buffers**: stored on model for self-contained inference

### Weight sharing makes it work on any chain length
All MLPs (node encoder, edge encoder, message MLP, update MLP) are applied per-bead with shared weights. A 32-bead chain uses the same 80K parameters as an 8-bead chain.

## Part 5: Files Created

### New files in `whip/gnn/`
1. **bead_mpgnn.py** — the MPGNN model
2. **generate_volume_data_v2.py** — variable-length data generator (8-32 beads, random taper ratios)
3. **bead_train_mpgnn.py** — training script with separate node/edge normalization
4. **bead_eval_variable.py** — multi-length generalization evaluation
5. **compare_models.py** — head-to-head MLP vs MPGNN comparison

### Modified files
- **bead_visualize.py** — added `--model_type`, `--n_beads`, `--init_angle`, `--init_vx`, `--init_vy` flags

## Part 6: Training Results

### Training setup
- 1M volume-sampled data points from chains of 8-32 beads
- 200 epochs, AdamW + cosine LR, batch size 512, MSE loss
- Separate normalization for node features and edge features

### Results
- Loss converged smoothly, train and val tracked closely (no overfitting)
- Predicted velocity deltas fell tight on the diagonal of true vs predicted
- Generalization test (bead_eval_variable.py):
  - N=8, 10, 12: excellent accuracy (green bars)
  - N=16: good
  - N=20, 24, 32: error grows with length (longer chains = more error accumulation)
- Constraint satisfaction: rod errors ~1e-3 to 6e-3 (modest)

## Part 7: Key Concepts Clarified

### GNN vs Hypergraph NN
- **Pairwise edge** connects exactly 2 nodes (what we have)
- **Hyperedge** connects 3 or more nodes as a single unit (what we don't have)
- Two beads connected by two pairwise edges (bead 5 with edges to 4 and 6) is NOT a hypergraph
- Increasing K (message passing rounds) does NOT make it a hypergraph — K controls depth of propagation, hypergraph controls width of connectivity
- Both regular GNNs and HGNNs use shared MLPs, not edge-specific ones

### Path graphs don't strictly need message passing
A previous report (gnn_report.tex, about the MeltFlow Euler/Sod shock tube project) showed that for 1D path graphs with fixed topology, concatenated MLPs work fine. The reason we added message passing wasn't because it was "better" on fixed chains — it was to enable **variable-length generalization**, which the MLP architecture cannot support.

### Ghost masking vs zero padding
- **Zero padding**: fills missing slots with zeros; model must learn "zeros mean nothing"
- **Ghost masking**: structurally zeros the message after computation, enforcing "no neighbor → no contribution" in the architecture itself

## Part 8: Equations of the MPGNN

### ResMLPBlock (building block)
```
x' = W_proj @ x + b_proj                 (projection if needed)
y = x' + W2 @ GELU(W1 @ LN(x') + b1) + b2  (residual + MLP)
```

### Full forward pass for one bead
1. Normalize: `x̃ = (x - μ) / σ` for node and edge features
2. Encode: `h_n = NodeEncoder(x̃_n)`, `h_eL = EdgeEncoder(x̃_eL)`, `h_eR = EdgeEncoder(x̃_eR)`
3. Compute messages: `m_L = MessageMLP(h_eL)`, `m_R = MessageMLP(h_eR)`
4. Ghost mask: `m_L ← 1_has_left · m_L`, `m_R ← 1_has_right · m_R`
5. Update: `h_n^(1) = NodeUpdateMLP([h_n | m_L | m_R])`
6. Output: `Δỹ = W_out · LN(h_n^(1)) + b_out`
7. Denormalize: `Δy = Δỹ · σ_y + μ_y`

## Part 9: The Email to ME Faculty

Subject: Whip simulation: GNN surrogate for a finite element bead chain

Dr. Lau and I have been exploring whether a graph neural network can act as a fast surrogate for a finite element bead chain simulation. The setup is a 1D chain of mass nodes connected by spring rod elements with tapered mass, modeling a whip, integrated with a standard explicit time stepping scheme.

Our first attempt was a shared multilayer perceptron that took each bead's state along with its two neighbors' relative features as a flat input vector. It worked well for a fixed 16 bead chain but could not run on chains of any other length. To address this, we restructured the network as a true message passing graph neural network with separate encoders for nodes and edges, learned messages computed per rod, and structured aggregation at each bead. Because all the weights are shared across nodes, the same trained model now runs on chains of any length.

Instead of replacing the entire FEA solver, the GNN learns the local per node update by treating each bead as a graph node and each rod as an edge. We tested this on chain configurations the model never saw during training, with promising generalization across both length and initial conditions.

A short animation showing the GNN surrogate and the reference Python simulation evolving side by side is on YouTube here: [YOUTUBE LINK]. The code is available at [GITHUB REPO LINK]. Happy to discuss if anyone is interested.

## Part 10: Next Steps Discussed

### Potential improvements to the MPGNN (in order of expected impact)
1. **Rollout training** — feed the model's own predictions back as input during training, so it learns to be robust to its own errors. Highest leverage for long-rollout drift.
2. **Three-population data sampling** — idea from an unpulled commit by the creator: combine full-volume + near-zero + trajectory samples for better coverage
3. **More training data** — 5M samples instead of 1M
4. **K=2 message passing** — 2-hop receptive field, but needs multi-hop data to work properly
5. **Constraint satisfaction loss** — penalize predictions that stretch rods
6. **Hybrid GNN + SHAKE** — let the GNN predict the unconstrained step, run SHAKE iterations to clean up
7. **Bigger model** — `hidden_dim=128` instead of 64
8. **Try the Hamiltonian GNN** — physics-informed, energy conservation by construction

### Video for faculty
- Record the matplotlib animation from `bead_visualize.py --model_path outputs/mpgnn_best.pt --model_type mpgnn`
- Upload to YouTube as unlisted
- ~1 minute with voiceover describing the FEA setup, the GNN's role, and generalization
- Title: "Finite Element Whip Simulation vs Graph Neural Network Surrogate"

## Part 11: Architecture Diagram Summary

```
Raw features
    ↓ normalize
Node features [vel, mass, is_fixed]     Edge features [dpos, dvel, nbr_mass, rest_len]
    ↓ NodeEncoder                          ↓ EdgeEncoder (shared for L and R)
Hidden node state                       Hidden edge encodings (left, right)
                                           ↓ MessageMLP (shared)
                                        Messages (left, right)
                                           ↓ ghost mask
                                        Masked messages
    ↓ concatenate all three [self_h | left_msg | right_msg]
    ↓ NodeUpdateMLP
Updated hidden state
    ↓ OutputHead
Normalized deltas [dx, dy, dvx, dvy]
    ↓ denormalize
Physical deltas → add to current state
```
