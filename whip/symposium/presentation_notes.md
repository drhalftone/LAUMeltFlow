# Presentation Notes

Prepared answers for questions that may come up during the poster session. Study these before presenting.

---

## The Physical System

### What is the simulation modeling?
A whip — modeled as a 1D chain of mass beads connected by spring rod elements. One end is fixed (the "hand"), the rest hang free under gravity. The mass tapers linearly from heavy at the anchor to light at the tip (10:1 ratio), which is what makes real whips crack — energy concentrates as it travels to lighter segments.

### What are the default parameters?
- 16 beads, 15 rods
- Total length: 2.0 m, total mass: 0.5 kg
- Spring stiffness: k = 10,000 N/m
- Damping: c = 0.5 (along rod axis)
- Drag: 0.02 (global velocity drag)
- Timestep: dt = 0.0001 s
- Gravity: 9.81 m/s^2

### What forces act on each bead?
1. **Gravity** — pulls downward, F = -mg
2. **Spring forces** — from left and right rods, Hooke's law: F = k * (stretch) * direction
3. **Damping** — resists relative motion along the rod axis
4. **Drag** — resists absolute motion, F = -drag * velocity

### What is symplectic Euler and why use it?
It's a time integration scheme where you update velocity FIRST, then use the NEW velocity to update position. The ordering matters:
- Regular Euler: v_new = v + dt*a, x_new = x + dt*v (uses OLD v for position)
- Symplectic Euler: v_new = v + dt*a, x_new = x + dt*v_new (uses NEW v for position)

The symplectic version conserves energy over long simulations. Regular Euler gains or loses energy over time, making long simulations drift.

### What is SHAKE?
An algorithm that iteratively corrects bead positions so rod lengths stay exactly at their rest length. Named by Ryckaert, Ciccotti, and Berendsen (1977) — it's not an acronym. We don't use SHAKE in the MPGNN approach — we use stiff springs instead, which keep rods close to their rest length without exact enforcement.

---

## Volume Sampling (Training Data)

### What is volume sampling and why not use trajectory data?
**Trajectory data**: record a simulation running over time. The data traces thin curves through state space. Problem: if the GNN drifts slightly off those curves during rollout, it encounters states it never trained on and predictions degrade.

**Volume sampling**: generate 1 million completely random, independent scenarios. For each: pick a random bead, give it random velocity, place neighbors at random positions near rest length, compute exact physics for one timestep. No simulation is ever run.

This fills the entire reachable state space uniformly, so the GNN has training coverage no matter what state it encounters.

### How are neighbor positions sampled?
Polar coordinates: radius sampled within +/- 5% of rest length, angle sampled uniformly in [0, 2pi). This ensures neighbors are at physically realistic distances (springs slightly stretched or compressed, never absurdly far away).

### What makes the v2 data generator different?
The v2 generator samples from chains of **different lengths** (8 to 32 beads) and **different taper ratios** (1 to 10). This means the mass values and rest lengths in the training data span a wide range, so the model isn't tied to one specific chain configuration.

### What does each training sample look like?
**Input (16 features):**
- Self (4): vel_x, vel_y, mass, is_fixed
- Left neighbor (6): relative_pos_x, relative_pos_y, relative_vel_x, relative_vel_y, neighbor_mass, rest_length
- Right neighbor (6): same format

**Output (4 values):**
- delta_pos_x, delta_pos_y, delta_vel_x, delta_vel_y

---

## The MPGNN Architecture

### What is a GNN?
A neural network that operates on graph-structured data (nodes connected by edges). Each node gathers information from its neighbors, processes it, and updates its own state. This is called "message passing."

### Why is the bead chain a graph?
Nodes = beads (with features like velocity, mass). Edges = rods connecting adjacent beads (with features like relative position, relative velocity). The physics is local — each bead only feels forces from its immediate neighbors — which maps exactly to one round of message passing.

### What was the original BeadGNN and what was wrong with it?
A shared MLP that took a flat 16-feature vector (self + left neighbor + right neighbor concatenated) and predicted the 4-dim state delta. It worked for a fixed 16-bead chain, but:
- The input size was hardcoded for N=16
- The mass distribution in training was specific to 16 beads
- Could not run on chains of other lengths

### What does the MPGNN do differently?
1. **Separate encoders**: node features and edge features go through their own MLPs, so the model knows "this is about the bead" vs "this is about the rod"
2. **Message computation**: each rod produces a learned message via a message MLP
3. **Ghost masking**: endpoint beads (bead 0 has no left, bead N-1 has no right) get their missing-neighbor messages forced to zero — the architecture enforces "no neighbor = no influence"
4. **Structured aggregation**: each bead updates from [self_hidden | left_message | right_message] — the slots are fixed so the model always knows which is which
5. **Shared weights**: same MLPs applied to every bead regardless of chain length

### What are the layers and their sizes?
- Node encoder: 4 -> 64 (ResMLPBlock)
- Edge encoder: 6 -> 64 (ResMLPBlock, shared for left and right)
- Message MLP: 64 -> 64 (ResMLPBlock)
- Node update MLP: 192 -> 64 (takes [self|left_msg|right_msg])
- Output head: 64 -> 4 (Linear)
- Total: 80,324 trainable parameters

### What is K (message passing rounds)?
K is how many times the "share with neighbors and update" cycle repeats. K=1 means each bead sees its immediate neighbors once. K=2 would let information travel 2 hops. We use K=1 because the physics is local (forces only come from immediate neighbors).

### What is ghost masking?
When a bead has no left neighbor (bead 0) or no right neighbor (bead N-1), the message from that missing neighbor is forcibly zeroed out: `m_L = has_left * m_L`. This is different from zero padding where the model has to LEARN that zeros mean "nothing" — ghost masking ENFORCES it structurally.

### Is this a hypergraph neural network?
No. A hypergraph would have single edges connecting 3+ nodes simultaneously. Our graph has only pairwise edges (each rod connects exactly 2 beads). Two edges meeting at a node is not a hyperedge — it's just a node with degree 2. Increasing K doesn't change this; K controls depth of propagation, not width of connectivity.

### Which parameters are trainable?
All the weights and biases in the 5 MLPs (node encoder, edge encoder, message MLP, node update MLP, output head) = 80,324 parameters. The normalization statistics (mean/std for inputs and outputs) are stored as frozen buffers — computed once from training data, never updated by gradient descent.

---

## Training

### How is the model trained?
- MSE loss on normalized state deltas
- AdamW optimizer with weight decay 1e-5
- Cosine annealing learning rate from 1e-3
- Batch size 512, 200 epochs
- 10% validation split
- Gradient clipping at norm 1.0
- Node and edge features normalized separately (each with own mean/std)

### What do the training curves show?
Train and validation loss decrease smoothly and track each other closely — no overfitting. Predicted vs true velocity deltas on held-out samples fall tightly along the diagonal — accurate single-step predictions.

---

## Results and Evaluation

### How is the model evaluated?
Run the trained MPGNN side-by-side with the reference physics solver from identical initial conditions for 20,000+ timesteps. Compare:
- **Position error**: mean distance between GNN bead positions and ground truth positions over time
- **Rod length violation**: how much the rods stretch beyond their rest length
- **Energy drift**: difference in total energy between GNN and ground truth

### Does it generalize to different chain lengths?
Yes. Trained on chains of 8-32 beads. Tested on N = {8, 10, 12, 16, 20, 24, 32}:
- Short/medium chains (N <= 16): track reference closely
- Longer chains (N >= 20): accumulate more drift over time
- Rod errors: ~1e-3 across all lengths

### Why do longer chains have more error?
More beads = more compounding. Each bead makes a small per-step error. With 32 beads, there are more light-mass tip beads that move fast and accumulate errors faster. Also, the light tip beads have the highest velocities, so small force prediction errors translate to larger position errors.

### Does it handle different initial conditions?
Yes. Tested with different starting angles (0, 45, 90 degrees from horizontal) and initial velocities (sideways kicks, downward drops). The model handles these within its training velocity bounds because volume sampling exposed it to those state regions. Pushing velocities far outside training bounds (e.g. vx > 20 m/s) causes divergence, as expected.

---

## Limitations (be honest about these)

### Why does the model drift over long rollouts?
The model is trained on SINGLE-STEP targets — it never sees its own mistakes during training. At inference, its predictions become the next step's input. Any systematic error compounds over thousands of steps. This is the #1 limitation.

**Fix (not implemented):** rollout training — feed the model's own predictions back during training and penalize multi-step error.

### Why aren't the rod lengths exact?
The reference solver uses stiff springs (k=10,000), not exact constraints. The GNN learns to match those spring dynamics, which allow ~0.1% rod stretch. If exact inextensibility were needed, you'd either add a SHAKE post-correction step after each GNN prediction, or add a rod-length penalty to the training loss.

### Why only K=1?
The physics is local (1-hop), so K=1 is sufficient for single-step accuracy. K>1 would require multi-hop training data or end-to-end rollout training to be effective. It's a natural next step but wasn't pursued in this work.

### How fast is it compared to the solver?
We haven't formally benchmarked speed. The GNN replaces the force computation with a forward pass, which on a GPU could be faster for large chains. But for 16 beads, the overhead of PyTorch (tensor creation, GPU transfer) likely makes the GNN slower than the simple NumPy solver. The speed advantage would appear at scale (thousands of beads, batched simulations).

---

## Connections and Context

### How does this relate to the earlier MeltFlow/Euler work?
The earlier report showed that for 1D path graphs with fixed topology, simple concatenation (MLP) is sufficient — no need for message passing. That insight is correct and still holds. We added message passing not because it's "better" on fixed chains, but because it's the only way to handle VARIABLE chain lengths with one model.

### What's the connection to real FEA?
The bead chain is a simple FEA problem: nodes are finite element nodes, rods are 1D spring elements, forces are computed element-by-element. The same GNN approach could apply to more complex FEA meshes (2D triangles, 3D tetrahedra) where the local force computation is more expensive and the speedup from a learned surrogate would be more significant.

### Could this work on 2D/3D meshes?
The architecture generalizes naturally. In 2D, each node would have more neighbors (4 for quads, 3-6 for triangles), and edge features would include 2D/3D relative positions. The message-passing structure stays the same. The main challenge is generating good training data for higher-dimensional problems.

### What would you do next?
1. Rollout training to reduce long-horizon drift
2. Constraint-aware loss or hybrid SHAKE correction
3. K=2 message passing with multi-hop training data
4. Apply the approach to a more complex FEA problem (2D heat transfer, structural mechanics)

---

## Quick Facts for Fast Answers

- **Total parameters**: 80,324
- **Training samples**: 1,000,000
- **Training time**: ~200 epochs
- **Chain lengths trained on**: 8-32 beads
- **Timestep**: 0.0001 seconds
- **Steps per simulated second**: 10,000
- **Architecture name**: BeadMPGNN (Message-Passing Graph Neural Network)
- **Not a hypergraph**: pairwise edges only
- **Not PyTorch Geometric**: all graph ops hand-written
- **Why message passing over MLP**: variable-length generalization
