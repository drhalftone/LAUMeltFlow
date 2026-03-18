# GNN Surrogate for Bead Chain Physics

Two approaches to learning bead chain dynamics with a GNN:

## Approach 1: Direct Physics (per-bead MLP, no SHAKE)

Learns the full single-step physics (spring + damping + gravity + drag + symplectic Euler) directly from volume-sampled training data. No constraint projection needed — just stiff springs.

### Architecture

Each bead runs the same MLP with shared weights:

```
 LEFT NEIGHBOR                      SELF                       RIGHT NEIGHBOR
 [dpos_x, dpos_y,              [vel_x, vel_y,                [dpos_x, dpos_y,
  dvel_x, dvel_y,               mass,                         dvel_x, dvel_y,
  mass,                          is_fixed]                     mass,
  rest_len]                        (4)                         rest_len]
     (6)                           |                              (6)
      |                            |                               |
      +------------+---------------+---------------+---------------+
                   |       CONCATENATE (16)        |
                   +---------------+---------------+
                                   |
                            Linear 16 -> H
                            LayerNorm
                            GELU
                            Linear H -> H
                            GELU
                            Linear H -> 4
                                   |
                          [d_pos_x, d_pos_y,
                           d_vel_x, d_vel_y]
                                (4)
                                   |
                    pos_new = self_pos + d_pos
                    vel_new = self_vel + d_vel
```

All features are **relative**: neighbor positions and velocities are expressed
as differences from the current bead's state. Self position is omitted
(implicitly zero in the local frame), making the network translation-invariant.

Ghost beads (missing neighbors at endpoints) have all-zero features:
d_pos=0, d_vel=0, mass=0, rest_len=0.

### Quick Start

#### 1. Generate simulation data (Qt app)

Run `qt/build/release/BeadChain.exe`, uncheck "Constraints", click **Record**.
This captures trajectory data to discover the reachable state space bounds.

#### 2. Generate volume-sampled training data

```bash
python generate_volume_data.py --n_samples 1000000
```

Uniformly samples the full reachable state space (not just trajectory paths),
computes exact per-bead physics for each sample, and saves to `volume_data.npz`.

Volume bounds (from simulation analysis):
- pos_x: [-2.0, 2.0], pos_y: [-2.0, 0.0]
- vel_x: [-13.1, 13.1], vel_y: [-7.8, 7.8]

#### 3. Analyze bounds (optional)

```bash
python analyze_bounds.py ../qt/bead_training.csv
```

Reads the Qt bead training CSV and prints per-feature min/max to verify
or update the volume bounds.

### Training Data Format

Single `volume_data.npz` with:
- `X`: (N, 16) input features per bead sample
- `Y`: (N, 4) target deltas [d_pos_x, d_pos_y, d_vel_x, d_vel_y]
- `bead_ids`: (N,) which bead index (0-15)
- Physics constants: stiffness, damping, drag, gravity, dt, rest_length

Input features (16):

| # | Feature | Description |
|---|---------|-------------|
| 0 | vel_x | Self velocity x |
| 1 | vel_y | Self velocity y |
| 2 | mass | Self mass |
| 3 | is_fixed | 1 if anchor bead, 0 otherwise |
| 4 | l_dpos_x | Left neighbor relative position x |
| 5 | l_dpos_y | Left neighbor relative position y |
| 6 | l_dvel_x | Left neighbor relative velocity x |
| 7 | l_dvel_y | Left neighbor relative velocity y |
| 8 | l_mass | Left neighbor mass (0 for ghost) |
| 9 | l_rest | Left rod rest length (0 for ghost) |
| 10 | r_dpos_x | Right neighbor relative position x |
| 11 | r_dpos_y | Right neighbor relative position y |
| 12 | r_dvel_x | Right neighbor relative velocity x |
| 13 | r_dvel_y | Right neighbor relative velocity y |
| 14 | r_mass | Right neighbor mass (0 for ghost) |
| 15 | r_rest | Right rod rest length (0 for ghost) |

---

## Approach 2: U-Net GNN (SHAKE corrections)

Learns to replace the iterative SHAKE constraint solver with a single forward
pass through a binary tree. See the encode/decode architecture below.

### Quick Start

#### 1. Generate training data

From the `whip/` directory:

```bash
python generate_data.py --n_samples 100000
```

#### 2. Train

```bash
python train.py --epochs 200 --hidden_dim 64 --batch_size 64 --lr 1e-3
```

#### 3. Evaluate

```bash
python evaluate.py --model_path outputs/best_model.pt --n_steps 30000
```

### Architecture

Binary tree U-Net with encode/decode passes and skip connections:

```
ENCODE (up-pass):
  Level 0: bead MLP([self | left | right]) -> h    (16 beads, save skip)
  Level 1: MLP([left_child | right_child]) -> h    (8 nodes, save skip)
  Level 2: MLP([left_child | right_child]) -> h    (4 nodes, save skip)
  Level 3: MLP([left_child | right_child]) -> h    (2 nodes, save skip)
  Level 4: MLP([left_child | right_child]) -> h    (1 root, bottleneck)

DECODE (down-pass):
  Level 3: MLP([parent_output | skip]) -> d         (2 nodes)
  Level 2: MLP([parent_output | skip]) -> d         (4 nodes)
  Level 1: MLP([parent_output | skip]) -> d         (8 nodes)
  Level 0: MLP([parent_output | skip]) -> correction (16 beads, output)
```

All nodes at the same level share MLP weights.

### Training Data Format

Paired `.npy` files matched by numeric suffix:

```
data/
  input/               # (16, 7) per frame: [pos_x, pos_y, vel_x, vel_y, mass, type, level]
    frame000000.npy
    ...
  target/              # (16, 4) per frame: [dx, dy, dvx, dvy] SHAKE corrections
    frame000000.npy
    ...
  graph.npz            # shared graph connectivity
```
