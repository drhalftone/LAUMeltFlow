# GNN Surrogate for SHAKE Constraint Projection

U-Net GNN that learns to replace the iterative SHAKE constraint solver with a single forward pass through a binary tree.

## Quick Start

### 1. Generate training data

From the `whip/` directory:

```bash
python generate_data.py --n_samples 100000
```

This generates 100K random bead chain configurations with importance sampling (60% core, 40% extended) and computes the SHAKE correction for each. Data is saved to `data/input/` and `data/target/`.

### 2. Train the model

From the `whip/gnn/` directory:

```bash
python train.py --epochs 200 --hidden_dim 64 --batch_size 64 --lr 1e-3
```

Outputs saved to `gnn/outputs/`:
- `best_model.pt` — best validation checkpoint
- `final_model.pt` — final epoch
- `losses.npz` — loss curves
- `training_results.png` — loss plot + sample prediction vs target

### 3. Evaluate

```bash
python evaluate.py --model_path outputs/best_model.pt --n_steps 30000
```

Runs three simulations side-by-side (SHAKE, GNN, no correction) and saves comparison plots to `outputs/evaluation.png`.

## Architecture

Binary tree U-Net with encode/decode passes and skip connections:

```
ENCODE (up-pass):
  Level 0: bead MLP([self | left | right]) → h    (16 beads, save skip)
  Level 1: MLP([left_child | right_child]) → h    (8 nodes, save skip)
  Level 2: MLP([left_child | right_child]) → h    (4 nodes, save skip)
  Level 3: MLP([left_child | right_child]) → h    (2 nodes, save skip)
  Level 4: MLP([left_child | right_child]) → h    (1 root, bottleneck)

DECODE (down-pass):
  Level 3: MLP([parent_output | skip]) → d         (2 nodes)
  Level 2: MLP([parent_output | skip]) → d         (4 nodes)
  Level 1: MLP([parent_output | skip]) → d         (8 nodes)
  Level 0: MLP([parent_output | skip]) → correction (16 beads, output)
```

All nodes at the same level share MLP weights.

## Training Data Format

Paired `.npy` files matched by numeric suffix:

```
data/
├── input/               # (16, 7) per frame: [pos_x, pos_y, vel_x, vel_y, mass, type, level]
│   ├── frame000000.npy
│   └── ...
├── target/              # (16, 4) per frame: [dx, dy, dvx, dvy] SHAKE corrections
│   ├── frame000000.npy
│   └── ...
└── graph.npz            # shared graph connectivity
```

Data is generated deterministically from a seed, so it does not need to be committed — just regenerate on each machine.
