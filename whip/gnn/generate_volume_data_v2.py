"""Generate volume-sampled per-bead training data for variable-length chains.

Extends generate_volume_data.py to sample from chains of different lengths
(8-32 beads), so the trained GNN generalizes to any chain size.

For each sample:
  1. Pick a random chain length N ~ Uniform(min_beads, max_beads)
  2. Compute tapered masses for that N
  3. Pick a random bead position within the chain
  4. Sample self velocity, neighbor relative positions (polar), neighbor velocities
  5. Compute exact single-step physics (spring + damping + gravity + drag + Euler)

Output format (split for GNN node/edge encoders):
  node_feat:       (N_samples, 4)  [vel_x, vel_y, mass, is_fixed]
  left_edge_feat:  (N_samples, 6)  [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]
  right_edge_feat: (N_samples, 6)  [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]
  has_left:        (N_samples,)    bool
  has_right:       (N_samples,)    bool
  Y:               (N_samples, 4)  [d_pos_x, d_pos_y, d_vel_x, d_vel_y]

Usage:
    python generate_volume_data_v2.py
    python generate_volume_data_v2.py --n_samples 2000000 --output volume_data_v2.npz
"""

import argparse
import numpy as np
import sys
sys.path.insert(0, "..")
from simulation import create_chain

# --- Physics constants (match simulation.py defaults) ---
TOTAL_LENGTH = 2.0
TOTAL_MASS = 0.5
STIFFNESS = 1e4
DAMPING = 0.5
DRAG = 0.02
GRAVITY = 9.81
DT = 1e-4

# --- Volume bounds (from simulation analysis) ---
VEL_X_RANGE = (-13.1, 13.1)
VEL_Y_RANGE = (-7.8, 7.8)
R_MIN_FACTOR = 0.95
R_MAX_FACTOR = 1.05
REL_VEL_X_RANGE = (-26.2, 26.2)
REL_VEL_Y_RANGE = (-15.6, 15.6)

# --- Chain length range ---
MIN_BEADS = 8
MAX_BEADS = 32


def compute_tapered_masses(n_beads, taper_ratio=10.0):
    """Compute per-bead masses with linear taper for a chain of n_beads."""
    t = np.linspace(1.0, 1.0 / taper_ratio, n_beads)
    return t / t.sum() * TOTAL_MASS


def sample_neighbor_polar(rng, n, rest_lengths):
    """Sample neighbor relative positions in polar coordinates.

    Args:
        rest_lengths: (n,) per-sample rest length (varies with chain length)

    Returns:
        (n, 2) relative positions with distance near rest_length
    """
    r = rng.uniform(0, 1, size=n) * (R_MAX_FACTOR - R_MIN_FACTOR) + R_MIN_FACTOR
    r = r * rest_lengths
    theta = rng.uniform(0, 2 * np.pi, size=n)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack([dx, dy])


def sample_rel_vel(rng, n):
    """Sample relative velocities."""
    dvx = rng.uniform(*REL_VEL_X_RANGE, size=n)
    dvy = rng.uniform(*REL_VEL_Y_RANGE, size=n)
    return np.column_stack([dvx, dvy])


def compute_bead_step(vel, mass, is_fixed,
                      left_rel_pos, left_rel_vel, has_left, left_rest_len,
                      right_rel_pos, right_rel_vel, has_right, right_rest_len):
    """Vectorized single-step physics for N samples.

    Identical to generate_volume_data.py — the physics is local and
    independent of chain length.
    """
    N = len(vel)
    force = np.zeros((N, 2))

    # Spring + damping from left neighbor
    dist_l = np.linalg.norm(left_rel_pos, axis=1, keepdims=True)
    safe_l = (dist_l.squeeze() > 1e-7) & has_left
    if safe_l.any():
        dir_l = np.where(safe_l[:, None], left_rel_pos / np.maximum(dist_l, 1e-7), 0)
        stretch_l = dist_l.squeeze() - left_rest_len
        f_spring = (STIFFNESS * stretch_l)[:, None] * dir_l
        v_along_l = np.sum(left_rel_vel * dir_l, axis=1)
        f_damp = (DAMPING * v_along_l)[:, None] * dir_l
        force += np.where(safe_l[:, None], f_spring + f_damp, 0)

    # Spring + damping from right neighbor
    dist_r = np.linalg.norm(right_rel_pos, axis=1, keepdims=True)
    safe_r = (dist_r.squeeze() > 1e-7) & has_right
    if safe_r.any():
        dir_r = np.where(safe_r[:, None], right_rel_pos / np.maximum(dist_r, 1e-7), 0)
        stretch_r = dist_r.squeeze() - right_rest_len
        f_spring = (STIFFNESS * stretch_r)[:, None] * dir_r
        v_along_r = np.sum(right_rel_vel * dir_r, axis=1)
        f_damp = (DAMPING * v_along_r)[:, None] * dir_r
        force += np.where(safe_r[:, None], f_spring + f_damp, 0)

    # Gravity
    force[:, 1] -= mass * GRAVITY

    # Drag
    force -= DRAG * vel

    # Symplectic Euler
    acc = force / mass[:, None]
    vel_new = vel + DT * acc
    delta_pos_raw = DT * vel_new
    delta_vel_raw = vel_new - vel

    # Fixed beads: zero delta
    free = ~is_fixed
    delta_pos = delta_pos_raw * free[:, None]
    delta_vel = delta_vel_raw * free[:, None]

    return delta_pos, delta_vel


def generate(args):
    rng = np.random.default_rng(args.seed)
    N = args.n_samples

    print(f"Generating {N:,} volume-sampled bead training samples...")
    print(f"  Chain lengths: {args.min_beads} to {args.max_beads} beads")
    print(f"  Taper ratios: {args.min_taper} to {args.max_taper}")

    # --- Sample chain configurations ---
    # Random chain length per sample
    chain_lengths = rng.integers(args.min_beads, args.max_beads + 1, size=N)

    # Random taper ratio per sample
    taper_ratios = rng.uniform(args.min_taper, args.max_taper, size=N)

    # Rest length per sample (depends on chain length)
    rest_length_per_sample = TOTAL_LENGTH / (chain_lengths - 1)

    # Random bead index within each chain
    bead_ids = np.array([rng.integers(0, n) for n in chain_lengths])

    # --- Compute per-sample mass from chain config ---
    mass = np.zeros(N)
    left_mass = np.zeros(N)
    right_mass = np.zeros(N)
    is_fixed = bead_ids == 0
    has_left = bead_ids > 0
    has_right = bead_ids < (chain_lengths - 1)

    # Precompute masses for all unique (chain_length, taper_ratio) combos
    # For efficiency, compute per-sample by vectorized taper formula
    for i in range(N):
        n = chain_lengths[i]
        tr = taper_ratios[i]
        t = np.linspace(1.0, 1.0 / tr, n)
        masses_chain = t / t.sum() * TOTAL_MASS
        bid = bead_ids[i]
        mass[i] = masses_chain[bid]
        if bid > 0:
            left_mass[i] = masses_chain[bid - 1]
        if bid < n - 1:
            right_mass[i] = masses_chain[bid + 1]

    # Rest lengths for ghost neighbors
    left_rest_len = np.where(has_left, rest_length_per_sample, 0.0)
    right_rest_len = np.where(has_right, rest_length_per_sample, 0.0)

    # --- Sample velocities and neighbor positions ---
    print("Sampling state space...")
    vel_x = rng.uniform(*VEL_X_RANGE, size=N)
    vel_y = rng.uniform(*VEL_Y_RANGE, size=N)
    vel = np.column_stack([vel_x, vel_y])

    # Left neighbor (polar sampling with per-sample rest length)
    left_rel_pos_rand = sample_neighbor_polar(rng, N, rest_length_per_sample)
    left_rel_vel_rand = sample_rel_vel(rng, N)
    left_rel_pos = np.where(has_left[:, None], left_rel_pos_rand, 0.0)
    left_rel_vel = np.where(has_left[:, None], left_rel_vel_rand, 0.0)

    # Right neighbor
    right_rel_pos_rand = sample_neighbor_polar(rng, N, rest_length_per_sample)
    right_rel_vel_rand = sample_rel_vel(rng, N)
    right_rel_pos = np.where(has_right[:, None], right_rel_pos_rand, 0.0)
    right_rel_vel = np.where(has_right[:, None], right_rel_vel_rand, 0.0)

    # --- Compute physics ---
    print("Computing per-bead physics...")
    delta_pos, delta_vel = compute_bead_step(
        vel, mass, is_fixed,
        left_rel_pos, left_rel_vel, has_left, left_rest_len,
        right_rel_pos, right_rel_vel, has_right, right_rest_len,
    )

    # --- Pack arrays ---
    node_feat = np.column_stack([
        vel, mass, is_fixed.astype(np.float32),
    ]).astype(np.float32)  # (N, 4)

    left_edge_feat = np.column_stack([
        left_rel_pos, left_rel_vel, left_mass, left_rest_len,
    ]).astype(np.float32)  # (N, 6)

    right_edge_feat = np.column_stack([
        right_rel_pos, right_rel_vel, right_mass, right_rest_len,
    ]).astype(np.float32)  # (N, 6)

    Y = np.column_stack([delta_pos, delta_vel]).astype(np.float32)  # (N, 4)

    # --- Save ---
    print(f"Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        node_feat=node_feat,
        left_edge_feat=left_edge_feat,
        right_edge_feat=right_edge_feat,
        has_left=has_left,
        has_right=has_right,
        Y=Y,
        chain_lengths=chain_lengths,
        taper_ratios=taper_ratios,
        bead_ids=bead_ids,
        stiffness=STIFFNESS,
        damping=DAMPING,
        drag=DRAG,
        gravity=GRAVITY,
        dt=DT,
    )

    # --- Summary ---
    print(f"\nNode features:       {node_feat.shape}")
    print(f"Left edge features:  {left_edge_feat.shape}")
    print(f"Right edge features: {right_edge_feat.shape}")
    print(f"Targets:             {Y.shape}")

    print(f"\n{'node feature':<14} {'min':>12} {'max':>12}")
    print("-" * 40)
    node_names = ["vel_x", "vel_y", "mass", "is_fixed"]
    for i, name in enumerate(node_names):
        print(f"{name:<14} {node_feat[:, i].min():>12.4e} {node_feat[:, i].max():>12.4e}")

    print(f"\n{'edge feature':<14} {'min':>12} {'max':>12}")
    print("-" * 40)
    edge_names = ["dpos_x", "dpos_y", "dvel_x", "dvel_y", "nbr_mass", "rest_len"]
    for i, name in enumerate(edge_names):
        vals = np.concatenate([left_edge_feat[:, i], right_edge_feat[:, i]])
        print(f"{name:<14} {vals.min():>12.4e} {vals.max():>12.4e}")

    print(f"\n{'target':<14} {'min':>12} {'max':>12} {'mean':>12}")
    print("-" * 52)
    target_names = ["d_pos_x", "d_pos_y", "d_vel_x", "d_vel_y"]
    for i, name in enumerate(target_names):
        print(f"{name:<14} {Y[:, i].min():>12.4e} {Y[:, i].max():>12.4e} {Y[:, i].mean():>12.4e}")

    print(f"\nChain lengths: {chain_lengths.min()}-{chain_lengths.max()} beads")
    print(f"Rest lengths:  {rest_length_per_sample.min():.4f}-{rest_length_per_sample.max():.4f} m")
    print(f"Mass range:    {mass.min():.4e}-{mass.max():.4e} kg")
    print(f"Endpoint samples (has_left=F): {(~has_left).sum():,} ({(~has_left).mean()*100:.1f}%)")
    print(f"Endpoint samples (has_right=F): {(~has_right).sum():,} ({(~has_right).mean()*100:.1f}%)")

    print(f"\nDone. Saved {N:,} samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate variable-length volume-sampled GNN data")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="volume_data_v2.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_beads", type=int, default=MIN_BEADS)
    parser.add_argument("--max_beads", type=int, default=MAX_BEADS)
    parser.add_argument("--min_taper", type=float, default=1.0)
    parser.add_argument("--max_taper", type=float, default=10.0)
    args = parser.parse_args()
    generate(args)
