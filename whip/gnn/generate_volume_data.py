"""Generate volume-sampled per-bead training data for the GNN.

Samples the full reachable state space uniformly, computes exact
single-step physics (spring + damping + gravity + drag + symplectic Euler)
for each sample, and saves input/target pairs.

Uses relative features: neighbor positions and velocities are expressed
relative to the current bead (translation invariant). Self position is
omitted (implicitly zero in the local frame).

Input features (16):
  Self:  [vel_x, vel_y, mass, is_fixed]                          (4)
  Left:  [Δpos_x, Δpos_y, Δvel_x, Δvel_y, mass, rest_len]       (6)
  Right: [Δpos_x, Δpos_y, Δvel_x, Δvel_y, mass, rest_len]       (6)

Target (4): [Δpos_x, Δpos_y, Δvel_x, Δvel_y] (residual)

No SHAKE constraints — just stiff springs.

Usage:
    python generate_volume_data.py
    python generate_volume_data.py --n_samples 2000000 --output volume_data.npz
"""

import argparse
import numpy as np

# --- Physics constants (must match Qt app / simulation.py) ---
N_BEADS = 16
TOTAL_LENGTH = 2.0
TOTAL_MASS = 0.5
REST_LENGTH = TOTAL_LENGTH / (N_BEADS - 1)  # ~0.1333
STIFFNESS = 1e4
DAMPING = 0.5
DRAG = 0.02
GRAVITY = 9.81
DT = 1e-4
TAPER_RATIO = 10.0

# --- Volume bounds (from simulation analysis, with symmetric velocities) ---
POS_X_RANGE = (-2.0, 2.0)
POS_Y_RANGE = (-2.0, 0.0)
VEL_X_RANGE = (-13.1, 13.1)
VEL_Y_RANGE = (-7.8, 7.8)


def compute_tapered_masses():
    """Compute per-bead masses with linear taper (matches simulation.py)."""
    t = np.linspace(1.0, 1.0 / TAPER_RATIO, N_BEADS)
    return t / t.sum() * TOTAL_MASS


def sample_pos_vel(rng, n):
    """Sample random positions and velocities from the volume bounds."""
    pos_x = rng.uniform(*POS_X_RANGE, size=n)
    pos_y = rng.uniform(*POS_Y_RANGE, size=n)
    vel_x = rng.uniform(*VEL_X_RANGE, size=n)
    vel_y = rng.uniform(*VEL_Y_RANGE, size=n)
    return np.column_stack([pos_x, pos_y]), np.column_stack([vel_x, vel_y])


def compute_bead_step(pos, vel, mass, is_fixed,
                      left_pos, left_vel, has_left, left_rest_len,
                      right_pos, right_vel, has_right, right_rest_len):
    """Vectorized single-step physics for N samples.

    Replicates Bead.cpp: onNeighborState (spring + damping) for each
    neighbor, then applyGravity (gravity + drag), then integrate
    (symplectic Euler). Fixed beads produce zero deltas.

    Returns:
        delta_pos: (N, 2)
        delta_vel: (N, 2)
    """
    N = len(pos)
    force = np.zeros((N, 2))

    # --- Spring + damping from left neighbor ---
    delta_l = left_pos - pos
    dist_l = np.linalg.norm(delta_l, axis=1, keepdims=True)
    safe_l = (dist_l.squeeze() > 1e-7) & has_left
    if safe_l.any():
        dir_l = np.where(safe_l[:, None], delta_l / np.maximum(dist_l, 1e-7), 0)
        stretch_l = dist_l.squeeze() - left_rest_len
        f_spring = (STIFFNESS * stretch_l)[:, None] * dir_l
        rel_vel_l = left_vel - vel
        v_along_l = np.sum(rel_vel_l * dir_l, axis=1)
        f_damp = (DAMPING * v_along_l)[:, None] * dir_l
        force += np.where(safe_l[:, None], f_spring + f_damp, 0)

    # --- Spring + damping from right neighbor ---
    delta_r = right_pos - pos
    dist_r = np.linalg.norm(delta_r, axis=1, keepdims=True)
    safe_r = (dist_r.squeeze() > 1e-7) & has_right
    if safe_r.any():
        dir_r = np.where(safe_r[:, None], delta_r / np.maximum(dist_r, 1e-7), 0)
        stretch_r = dist_r.squeeze() - right_rest_len
        f_spring = (STIFFNESS * stretch_r)[:, None] * dir_r
        rel_vel_r = right_vel - vel
        v_along_r = np.sum(rel_vel_r * dir_r, axis=1)
        f_damp = (DAMPING * v_along_r)[:, None] * dir_r
        force += np.where(safe_r[:, None], f_spring + f_damp, 0)

    # --- Gravity ---
    force[:, 1] -= mass * GRAVITY

    # --- Drag ---
    force -= DRAG * vel

    # --- Symplectic Euler ---
    acc = force / mass[:, None]
    vel_after = vel + DT * acc
    pos_after = pos + DT * vel_after

    # Fixed beads: zero delta
    free = ~is_fixed
    delta_pos = (pos_after - pos) * free[:, None]
    delta_vel = (vel_after - vel) * free[:, None]

    return delta_pos, delta_vel


def generate(args):
    rng = np.random.default_rng(args.seed)
    masses = compute_tapered_masses()

    print(f"Generating {args.n_samples:,} volume-sampled bead training samples...")
    print(f"  Masses: {masses}")
    print(f"  Bounds: pos_x={POS_X_RANGE}, pos_y={POS_Y_RANGE}")
    print(f"          vel_x={VEL_X_RANGE}, vel_y={VEL_Y_RANGE}")

    N = args.n_samples

    # Sample random bead indices (0-15)
    bead_ids = rng.integers(0, N_BEADS, size=N)

    # Look up per-bead properties
    mass = masses[bead_ids]
    is_fixed = bead_ids == 0
    has_left = bead_ids > 0
    has_right = bead_ids < (N_BEADS - 1)

    # Neighbor masses (0 for ghost)
    left_mass = np.where(has_left, masses[np.clip(bead_ids - 1, 0, N_BEADS - 1)], 0.0)
    right_mass = np.where(has_right, masses[np.clip(bead_ids + 1, 0, N_BEADS - 1)], 0.0)

    # Rest lengths (0 for ghost)
    left_rest_len = np.where(has_left, REST_LENGTH, 0.0)
    right_rest_len = np.where(has_right, REST_LENGTH, 0.0)

    # Sample self state (absolute)
    pos, vel = sample_pos_vel(rng, N)

    # Sample left neighbor state (ghost beads get self_pos, zero vel)
    left_pos_rand, left_vel_rand = sample_pos_vel(rng, N)
    left_pos = np.where(has_left[:, None], left_pos_rand, pos)
    left_vel = np.where(has_left[:, None], left_vel_rand, 0.0)

    # Sample right neighbor state
    right_pos_rand, right_vel_rand = sample_pos_vel(rng, N)
    right_pos = np.where(has_right[:, None], right_pos_rand, pos)
    right_vel = np.where(has_right[:, None], right_vel_rand, 0.0)

    # Compute physics (uses absolute coords internally)
    print("Computing per-bead physics...")
    delta_pos, delta_vel = compute_bead_step(
        pos, vel, mass, is_fixed,
        left_pos, left_vel, has_left, left_rest_len,
        right_pos, right_vel, has_right, right_rest_len,
    )

    # Build input array with RELATIVE features:
    #   Self:  [vel_x, vel_y, mass, is_fixed]                     (4)
    #   Left:  [Δpos_x, Δpos_y, Δvel_x, Δvel_y, mass, rest_len]  (6)
    #   Right: [Δpos_x, Δpos_y, Δvel_x, Δvel_y, mass, rest_len]  (6)
    # Total: 16 features
    left_rel_pos = left_pos - pos    # (N, 2) relative position
    left_rel_vel = left_vel - vel    # (N, 2) relative velocity
    right_rel_pos = right_pos - pos
    right_rel_vel = right_vel - vel

    X = np.column_stack([
        vel, mass, is_fixed.astype(np.float32),
        left_rel_pos, left_rel_vel, left_mass, left_rest_len,
        right_rel_pos, right_rel_vel, right_mass, right_rest_len,
    ]).astype(np.float32)

    Y = np.column_stack([delta_pos, delta_vel]).astype(np.float32)

    # Save
    print(f"Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        X=X,
        Y=Y,
        bead_ids=bead_ids,
        masses=masses,
        stiffness=STIFFNESS,
        damping=DAMPING,
        drag=DRAG,
        gravity=GRAVITY,
        dt=DT,
        rest_length=REST_LENGTH,
    )

    # Summary
    print(f"\nInput X: {X.shape}  Target Y: {Y.shape}")
    print(f"\n{'feature':<14} {'min':>12} {'max':>12}")
    print("-" * 40)
    input_names = [
        "vel_x", "vel_y", "mass", "is_fixed",
        "l_dpos_x", "l_dpos_y", "l_dvel_x", "l_dvel_y", "l_mass", "l_rest",
        "r_dpos_x", "r_dpos_y", "r_dvel_x", "r_dvel_y", "r_mass", "r_rest",
    ]
    for i, name in enumerate(input_names):
        print(f"{name:<14} {X[:, i].min():>12.4e} {X[:, i].max():>12.4e}")

    print(f"\n{'target':<14} {'min':>12} {'max':>12} {'mean':>12}")
    print("-" * 52)
    target_names = ["d_pos_x", "d_pos_y", "d_vel_x", "d_vel_y"]
    for i, name in enumerate(target_names):
        print(f"{name:<14} {Y[:, i].min():>12.4e} {Y[:, i].max():>12.4e} {Y[:, i].mean():>12.4e}")

    print(f"\nDone. Saved {args.n_samples:,} samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate volume-sampled GNN training data")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="volume_data.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args)
