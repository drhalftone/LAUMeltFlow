"""Generate volume-sampled per-bead training data for the GNN.

Samples the full reachable state space uniformly, computes exact
single-step physics (spring + damping + gravity + drag + symplectic Euler)
for each sample, and saves input/target pairs.

Uses relative features: neighbor positions are sampled in polar coordinates
(r, theta) around the bead, with r near the rest length. This ensures
physically realistic neighbor distances. Neighbor velocities are expressed
relative to the current bead.

Input features (16):
  Self:  [vel_x, vel_y, mass, is_fixed]                          (4)
  Left:  [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]       (6)
  Right: [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]       (6)

Target (4): [d_pos_x, d_pos_y, d_vel_x, d_vel_y] (residual)

No SHAKE constraints -- just stiff springs.

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

# --- Volume bounds (from simulation analysis) ---
# Self velocity bounds (symmetric, from Qt trajectory analysis)
VEL_X_RANGE = (-13.1, 13.1)
VEL_Y_RANGE = (-7.8, 7.8)

# Neighbor distance bounds (polar): r near rest length
# Qt data shows actual stretch is -0.02% to +0.70% of rest length.
# We use +/- 5% for margin.
R_MIN_FACTOR = 0.95   # 0.95 * rest_length
R_MAX_FACTOR = 1.05   # 1.05 * rest_length

# Neighbor relative velocity bounds (from Qt analysis: rel vel ~ 2x self vel range)
REL_VEL_X_RANGE = (-26.2, 26.2)
REL_VEL_Y_RANGE = (-15.6, 15.6)


def compute_tapered_masses():
    """Compute per-bead masses with linear taper (matches simulation.py)."""
    t = np.linspace(1.0, 1.0 / TAPER_RATIO, N_BEADS)
    return t / t.sum() * TOTAL_MASS


def sample_neighbor_polar(rng, n, rest_length):
    """Sample neighbor relative positions in polar coordinates.

    Returns (N, 2) relative positions with distance near rest_length
    and uniform angle.
    """
    r = rng.uniform(rest_length * R_MIN_FACTOR,
                    rest_length * R_MAX_FACTOR, size=n)
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

    Replicates Bead.cpp: onNeighborState (spring + damping) for each
    neighbor, then applyGravity (gravity + drag), then integrate
    (symplectic Euler). Fixed beads produce zero deltas.

    All positions are relative (self is at origin).

    Returns:
        delta_pos: (N, 2)
        delta_vel: (N, 2)
    """
    N = len(vel)
    force = np.zeros((N, 2))

    # --- Spring + damping from left neighbor ---
    # left_rel_pos IS the delta (neighbor_pos - self_pos)
    dist_l = np.linalg.norm(left_rel_pos, axis=1, keepdims=True)
    safe_l = (dist_l.squeeze() > 1e-7) & has_left
    if safe_l.any():
        dir_l = np.where(safe_l[:, None], left_rel_pos / np.maximum(dist_l, 1e-7), 0)
        stretch_l = dist_l.squeeze() - left_rest_len
        f_spring = (STIFFNESS * stretch_l)[:, None] * dir_l
        v_along_l = np.sum(left_rel_vel * dir_l, axis=1)
        f_damp = (DAMPING * v_along_l)[:, None] * dir_l
        force += np.where(safe_l[:, None], f_spring + f_damp, 0)

    # --- Spring + damping from right neighbor ---
    dist_r = np.linalg.norm(right_rel_pos, axis=1, keepdims=True)
    safe_r = (dist_r.squeeze() > 1e-7) & has_right
    if safe_r.any():
        dir_r = np.where(safe_r[:, None], right_rel_pos / np.maximum(dist_r, 1e-7), 0)
        stretch_r = dist_r.squeeze() - right_rest_len
        f_spring = (STIFFNESS * stretch_r)[:, None] * dir_r
        v_along_r = np.sum(right_rel_vel * dir_r, axis=1)
        f_damp = (DAMPING * v_along_r)[:, None] * dir_r
        force += np.where(safe_r[:, None], f_spring + f_damp, 0)

    # --- Gravity ---
    force[:, 1] -= mass * GRAVITY

    # --- Drag ---
    force -= DRAG * vel

    # --- Symplectic Euler ---
    acc = force / mass[:, None]
    vel_new = vel + DT * acc
    # delta_pos = dt * vel_new (position change using new velocity)
    delta_pos_raw = DT * vel_new
    delta_vel_raw = vel_new - vel

    # Fixed beads: zero delta
    free = ~is_fixed
    delta_pos = delta_pos_raw * free[:, None]
    delta_vel = delta_vel_raw * free[:, None]

    return delta_pos, delta_vel


def generate(args):
    rng = np.random.default_rng(args.seed)
    masses = compute_tapered_masses()

    r_min = REST_LENGTH * R_MIN_FACTOR
    r_max = REST_LENGTH * R_MAX_FACTOR

    print(f"Generating {args.n_samples:,} volume-sampled bead training samples...")
    print(f"  Masses: {masses}")
    print(f"  Self vel bounds: vel_x={VEL_X_RANGE}, vel_y={VEL_Y_RANGE}")
    print(f"  Neighbor distance: r=[{r_min:.4f}, {r_max:.4f}] "
          f"(rest={REST_LENGTH:.4f}, +/-{(R_MAX_FACTOR-1)*100:.0f}%)")
    print(f"  Neighbor rel vel: dvx={REL_VEL_X_RANGE}, dvy={REL_VEL_Y_RANGE}")

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

    # Sample self velocity
    vel_x = rng.uniform(*VEL_X_RANGE, size=N)
    vel_y = rng.uniform(*VEL_Y_RANGE, size=N)
    vel = np.column_stack([vel_x, vel_y])

    # Sample left neighbor relative position (polar) and relative velocity
    left_rel_pos_rand = sample_neighbor_polar(rng, N, REST_LENGTH)
    left_rel_vel_rand = sample_rel_vel(rng, N)
    # Ghost beads: zero relative position and velocity
    left_rel_pos = np.where(has_left[:, None], left_rel_pos_rand, 0.0)
    left_rel_vel = np.where(has_left[:, None], left_rel_vel_rand, 0.0)

    # Sample right neighbor relative position and velocity
    right_rel_pos_rand = sample_neighbor_polar(rng, N, REST_LENGTH)
    right_rel_vel_rand = sample_rel_vel(rng, N)
    right_rel_pos = np.where(has_right[:, None], right_rel_pos_rand, 0.0)
    right_rel_vel = np.where(has_right[:, None], right_rel_vel_rand, 0.0)

    # Compute physics
    print("Computing per-bead physics...")
    delta_pos, delta_vel = compute_bead_step(
        vel, mass, is_fixed,
        left_rel_pos, left_rel_vel, has_left, left_rest_len,
        right_rel_pos, right_rel_vel, has_right, right_rest_len,
    )

    # Build input array:
    #   Self:  [vel_x, vel_y, mass, is_fixed]                     (4)
    #   Left:  [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]  (6)
    #   Right: [dpos_x, dpos_y, dvel_x, dvel_y, mass, rest_len]  (6)
    # Total: 16 features
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
