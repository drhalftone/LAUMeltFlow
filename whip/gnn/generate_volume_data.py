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

# --- Volume bounds (from simulation analysis + safety margin) ---
# Self velocity bounds: actual max is 13.1 m/s at tip bead.
# Add 50% margin so the model handles GNN rollout errors gracefully.
VEL_X_RANGE = (-20.0, 20.0)
VEL_Y_RANGE = (-12.0, 12.0)

# Neighbor distance bounds (polar): r near rest length
# Qt data shows actual stretch is -0.02% to +0.70% of rest length.
# Use +/- 10% for margin against rollout drift.
R_MIN_FACTOR = 0.90   # 0.90 * rest_length
R_MAX_FACTOR = 1.10   # 1.10 * rest_length

# Neighbor relative velocity bounds (2x self vel range)
REL_VEL_X_RANGE = (-40.0, 40.0)
REL_VEL_Y_RANGE = (-24.0, 24.0)


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


def generate_batch(rng, n, masses, near_zero_scale=None):
    """Generate n samples. If near_zero_scale is set, sample velocities and
    stretch from a narrow band around zero (Gaussian with that scale)."""

    bead_ids = rng.integers(0, N_BEADS, size=n)
    mass = masses[bead_ids]
    is_fixed = bead_ids == 0
    has_left = bead_ids > 0
    has_right = bead_ids < (N_BEADS - 1)

    left_mass = np.where(has_left, masses[np.clip(bead_ids - 1, 0, N_BEADS - 1)], 0.0)
    right_mass = np.where(has_right, masses[np.clip(bead_ids + 1, 0, N_BEADS - 1)], 0.0)
    left_rest_len = np.where(has_left, REST_LENGTH, 0.0)
    right_rest_len = np.where(has_right, REST_LENGTH, 0.0)

    if near_zero_scale is None:
        # Full volume sampling
        vel_x = rng.uniform(*VEL_X_RANGE, size=n)
        vel_y = rng.uniform(*VEL_Y_RANGE, size=n)
        left_rel_pos_rand = sample_neighbor_polar(rng, n, REST_LENGTH)
        left_rel_vel_rand = sample_rel_vel(rng, n)
        right_rel_pos_rand = sample_neighbor_polar(rng, n, REST_LENGTH)
        right_rel_vel_rand = sample_rel_vel(rng, n)
    else:
        # Near-zero sampling: small velocities, rods near rest length
        s = near_zero_scale
        vel_x = rng.normal(0, s * VEL_X_RANGE[1], size=n)
        vel_y = rng.normal(0, s * VEL_Y_RANGE[1], size=n)

        # Neighbor at rest length with tiny stretch (Gaussian around 0%)
        r = REST_LENGTH + rng.normal(0, s * REST_LENGTH * 0.05, size=n)
        r = np.clip(r, REST_LENGTH * 0.9, REST_LENGTH * 1.1)
        theta = rng.uniform(0, 2 * np.pi, size=n)
        left_rel_pos_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        right_rel_pos_rand = np.column_stack([r * np.cos(rng.uniform(0, 2*np.pi, n)),
                                               r * np.sin(rng.uniform(0, 2*np.pi, n))])

        # Small relative velocities
        left_rel_vel_rand = np.column_stack([
            rng.normal(0, s * REL_VEL_X_RANGE[1], size=n),
            rng.normal(0, s * REL_VEL_Y_RANGE[1], size=n),
        ])
        right_rel_vel_rand = np.column_stack([
            rng.normal(0, s * REL_VEL_X_RANGE[1], size=n),
            rng.normal(0, s * REL_VEL_Y_RANGE[1], size=n),
        ])

    vel = np.column_stack([vel_x, vel_y])
    left_rel_pos = np.where(has_left[:, None], left_rel_pos_rand, 0.0)
    left_rel_vel = np.where(has_left[:, None], left_rel_vel_rand, 0.0)
    right_rel_pos = np.where(has_right[:, None], right_rel_pos_rand, 0.0)
    right_rel_vel = np.where(has_right[:, None], right_rel_vel_rand, 0.0)

    return (bead_ids, vel, mass, is_fixed, has_left, has_right,
            left_rel_pos, left_rel_vel, left_mass, left_rest_len,
            right_rel_pos, right_rel_vel, right_mass, right_rest_len)


def generate_trajectory_samples(rng, n_sims, n_steps, masses):
    """Run actual simulations and extract per-bead samples from trajectories.

    Each simulation starts with random velocity perturbations. Every step,
    we record per-bead input features and the resulting deltas — capturing
    the correlated states that actually occur during chain dynamics.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from simulation import create_chain, compute_forces

    all_vel = []
    all_mass = []
    all_fixed = []
    all_has_left = []
    all_has_right = []
    all_left_rel_pos = []
    all_left_rel_vel = []
    all_left_mass = []
    all_left_rest = []
    all_right_rel_pos = []
    all_right_rel_vel = []
    all_right_mass = []
    all_right_rest = []

    for sim_id in range(n_sims):
        state = create_chain(
            n_nodes=N_BEADS, total_length=TOTAL_LENGTH, total_mass=TOTAL_MASS,
            stiffness=STIFFNESS, damping=DAMPING, drag=DRAG,
            taper_ratio=TAPER_RATIO,
        )
        anchor = state.pos[0].copy()

        # Random initial velocity perturbation
        for i in range(1, N_BEADS):
            state.vel[i] = rng.normal(0, 2.0, size=2)

        for step in range(n_steps):
            # Capture pre-step state
            pos_pre = state.pos.copy()
            vel_pre = state.vel.copy()

            # Step physics (no SHAKE)
            forces = compute_forces(state, GRAVITY)
            acc = forces / state.mass[:, None]
            acc[state.fixed] = 0.0
            state.vel += DT * acc
            state.vel[state.fixed] = 0.0
            state.pos += DT * state.vel
            state.pos[0] = anchor
            state.vel[0] = 0.0

            # Extract per-bead relative features from pre-step state
            for i in range(N_BEADS):
                all_vel.append(vel_pre[i])
                all_mass.append(state.mass[i])
                all_fixed.append(state.fixed[i])
                all_has_left.append(i > 0)
                all_has_right.append(i < N_BEADS - 1)

                if i > 0:
                    all_left_rel_pos.append(pos_pre[i-1] - pos_pre[i])
                    all_left_rel_vel.append(vel_pre[i-1] - vel_pre[i])
                    all_left_mass.append(state.mass[i-1])
                    all_left_rest.append(state.rest_lengths[i-1])
                else:
                    all_left_rel_pos.append([0.0, 0.0])
                    all_left_rel_vel.append([0.0, 0.0])
                    all_left_mass.append(0.0)
                    all_left_rest.append(0.0)

                if i < N_BEADS - 1:
                    all_right_rel_pos.append(pos_pre[i+1] - pos_pre[i])
                    all_right_rel_vel.append(vel_pre[i+1] - vel_pre[i])
                    all_right_mass.append(state.mass[i+1])
                    all_right_rest.append(state.rest_lengths[i])
                else:
                    all_right_rel_pos.append([0.0, 0.0])
                    all_right_rel_vel.append([0.0, 0.0])
                    all_right_mass.append(0.0)
                    all_right_rest.append(0.0)

        if (sim_id + 1) % max(1, n_sims // 10) == 0:
            print(f"\r    Trajectory {sim_id+1}/{n_sims}", end="", flush=True)

    print()

    n = len(all_vel)
    bead_ids = np.tile(np.arange(N_BEADS), n_sims * n_steps)[:n]
    vel = np.array(all_vel)
    mass = np.array(all_mass)
    is_fixed = np.array(all_fixed)
    has_left = np.array(all_has_left)
    has_right = np.array(all_has_right)
    left_rel_pos = np.array(all_left_rel_pos)
    left_rel_vel = np.array(all_left_rel_vel)
    left_mass = np.array(all_left_mass)
    left_rest_len = np.array(all_left_rest)
    right_rel_pos = np.array(all_right_rel_pos)
    right_rel_vel = np.array(all_right_rel_vel)
    right_mass = np.array(all_right_mass)
    right_rest_len = np.array(all_right_rest)

    return (bead_ids, vel, mass, is_fixed, has_left, has_right,
            left_rel_pos, left_rel_vel, left_mass, left_rest_len,
            right_rel_pos, right_rel_vel, right_mass, right_rest_len)


def generate(args):
    rng = np.random.default_rng(args.seed)
    masses = compute_tapered_masses()

    r_min = REST_LENGTH * R_MIN_FACTOR
    r_max = REST_LENGTH * R_MAX_FACTOR

    # Split: 40% full-volume, 40% near-zero, 20% trajectory
    n_full = args.n_samples * 2 // 5
    n_near = args.n_samples * 2 // 5
    # Trajectory samples: n_sims simulations, each n_traj_steps steps, 16 beads each
    n_traj_target = args.n_samples - n_full - n_near
    n_traj_steps = 2000  # steps per simulation
    n_sims = max(1, n_traj_target // (n_traj_steps * N_BEADS))
    n_traj_actual = n_sims * n_traj_steps * N_BEADS

    print(f"Generating ~{args.n_samples:,} bead training samples...")
    print(f"  {n_full:,} full-volume + {n_near:,} near-zero + "
          f"{n_traj_actual:,} trajectory ({n_sims} sims x {n_traj_steps} steps)")
    print(f"  Masses: {masses}")
    print(f"  Self vel bounds: vel_x={VEL_X_RANGE}, vel_y={VEL_Y_RANGE}")
    print(f"  Neighbor distance: r=[{r_min:.4f}, {r_max:.4f}] "
          f"(rest={REST_LENGTH:.4f}, +/-{(R_MAX_FACTOR-1)*100:.0f}%)")
    print(f"  Near-zero scale: {args.near_zero_scale}")

    # Generate full-volume batch
    print("  Generating full-volume samples...")
    full = generate_batch(rng, n_full, masses, near_zero_scale=None)

    # Generate near-zero batch (small velocities, rods near rest)
    print("  Generating near-zero samples...")
    near = generate_batch(rng, n_near, masses, near_zero_scale=args.near_zero_scale)

    # Generate trajectory samples from actual simulations
    print("  Generating trajectory samples...")
    traj = generate_trajectory_samples(rng, n_sims, n_traj_steps, masses)

    # Concatenate all three populations
    def cat(*arrays):
        return np.concatenate(arrays, axis=0)

    bead_ids = cat(full[0], near[0], traj[0])
    vel = cat(full[1], near[1], traj[1])
    mass = cat(full[2], near[2], traj[2])
    is_fixed = cat(full[3], near[3], traj[3])
    has_left = cat(full[4], near[4], traj[4])
    has_right = cat(full[5], near[5], traj[5])
    left_rel_pos = cat(full[6], near[6], traj[6])
    left_rel_vel = cat(full[7], near[7], traj[7])
    left_mass = cat(full[8], near[8], traj[8])
    left_rest_len = cat(full[9], near[9], traj[9])
    right_rel_pos = cat(full[10], near[10], traj[10])
    right_rel_vel = cat(full[11], near[11], traj[11])
    right_mass = cat(full[12], near[12], traj[12])
    right_rest_len = cat(full[13], near[13], traj[13])

    N = len(bead_ids)

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
    parser.add_argument("--n_samples", type=int, default=2_000_000)
    parser.add_argument("--output", type=str, default="volume_data.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near_zero_scale", type=float, default=0.05,
                        help="Gaussian scale for near-zero samples (fraction of full range)")
    args = parser.parse_args()
    generate(args)
