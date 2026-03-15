"""Generate training data for U-Net GNN surrogate.

Instead of sampling from simulation trajectories (which gives biased coverage),
we uniformly sample random bead chain configurations and compute the SHAKE
correction for each. This gives uniform coverage of the input state space,
following the approach from the Roe flux GNN report (Section 5).

Each sample:
  1. Start from a base chain configuration
  2. Apply random perturbations to positions and velocities
  3. Run SHAKE to get corrected state
  4. Save (perturbed state, SHAKE correction) as input/target pair

Outputs paired input/target folders:
    data/input/frame00001.npy   -- (16, 7) bead features before SHAKE
    data/target/frame00001.npy  -- (16, 4) SHAKE corrections [dx, dy, dvx, dvy]
    data/graph.npz              -- shared graph structure
"""

import os
import argparse
import numpy as np
from simulation import create_chain, project_constraints, build_tree_edges


def generate_random_sample(base_state, pos_scale, vel_scale, rng):
    """Generate one random chain configuration and its SHAKE correction.

    Perturbs positions and velocities randomly around a physically plausible
    configuration, then runs SHAKE to get the correction.

    Args:
        base_state: template ChainState (for mass, edges, rest_lengths)
        pos_scale: max position perturbation per bead (meters)
        vel_scale: max velocity magnitude (m/s)
        rng: numpy random generator

    Returns:
        pre_pos, pre_vel: state before SHAKE (16, 2)
        pos_corr, vel_corr: SHAKE corrections (16, 2)
    """
    n = len(base_state.mass)
    L0 = base_state.rest_lengths[0]

    # Generate a random chain configuration:
    # Start from anchor at origin, then place each bead at rest_length
    # from the previous one in a random direction. This ensures the chain
    # is roughly connected but with random shape.
    pos = np.zeros((n, 2))
    for i in range(1, n):
        # Random angle for this segment
        angle = rng.uniform(-np.pi, np.pi)
        pos[i] = pos[i - 1] + L0 * np.array([np.cos(angle), np.sin(angle)])

    # Add small random perturbation to stretch/compress rods slightly
    # (this is what SHAKE needs to correct)
    pos[1:] += rng.uniform(-pos_scale, pos_scale, size=(n - 1, 2))

    # Random velocities
    vel = rng.uniform(-vel_scale, vel_scale, size=(n, 2))
    vel[0] = 0.0  # anchor is fixed

    # Copy into a temporary state for SHAKE
    state = create_chain(
        n_nodes=n,
        total_length=base_state.rest_lengths.sum() + L0,
        total_mass=base_state.mass.sum(),
        stiffness=base_state.stiffness,
        damping=base_state.damping,
        taper_ratio=base_state.taper_ratio,
    )
    state.pos = pos.copy()
    state.vel = vel.copy()

    # Save pre-SHAKE state
    pre_pos = pos.copy()
    pre_vel = vel.copy()

    # Run SHAKE
    project_constraints(state, dt=0.0001, n_iters=20, tol=1e-10)

    # Corrections
    pos_corr = state.pos - pre_pos
    vel_corr = state.vel - pre_vel

    return pre_pos, pre_vel, pos_corr, vel_corr


def main(args):
    DATA_DIR = args.data_dir
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    TARGET_DIR = os.path.join(DATA_DIR, "target")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_beads = 16

    # Base chain for structure (mass, edges, rest_lengths)
    base = create_chain(
        n_nodes=n_beads,
        total_length=2.0,
        total_mass=0.5,
        stiffness=1e4,
        damping=0.5,
        taper_ratio=10.0,
    )

    node_mass = base.mass
    node_types = base.fixed.astype(np.float32)
    bead_levels = np.zeros(n_beads)

    # Position perturbation ranges (importance sampling like the Roe report)
    # Core region (60%): small perturbations (typical simulation states)
    # Extended region (40%): larger perturbations (extreme states)
    n_core = int(args.n_samples * 0.6)
    n_extended = args.n_samples - n_core

    print(f"Generating {args.n_samples} samples:")
    print(f"  Core region:     {n_core} (pos_scale={args.pos_scale_core:.4f}, vel_scale={args.vel_scale_core:.1f})")
    print(f"  Extended region: {n_extended} (pos_scale={args.pos_scale_ext:.4f}, vel_scale={args.vel_scale_ext:.1f})")

    sample_idx = 0
    for region, n_region, pos_scale, vel_scale in [
        ("core", n_core, args.pos_scale_core, args.vel_scale_core),
        ("extended", n_extended, args.pos_scale_ext, args.vel_scale_ext),
    ]:
        for i in range(n_region):
            if (sample_idx + 1) % (args.n_samples // 10) == 0:
                pct = 100 * (sample_idx + 1) / args.n_samples
                print(f"\r  Generating... {pct:.0f}%", end="", flush=True)

            pre_pos, pre_vel, pos_corr, vel_corr = generate_random_sample(
                base, pos_scale, vel_scale, rng
            )

            # Input: (16, 7) [pos_x, pos_y, vel_x, vel_y, mass, type, level]
            input_tensor = np.column_stack([
                pre_pos, pre_vel, node_mass, node_types, bead_levels,
            ]).astype(np.float32)

            # Target: (16, 4) [dx, dy, dvx, dvy]
            target_tensor = np.column_stack([
                pos_corr, vel_corr,
            ]).astype(np.float32)

            fname = f"frame{sample_idx:06d}.npy"
            np.save(os.path.join(INPUT_DIR, fname), input_tensor)
            np.save(os.path.join(TARGET_DIR, fname), target_tensor)
            sample_idx += 1

    print(f"\r  Generating... done.          ")

    # Save graph structure
    tree = build_tree_edges(n_beads)
    graph_path = os.path.join(DATA_DIR, "graph.npz")
    np.savez_compressed(
        graph_path,
        chain_edges=np.concatenate([base.edges, base.edges[:, ::-1]], axis=0).T,
        tree_edges=tree["tree_edges"],
        rest_lengths=base.rest_lengths,
        node_mass=node_mass,
        node_types=base.fixed.astype(np.int32),
        node_levels=tree["node_levels"],
        n_beads=np.array(n_beads),
        n_total_nodes=np.array(tree["n_total_nodes"]),
        n_levels=np.array(tree["n_levels"]),
    )

    # Summary statistics
    print(f"\nOutput:")
    print(f"  {INPUT_DIR}/ -- {sample_idx} files, shape (16, 7)")
    print(f"  {TARGET_DIR}/ -- {sample_idx} files, shape (16, 4)")
    print(f"  {graph_path}")

    # Quick stats on corrections
    print(f"\nSampling correction magnitudes from last 100 samples...")
    pos_mags = []
    vel_mags = []
    for i in range(max(0, sample_idx - 100), sample_idx):
        t = np.load(os.path.join(TARGET_DIR, f"frame{i:06d}.npy"))
        pos_mags.append(np.linalg.norm(t[:, :2], axis=1).mean())
        vel_mags.append(np.linalg.norm(t[:, 2:], axis=1).mean())
    print(f"  Position correction: mean={np.mean(pos_mags):.4e}, max={np.max(pos_mags):.4e}")
    print(f"  Velocity correction: mean={np.mean(vel_mags):.4e}, max={np.max(vel_mags):.4e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate uniform training data")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    # Core region: small perturbations (typical simulation states)
    parser.add_argument("--pos_scale_core", type=float, default=0.005,
                        help="Position perturbation for core samples (meters)")
    parser.add_argument("--vel_scale_core", type=float, default=5.0,
                        help="Velocity range for core samples (m/s)")
    # Extended region: larger perturbations (extreme states)
    parser.add_argument("--pos_scale_ext", type=float, default=0.02,
                        help="Position perturbation for extended samples (meters)")
    parser.add_argument("--vel_scale_ext", type=float, default=20.0,
                        help="Velocity range for extended samples (m/s)")
    args = parser.parse_args()

    main(args)
