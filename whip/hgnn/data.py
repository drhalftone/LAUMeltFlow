"""Generate conservative (no drag/damping) trajectory data for HGNN training.

Runs the existing bead chain simulation with drag=0, damping=0 and
SHAKE constraints enabled. Saves (q, p) trajectory chunks for training
the Hamiltonian GNN.

Usage:
    python data.py
    python data.py --n_sims 100 --n_steps 30000 --chunk_len 5
"""

import os
import sys
import argparse
import numpy as np

# Add parent dir for simulation imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import create_chain, step_symplectic_euler, compute_energy


def generate_one_trajectory(sim_id, args):
    """Run one simulation with random initial perturbation."""
    rng = np.random.RandomState(sim_id)

    state = create_chain(
        n_nodes=args.n_nodes,
        total_length=args.total_length,
        total_mass=args.total_mass,
        stiffness=args.stiffness,
        damping=0.0,       # conservative!
        drag=0.0,          # conservative!
        taper_ratio=args.taper_ratio,
    )
    anchor_origin = state.pos[0].copy()

    # Random initial perturbation: small velocity kick to each free bead
    for i in range(1, args.n_nodes):
        state.vel[i] = rng.randn(2) * args.init_vel_scale

    # Storage
    n_saved = args.n_steps // args.save_interval + 1
    positions = np.zeros((n_saved, args.n_nodes, 2))
    momenta = np.zeros((n_saved, args.n_nodes, 2))
    times = np.zeros(n_saved)

    # Initial state
    positions[0] = state.pos.copy()
    momenta[0] = state.vel * state.mass[:, None]  # p = m * v
    times[0] = 0.0
    save_idx = 1

    for step in range(1, args.n_steps + 1):
        step_symplectic_euler(
            state, args.dt, args.gravity,
            use_constraints=True,
            constraint_iters=10,
        )
        # Re-pin anchor
        state.pos[0] = anchor_origin
        state.vel[0] = 0.0

        if step % args.save_interval == 0 and save_idx < n_saved:
            positions[save_idx] = state.pos.copy()
            momenta[save_idx] = state.vel * state.mass[:, None]
            times[save_idx] = step * args.dt
            save_idx += 1

    positions = positions[:save_idx]
    momenta = momenta[:save_idx]
    times = times[:save_idx]

    return positions, momenta, times, state.mass.copy(), state.rest_lengths.copy()


def chunk_trajectory(positions, momenta, times, chunk_len):
    """Split a trajectory into overlapping chunks of length chunk_len."""
    n_frames = len(positions)
    chunks_q = []
    chunks_p = []
    chunks_t = []

    for start in range(n_frames - chunk_len):
        end = start + chunk_len
        chunks_q.append(positions[start:end])
        chunks_p.append(momenta[start:end])
        chunks_t.append(times[start:end])

    return (np.array(chunks_q), np.array(chunks_p), np.array(chunks_t))


def main(args):
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    all_chunks_q = []
    all_chunks_p = []
    all_chunks_t = []
    masses = None
    rest_lengths = None

    for sim_id in range(args.n_sims):
        pct = 100 * (sim_id + 1) / args.n_sims
        print(f"\rSimulation {sim_id + 1}/{args.n_sims} ({pct:.0f}%)", end="", flush=True)

        positions, momenta, times, mass, rl = generate_one_trajectory(sim_id, args)
        masses = mass
        rest_lengths = rl

        cq, cp, ct = chunk_trajectory(positions, momenta, times, args.chunk_len)
        all_chunks_q.append(cq)
        all_chunks_p.append(cp)
        all_chunks_t.append(ct)

    print()

    # Concatenate all chunks
    all_q = np.concatenate(all_chunks_q, axis=0)  # (N_chunks, chunk_len, n_nodes, 2)
    all_p = np.concatenate(all_chunks_p, axis=0)
    all_t = np.concatenate(all_chunks_t, axis=0)   # (N_chunks, chunk_len)

    print(f"Total chunks: {len(all_q)}")
    print(f"Chunk shape: q={all_q.shape}, p={all_p.shape}")

    # Save
    np.savez(
        os.path.join(out_dir, "trajectories.npz"),
        q=all_q,
        p=all_p,
        t=all_t,
        mass=masses,
        rest_lengths=rest_lengths,
        n_nodes=args.n_nodes,
        total_length=args.total_length,
        total_mass=args.total_mass,
        taper_ratio=args.taper_ratio,
        gravity=args.gravity,
        dt_save=args.dt * args.save_interval,
    )

    print(f"Saved to {out_dir}/trajectories.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conservative trajectory data")
    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--chunk_len", type=int, default=5)
    parser.add_argument("--n_nodes", type=int, default=16)
    parser.add_argument("--total_length", type=float, default=2.0)
    parser.add_argument("--total_mass", type=float, default=0.5)
    parser.add_argument("--stiffness", type=float, default=1e4)
    parser.add_argument("--taper_ratio", type=float, default=10.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--dt", type=float, default=0.0001)
    parser.add_argument("--init_vel_scale", type=float, default=2.0)
    args = parser.parse_args()

    main(args)
