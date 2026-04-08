"""Evaluate BeadMPGNN generalization across different chain lengths.

Tests the trained model on chains of various lengths (some seen during
training, some not) and reports trajectory error, rod length violations,
and energy drift.

This is the key test: can a GNN trained on variable-length volume data
generalize to chain lengths it has never seen?

Usage:
    python bead_eval_variable.py
    python bead_eval_variable.py --model_path outputs/mpgnn_best.pt --n_steps 20000
"""

import os
import sys
import argparse
import numpy as np
import torch

from bead_mpgnn import BeadMPGNN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation import create_chain, compute_forces, compute_energy


def load_model(model_path, device):
    """Load trained BeadMPGNN from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = BeadMPGNN(
        node_dim=4, edge_dim=6, output_dim=4,
        hidden_dim=args["hidden_dim"],
        n_message_passes=args.get("n_message_passes", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded BeadMPGNN (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.2e})")
    return model


def step_ground_truth(state, dt, gravity):
    """One timestep: forces + symplectic Euler (no SHAKE)."""
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def step_gnn(state, model, device):
    """One timestep: MPGNN replaces all physics."""
    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest_lengths = torch.from_numpy(state.rest_lengths).float().to(device)

    with torch.no_grad():
        pos_new, vel_new = model.step_chain(pos, vel, mass, is_fixed, edges, rest_lengths)

    state.pos = pos_new.cpu().numpy()
    state.vel = vel_new.cpu().numpy()


def compute_rod_errors(state):
    """Max and mean relative rod length error."""
    errors = []
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        d = np.linalg.norm(state.pos[j] - state.pos[i])
        err = abs(d - state.rest_lengths[e]) / state.rest_lengths[e]
        errors.append(err)
    return np.max(errors), np.mean(errors)


def evaluate_chain(n_beads, model, device, n_steps, dt, gravity, taper_ratio):
    """Run GT and GNN side by side on a chain of n_beads."""
    params = dict(
        n_nodes=n_beads, total_length=2.0, total_mass=0.5,
        stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=taper_ratio,
    )

    state_gt = create_chain(**params)
    state_gnn = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    sample_interval = max(1, n_steps // 100)
    pos_errors = []
    rod_errs_max = []
    rod_errs_mean = []
    energy_gt_list = []
    energy_gnn_list = []

    for step in range(1, n_steps + 1):
        # Ground truth
        step_ground_truth(state_gt, dt, gravity)
        state_gt.pos[0] = anchor
        state_gt.vel[0] = 0.0

        # GNN
        step_gnn(state_gnn, model, device)
        state_gnn.pos[0] = anchor
        state_gnn.vel[0] = 0.0

        if step % sample_interval == 0:
            # Position error (mean over beads)
            err = np.linalg.norm(state_gnn.pos - state_gt.pos, axis=-1).mean()
            pos_errors.append(err)

            # Rod length errors
            re_max, re_mean = compute_rod_errors(state_gnn)
            rod_errs_max.append(re_max)
            rod_errs_mean.append(re_mean)

            # Energy
            energy_gt_list.append(compute_energy(state_gt, gravity)["total"])
            energy_gnn_list.append(compute_energy(state_gnn, gravity)["total"])

    return {
        "pos_error_mean": np.mean(pos_errors),
        "pos_error_max": np.max(pos_errors),
        "pos_error_final": pos_errors[-1],
        "rod_err_max": np.max(rod_errs_max),
        "rod_err_mean": np.mean(rod_errs_mean),
        "energy_drift": abs(energy_gnn_list[-1] - energy_gt_list[-1]),
        "pos_errors": pos_errors,
    }


def main(args):
    device = torch.device("cpu")
    model = load_model(args.model_path, device)

    chain_lengths = [int(n) for n in args.chain_lengths.split(",")]
    dt = 0.0001
    gravity = 9.81

    print(f"\nEvaluating on chain lengths: {chain_lengths}")
    print(f"Steps per chain: {args.n_steps}")
    print(f"Taper ratio: {args.taper_ratio}")
    print()

    # Header
    print(f"{'N beads':>8} | {'pos err (mean)':>14} | {'pos err (final)':>15} | "
          f"{'rod err (max)':>14} | {'energy drift':>13}")
    print("-" * 80)

    results = {}
    for n in chain_lengths:
        print(f"  Running N={n}...", end="", flush=True)
        r = evaluate_chain(n, model, device, args.n_steps, dt, gravity, args.taper_ratio)
        results[n] = r
        print(f"\r{n:>8} | {r['pos_error_mean']:>14.4e} | {r['pos_error_final']:>15.4e} | "
              f"{r['rod_err_max']:>14.4e} | {r['energy_drift']:>13.4e}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Position error over time for each chain length
        ax = axes[0]
        for n in chain_lengths:
            t = np.linspace(0, args.n_steps * dt, len(results[n]["pos_errors"]))
            ax.semilogy(t, results[n]["pos_errors"], label=f"N={n}", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Position Error (m)")
        ax.set_title("Position Error vs Ground Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bar chart: final position error by chain length
        ax = axes[1]
        final_errs = [results[n]["pos_error_final"] for n in chain_lengths]
        colors = ["green" if e < 0.1 else "orange" if e < 1.0 else "red" for e in final_errs]
        ax.bar(range(len(chain_lengths)), final_errs, color=colors, alpha=0.8)
        ax.set_xticks(range(len(chain_lengths)))
        ax.set_xticklabels([str(n) for n in chain_lengths])
        ax.set_xlabel("Chain Length (beads)")
        ax.set_ylabel("Final Position Error (m)")
        ax.set_title("Generalization by Chain Length")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Bar chart: max rod error by chain length
        ax = axes[2]
        rod_errs = [results[n]["rod_err_max"] for n in chain_lengths]
        ax.bar(range(len(chain_lengths)), rod_errs, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(chain_lengths)))
        ax.set_xticklabels([str(n) for n in chain_lengths])
        ax.set_xlabel("Chain Length (beads)")
        ax.set_ylabel("Max Rod Length Error (relative)")
        ax.set_title("Constraint Satisfaction")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out_dir = os.path.dirname(args.model_path)
        plot_path = os.path.join(out_dir, "mpgnn_variable_eval.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MPGNN on variable chain lengths")
    parser.add_argument("--model_path", type=str, default="outputs/mpgnn_best.pt")
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--chain_lengths", type=str, default="8,10,12,16,20,24,32",
                        help="Comma-separated list of chain lengths to test")
    parser.add_argument("--taper_ratio", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
