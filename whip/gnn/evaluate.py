"""Evaluate trained U-Net GNN by running simulation with GNN replacing SHAKE.

Runs three simulations side-by-side:
  1. Ground truth: symplectic Euler + SHAKE (10 iterations)
  2. GNN surrogate: symplectic Euler + GNN correction (no SHAKE)
  3. No correction: symplectic Euler only (no SHAKE, no GNN)

Compares trajectories, rod length violations, and energy drift.

Usage:
    python evaluate.py
    python evaluate.py --model_path outputs/best_model.pt --n_steps 30000
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add parent dir so we can import simulation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import (
    create_chain, compute_forces, project_constraints, compute_energy,
    build_tree_edges,
)
from model import BeadChainUNet


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = BeadChainUNet(
        input_dim=7,
        output_dim=4,
        hidden_dim=args["hidden_dim"],
        n_beads=16,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path} (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.2e})")
    return model


def build_graph_tensors(n_beads, device):
    """Build graph connectivity tensors for the model."""
    tree = build_tree_edges(n_beads)
    n_total = tree["n_total_nodes"]

    # Chain edges (bidirectional)
    edges = np.array([[i, i + 1] for i in range(n_beads - 1)])
    fwd = edges
    bwd = edges[:, ::-1]
    chain_edges = np.concatenate([fwd, bwd], axis=0).T  # (2, 30)

    # Tree edges
    tree_src = tree["tree_edges"][:, 0]
    tree_dst = tree["tree_edges"][:, 1]
    tree_edge_index = np.stack([tree_src, tree_dst])  # (2, 60)

    return {
        "chain_edges": torch.from_numpy(chain_edges).long().to(device),
        "tree_edge_index": torch.from_numpy(tree_edge_index).long().to(device),
        "node_levels": torch.from_numpy(tree["node_levels"]).long().to(device),
        "n_total": n_total,
        "n_interior": tree["n_total_nodes"] - n_beads,
    }


def state_to_input_tensor(state, graph, device):
    """Convert simulation state to model input tensor (1, 31, 7)."""
    n_beads = len(state.mass)
    n_interior = graph["n_interior"]

    # Bead features: [pos_x, pos_y, vel_x, vel_y, mass, type, level]
    bead_feat = np.column_stack([
        state.pos,                                # (16, 2)
        state.vel,                                # (16, 2)
        state.mass,                               # (16,)
        state.fixed.astype(np.float32),           # (16,)
        np.zeros(n_beads),                        # (16,) level=0
    ])  # (16, 7)

    # Interior node features (zeros + level)
    interior_feat = np.zeros((n_interior, 7))
    interior_feat[:, 5] = 2  # node_type = interior
    levels_np = graph["node_levels"].cpu().numpy()
    interior_feat[:, 6] = levels_np[n_beads:]

    full = np.vstack([bead_feat, interior_feat])  # (31, 7)
    return torch.from_numpy(full).float().unsqueeze(0).to(device)  # (1, 31, 7)


def step_with_gnn(state, dt, gravity, model, graph, device):
    """One timestep: symplectic Euler + GNN correction (no SHAKE)."""
    # Phase 1-2: normal integration
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel

    # Phase 3: GNN correction instead of SHAKE
    x = state_to_input_tensor(state, graph, device)
    with torch.no_grad():
        correction = model(
            x,
            graph["chain_edges"],
            graph["tree_edge_index"],
            graph["node_levels"],
            graph["n_total"],
        )[0].cpu().numpy()  # (16, 4)

    # Apply corrections
    state.pos += correction[:, :2]
    state.vel += correction[:, 2:]


def step_no_correction(state, dt, gravity):
    """One timestep: symplectic Euler only (no SHAKE, no GNN)."""
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def step_with_shake(state, dt, gravity):
    """One timestep: symplectic Euler + SHAKE (ground truth)."""
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel
    project_constraints(state, dt, n_iters=10)


def compute_rod_errors(state):
    """Max relative rod length error."""
    max_err = 0.0
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        d = np.linalg.norm(state.pos[j] - state.pos[i])
        err = abs(d - state.rest_lengths[e]) / state.rest_lengths[e]
        max_err = max(max_err, err)
    return max_err


def run_evaluation(args):
    device = torch.device("cpu")

    # Load model
    model = load_model(args.model_path, device)
    graph = build_graph_tensors(16, device)

    # Simulation parameters (match training data)
    params = dict(
        n_nodes=16, total_length=2.0, total_mass=0.5,
        stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=10.0,
    )
    dt = 0.0001
    gravity = 9.81
    save_interval = 50

    # Create three identical chains
    state_gt = create_chain(**params)
    state_gnn = create_chain(**params)
    state_none = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    n_saved = args.n_steps // save_interval + 1
    pos_gt = np.zeros((n_saved, 16, 2))
    pos_gnn = np.zeros((n_saved, 16, 2))
    pos_none = np.zeros((n_saved, 16, 2))
    energy_gt = np.zeros(n_saved)
    energy_gnn = np.zeros(n_saved)
    energy_none = np.zeros(n_saved)
    rod_err_gt = np.zeros(n_saved)
    rod_err_gnn = np.zeros(n_saved)
    rod_err_none = np.zeros(n_saved)
    times = np.zeros(n_saved)

    # Initial state
    pos_gt[0] = state_gt.pos.copy()
    pos_gnn[0] = state_gnn.pos.copy()
    pos_none[0] = state_none.pos.copy()
    energy_gt[0] = compute_energy(state_gt, gravity)["total"]
    energy_gnn[0] = compute_energy(state_gnn, gravity)["total"]
    energy_none[0] = compute_energy(state_none, gravity)["total"]
    save_idx = 1

    print(f"Running {args.n_steps} steps...")
    print_interval = max(1, args.n_steps // 10)

    for step in range(1, args.n_steps + 1):
        if step % print_interval == 0:
            pct = 100 * step / args.n_steps
            print(f"\r  {pct:.0f}%", end="", flush=True)

        # Ground truth (SHAKE)
        step_with_shake(state_gt, dt, gravity)
        state_gt.pos[0] = anchor
        state_gt.vel[0] = 0.0

        # GNN surrogate
        step_with_gnn(state_gnn, dt, gravity, model, graph, device)
        state_gnn.pos[0] = anchor
        state_gnn.vel[0] = 0.0

        # No correction
        step_no_correction(state_none, dt, gravity)
        state_none.pos[0] = anchor
        state_none.vel[0] = 0.0

        if step % save_interval == 0 and save_idx < n_saved:
            pos_gt[save_idx] = state_gt.pos.copy()
            pos_gnn[save_idx] = state_gnn.pos.copy()
            pos_none[save_idx] = state_none.pos.copy()
            energy_gt[save_idx] = compute_energy(state_gt, gravity)["total"]
            energy_gnn[save_idx] = compute_energy(state_gnn, gravity)["total"]
            energy_none[save_idx] = compute_energy(state_none, gravity)["total"]
            rod_err_gt[save_idx] = compute_rod_errors(state_gt)
            rod_err_gnn[save_idx] = compute_rod_errors(state_gnn)
            rod_err_none[save_idx] = compute_rod_errors(state_none)
            times[save_idx] = step * dt
            save_idx += 1

    print("\r  done.       ")

    # Trim
    pos_gt = pos_gt[:save_idx]
    pos_gnn = pos_gnn[:save_idx]
    pos_none = pos_none[:save_idx]
    energy_gt = energy_gt[:save_idx]
    energy_gnn = energy_gnn[:save_idx]
    energy_none = energy_none[:save_idx]
    rod_err_gt = rod_err_gt[:save_idx]
    rod_err_gnn = rod_err_gnn[:save_idx]
    rod_err_none = rod_err_none[:save_idx]
    times = times[:save_idx]

    # --- Position error ---
    pos_error_gnn = np.linalg.norm(pos_gnn - pos_gt, axis=-1).mean(axis=1)
    pos_error_none = np.linalg.norm(pos_none - pos_gt, axis=-1).mean(axis=1)

    print(f"\nPosition error (mean over beads):")
    print(f"  GNN  final: {pos_error_gnn[-1]:.4e} m")
    print(f"  None final: {pos_error_none[-1]:.4e} m")

    print(f"\nRod length error (max relative):")
    print(f"  SHAKE final: {rod_err_gt[-1]:.4e}")
    print(f"  GNN   final: {rod_err_gnn[-1]:.4e}")
    print(f"  None  final: {rod_err_none[-1]:.4e}")

    print(f"\nEnergy drift:")
    print(f"  SHAKE: {energy_gt[-1] - energy_gt[0]:.4e} J")
    print(f"  GNN:   {energy_gnn[-1] - energy_gnn[0]:.4e} J")
    print(f"  None:  {energy_none[-1] - energy_none[0]:.4e} J")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Position error over time
    ax = axes[0, 0]
    ax.semilogy(times, pos_error_gnn, "r-", label="GNN vs GT", alpha=0.8)
    ax.semilogy(times, pos_error_none, "b-", label="No correction vs GT", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title("Trajectory Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rod length error
    ax = axes[0, 1]
    ax.semilogy(times, rod_err_gt, "k-", label="SHAKE", alpha=0.8)
    ax.semilogy(times, rod_err_gnn, "r-", label="GNN", alpha=0.8)
    ax.semilogy(times, rod_err_none, "b-", label="None", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max relative rod error")
    ax.set_title("Constraint Satisfaction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[1, 0]
    ax.plot(times, energy_gt, "k-", label="SHAKE", alpha=0.8)
    ax.plot(times, energy_gnn, "r-", label="GNN", alpha=0.8)
    ax.plot(times, energy_none, "b-", label="None", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total energy (J)")
    ax.set_title("Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final frame comparison
    ax = axes[1, 1]
    frame = -1
    ax.plot(pos_gt[frame, :, 0], pos_gt[frame, :, 1],
            "k-o", markersize=4, label="SHAKE")
    ax.plot(pos_gnn[frame, :, 0], pos_gnn[frame, :, 1],
            "r--x", markersize=4, label="GNN")
    ax.plot(pos_none[frame, :, 0], pos_none[frame, :, 1],
            "b:s", markersize=3, label="None")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Chain at t={times[frame]:.2f}s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(args.model_path), "evaluation.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net GNN on bead chain")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--n_steps", type=int, default=30000)
    args = parser.parse_args()

    run_evaluation(args)
