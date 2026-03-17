"""Live side-by-side visualization: SHAKE vs GNN vs No correction.

Shows three bead chains evolving simultaneously so you can see how the
GNN surrogate compares to ground truth in real time.

Usage:
    python visualize_eval.py
    python visualize_eval.py --model_path outputs/best_model.pt --n_steps 30000 --speed 5
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import BeadChainUNet, build_chain_adj, build_tree_children

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation import (
    create_chain, compute_forces, project_constraints, compute_energy,
)


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = BeadChainUNet(
        state_dim=7, output_dim=4,
        hidden_dim=args["hidden_dim"], n_beads=16,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.2e})")
    return model


def state_to_bead_tensor(state, device):
    n_beads = len(state.mass)
    bead_feat = np.column_stack([
        state.pos, state.vel, state.mass,
        state.fixed.astype(np.float32), np.zeros(n_beads),
    ])
    return torch.from_numpy(bead_feat).float().unsqueeze(0).to(device)


def run_and_visualize(args):
    device = torch.device("cpu")
    n_beads = 16

    model = load_model(args.model_path, device)
    chain_adj = torch.from_numpy(build_chain_adj(n_beads)).long().to(device)
    tree_children = build_tree_children(n_beads)

    params = dict(
        n_nodes=16, total_length=2.0, total_mass=0.5,
        stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=10.0,
    )
    dt = 0.0001
    gravity = 9.81
    steps_per_frame = args.speed  # sim steps between animation frames

    state_gt = create_chain(**params)
    state_gnn = create_chain(**params)
    state_none = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    n_frames = args.n_steps // steps_per_frame

    # Pre-simulate to collect frames
    print(f"Simulating {args.n_steps} steps ({n_frames} frames)...")
    frames_gt = [state_gt.pos.copy()]
    frames_gnn = [state_gnn.pos.copy()]
    frames_none = [state_none.pos.copy()]
    rod_errs_gt = [0.0]
    rod_errs_gnn = [0.0]
    rod_errs_none = [0.0]
    energies_gt = [compute_energy(state_gt, gravity)["total"]]
    energies_gnn = [compute_energy(state_gnn, gravity)["total"]]
    energies_none = [compute_energy(state_none, gravity)["total"]]
    times = [0.0]

    for step in range(1, args.n_steps + 1):
        if step % max(1, args.n_steps // 10) == 0:
            print(f"\r  {100 * step / args.n_steps:.0f}%", end="", flush=True)

        # SHAKE
        forces = compute_forces(state_gt, gravity)
        acc = forces / state_gt.mass[:, None]
        acc[state_gt.fixed] = 0.0
        state_gt.vel += dt * acc
        state_gt.vel[state_gt.fixed] = 0.0
        state_gt.pos += dt * state_gt.vel
        project_constraints(state_gt, dt, n_iters=10)
        state_gt.pos[0] = anchor; state_gt.vel[0] = 0.0

        # GNN
        forces = compute_forces(state_gnn, gravity)
        acc = forces / state_gnn.mass[:, None]
        acc[state_gnn.fixed] = 0.0
        state_gnn.vel += dt * acc
        state_gnn.vel[state_gnn.fixed] = 0.0
        state_gnn.pos += dt * state_gnn.vel
        x = state_to_bead_tensor(state_gnn, device)
        with torch.no_grad():
            correction = model(x, chain_adj, tree_children)[0].cpu().numpy()
        state_gnn.pos += correction[:, :2]
        state_gnn.vel += correction[:, 2:]
        state_gnn.pos[0] = anchor; state_gnn.vel[0] = 0.0

        # No correction
        forces = compute_forces(state_none, gravity)
        acc = forces / state_none.mass[:, None]
        acc[state_none.fixed] = 0.0
        state_none.vel += dt * acc
        state_none.vel[state_none.fixed] = 0.0
        state_none.pos += dt * state_none.vel
        state_none.pos[0] = anchor; state_none.vel[0] = 0.0

        if step % steps_per_frame == 0:
            frames_gt.append(state_gt.pos.copy())
            frames_gnn.append(state_gnn.pos.copy())
            frames_none.append(state_none.pos.copy())
            times.append(step * dt)

            # Rod errors
            def rod_err(state):
                max_err = 0.0
                for e in range(len(state.edges)):
                    i, j = state.edges[e]
                    d = np.linalg.norm(state.pos[j] - state.pos[i])
                    err = abs(d - state.rest_lengths[e]) / state.rest_lengths[e]
                    max_err = max(max_err, err)
                return max_err
            rod_errs_gt.append(rod_err(state_gt))
            rod_errs_gnn.append(rod_err(state_gnn))
            rod_errs_none.append(rod_err(state_none))
            energies_gt.append(compute_energy(state_gt, gravity)["total"])
            energies_gnn.append(compute_energy(state_gnn, gravity)["total"])
            energies_none.append(compute_energy(state_none, gravity)["total"])

    print("\r  done.       ")
    n_frames = len(frames_gt)

    # --- Animation ---
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)

    # Three chain panels (top row)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gnn = fig.add_subplot(gs[0, 1])
    ax_none = fig.add_subplot(gs[0, 2])

    # Metrics panel (bottom row, spanning all columns)
    ax_metric = fig.add_subplot(gs[1, :])

    # Compute axis limits from GT trajectory
    all_pos = np.array(frames_gt)
    margin = 0.5
    x_min, x_max = all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin
    y_min, y_max = all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    for ax, title, color in [
        (ax_gt, "SHAKE (Ground Truth)", "black"),
        (ax_gnn, "GNN Surrogate", "red"),
        (ax_none, "No Correction", "blue"),
    ]:
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_aspect("equal")
        ax.set_title(title, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Chain drawing elements
    edges = np.array([[i, i + 1] for i in range(n_beads - 1)])

    def make_chain_artists(ax, color):
        lines = []
        for _ in edges:
            (line,) = ax.plot([], [], color=color, linewidth=2, alpha=0.7)
            lines.append(line)
        (nodes,) = ax.plot([], [], "o", color=color, markersize=5, zorder=5)
        (anchor_dot,) = ax.plot([], [], "rs", markersize=8, zorder=6)
        return lines, nodes, anchor_dot

    gt_lines, gt_nodes, gt_anchor = make_chain_artists(ax_gt, "black")
    gnn_lines, gnn_nodes, gnn_anchor = make_chain_artists(ax_gnn, "red")
    none_lines, none_nodes, none_anchor = make_chain_artists(ax_none, "blue")

    time_text = ax_gt.text(0.02, 0.98, "", transform=ax_gt.transAxes,
                           va="top", fontsize=10)
    rod_text_gnn = ax_gnn.text(0.02, 0.98, "", transform=ax_gnn.transAxes,
                                va="top", fontsize=9, color="red")
    rod_text_none = ax_none.text(0.02, 0.98, "", transform=ax_none.transAxes,
                                  va="top", fontsize=9, color="blue")

    # Metrics plot
    ax_metric.set_xlabel("Time (s)")
    ax_metric.set_ylabel("Position Error vs GT (m)")
    ax_metric.set_title("Trajectory Error")
    ax_metric.grid(True, alpha=0.3)
    (err_line_gnn,) = ax_metric.plot([], [], "r-", label="GNN", alpha=0.8)
    (err_line_none,) = ax_metric.plot([], [], "b-", label="No corr.", alpha=0.8)
    ax_metric.legend(loc="upper left")

    # Pre-compute position errors
    pos_err_gnn = [np.linalg.norm(np.array(frames_gnn[i]) - np.array(frames_gt[i]),
                                   axis=-1).mean() for i in range(n_frames)]
    pos_err_none = [np.linalg.norm(np.array(frames_none[i]) - np.array(frames_gt[i]),
                                    axis=-1).mean() for i in range(n_frames)]
    max_err = max(max(pos_err_gnn), max(pos_err_none)) * 1.1
    ax_metric.set_xlim(0, times[-1])
    ax_metric.set_ylim(0, max(max_err, 0.01))

    def update_chain(pos, lines, nodes, anchor_dot):
        for e_idx, (i, j) in enumerate(edges):
            lines[e_idx].set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
        nodes.set_data(pos[:, 0], pos[:, 1])
        anchor_dot.set_data([pos[0, 0]], [pos[0, 1]])

    def animate(frame):
        update_chain(frames_gt[frame], gt_lines, gt_nodes, gt_anchor)
        update_chain(frames_gnn[frame], gnn_lines, gnn_nodes, gnn_anchor)
        update_chain(frames_none[frame], none_lines, none_nodes, none_anchor)

        time_text.set_text(f"t = {times[frame]:.3f} s")
        rod_text_gnn.set_text(f"rod err: {rod_errs_gnn[frame]:.2e}")
        rod_text_none.set_text(f"rod err: {rod_errs_none[frame]:.2e}")

        err_line_gnn.set_data(times[:frame+1], pos_err_gnn[:frame+1])
        err_line_none.set_data(times[:frame+1], pos_err_none[:frame+1])

        return (gt_lines + gnn_lines + none_lines +
                [gt_nodes, gnn_nodes, none_nodes,
                 gt_anchor, gnn_anchor, none_anchor,
                 time_text, rod_text_gnn, rod_text_none,
                 err_line_gnn, err_line_none])

    print(f"Animating {n_frames} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=args.interval_ms, blit=True,
    )

    if args.save:
        out_path = os.path.join(os.path.dirname(args.model_path), "evaluation_anim.gif")
        print(f"Saving to {out_path}...")
        anim.save(out_path, writer="pillow", fps=1000 // args.interval_ms)
        print(f"Saved.")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GNN vs SHAKE vs None")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--speed", type=int, default=50,
                        help="Simulation steps per animation frame")
    parser.add_argument("--interval_ms", type=int, default=33,
                        help="Milliseconds between animation frames (~30fps)")
    parser.add_argument("--save", action="store_true",
                        help="Save as GIF instead of showing live")
    args = parser.parse_args()

    run_and_visualize(args)
