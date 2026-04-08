"""Live side-by-side visualization: ground truth physics vs GNN.

Shows two bead chains evolving simultaneously:
  1. Ground truth: Python physics (forces + symplectic Euler, no SHAKE)
  2. GNN: model replaces physics entirely (step_chain)

Supports both the original BeadGNN (MLP) and the new BeadMPGNN.

Usage:
    python bead_visualize.py
    python bead_visualize.py --model_path outputs/bead_best.pt --n_steps 30000
    python bead_visualize.py --model_path outputs/mpgnn_best.pt --model_type mpgnn
    python bead_visualize.py --model_path outputs/mpgnn_best.pt --model_type mpgnn --n_beads 24
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bead_model import BeadGNN
from bead_mpgnn import BeadMPGNN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation import create_chain, compute_forces, compute_energy


def load_model(model_path, model_type, device):
    """Load trained model from checkpoint.

    Args:
        model_type: 'mlp' for BeadGNN, 'mpgnn' for BeadMPGNN
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    if model_type == "mpgnn":
        model = BeadMPGNN(
            node_dim=4, edge_dim=6, output_dim=4,
            hidden_dim=args["hidden_dim"],
            n_message_passes=args.get("n_message_passes", 1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded BeadMPGNN (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.2e}, "
              f"hidden={args['hidden_dim']}, K={args.get('n_message_passes', 1)})")
        return model, None, None, None, None  # normalization is in model buffers
    else:
        model = BeadGNN(
            input_dim=16, output_dim=4,
            hidden_dim=args["hidden_dim"],
            n_layers=args["n_layers"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        x_mean = ckpt["x_mean"].to(device)
        x_std = ckpt["x_std"].to(device)
        y_mean = ckpt["y_mean"].to(device)
        y_std = ckpt["y_std"].to(device)
        print(f"Loaded BeadGNN (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.2e}, "
              f"hidden={args['hidden_dim']}, layers={args['n_layers']})")
        return model, x_mean, x_std, y_mean, y_std


def step_ground_truth(state, dt, gravity):
    """One timestep: forces + symplectic Euler (no SHAKE)."""
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def step_gnn(state, dt, model, x_mean, x_std, y_mean, y_std, device, model_type="mlp"):
    """One timestep: GNN replaces all physics."""
    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest_lengths = torch.from_numpy(state.rest_lengths).float().to(device)

    if model_type == "mpgnn":
        # MPGNN handles everything in step_chain (including normalization)
        with torch.no_grad():
            pos_new, vel_new = model.step_chain(pos, vel, mass, is_fixed, edges, rest_lengths)
        state.pos = pos_new.cpu().numpy()
        state.vel = vel_new.cpu().numpy()
    else:
        # Original BeadGNN: manually build features and normalize
        N = len(state.mass)
        left_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        right_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        left_rest = torch.zeros(N, device=device)
        right_rest = torch.zeros(N, device=device)

        for e, (i, j) in enumerate(edges):
            left_idx[j] = i
            right_idx[i] = j
            left_rest[j] = rest_lengths[e]
            right_rest[i] = rest_lengths[e]

        has_left = left_idx >= 0
        has_right = right_idx >= 0

        safe_left = left_idx.clamp(min=0)
        l_dpos = torch.where(has_left.unsqueeze(1), pos[safe_left] - pos, torch.zeros_like(pos))
        l_dvel = torch.where(has_left.unsqueeze(1), vel[safe_left] - vel, torch.zeros_like(vel))
        l_mass = torch.where(has_left, mass[safe_left], torch.zeros_like(mass))

        safe_right = right_idx.clamp(min=0)
        r_dpos = torch.where(has_right.unsqueeze(1), pos[safe_right] - pos, torch.zeros_like(pos))
        r_dvel = torch.where(has_right.unsqueeze(1), vel[safe_right] - vel, torch.zeros_like(vel))
        r_mass = torch.where(has_right, mass[safe_right], torch.zeros_like(mass))

        x = torch.cat([
            vel, mass.unsqueeze(1), is_fixed.float().unsqueeze(1),
            l_dpos, l_dvel, l_mass.unsqueeze(1), left_rest.unsqueeze(1),
            r_dpos, r_dvel, r_mass.unsqueeze(1), right_rest.unsqueeze(1),
        ], dim=1)  # (N, 16)

        x_norm = (x - x_mean) / x_std
        with torch.no_grad():
            delta_norm = model(x_norm)
        delta = delta_norm * y_std + y_mean

        state.pos += delta[:, :2].cpu().numpy()
        state.vel += delta[:, 2:].cpu().numpy()


def compute_rod_errors(state):
    """Max relative rod length error."""
    max_err = 0.0
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        d = np.linalg.norm(state.pos[j] - state.pos[i])
        err = abs(d - state.rest_lengths[e]) / state.rest_lengths[e]
        max_err = max(max_err, err)
    return max_err


def run_and_visualize(args):
    device = torch.device("cpu")
    n_beads = args.n_beads

    model, x_mean, x_std, y_mean, y_std = load_model(
        args.model_path, args.model_type, device)

    params = dict(
        n_nodes=n_beads, total_length=2.0, total_mass=0.5,
        stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=10.0,
    )
    dt = 0.0001
    gravity = 9.81
    steps_per_frame = args.speed

    state_gt = create_chain(**params)
    state_gnn = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    # Rotate initial chain around anchor by --init_angle degrees
    # 0 = horizontal (default), -90 = pointing straight down, 90 = straight up
    if args.init_angle != 0.0:
        theta = np.deg2rad(args.init_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        for state in (state_gt, state_gnn):
            rel = state.pos - anchor
            state.pos = anchor + rel @ R.T

    # Apply initial velocity to all non-fixed beads
    if args.init_vx != 0.0 or args.init_vy != 0.0:
        v0 = np.array([args.init_vx, args.init_vy])
        for state in (state_gt, state_gnn):
            state.vel[~state.fixed] = v0

    n_frames = args.n_steps // steps_per_frame

    # Pre-simulate to collect frames
    print(f"Simulating {args.n_steps} steps ({n_frames} frames)...")
    frames_gt = [state_gt.pos.copy()]
    frames_gnn = [state_gnn.pos.copy()]
    rod_errs_gt = [0.0]
    rod_errs_gnn = [0.0]
    energies_gt = [compute_energy(state_gt, gravity)["total"]]
    energies_gnn = [compute_energy(state_gnn, gravity)["total"]]
    times = [0.0]

    for step in range(1, args.n_steps + 1):
        if step % max(1, args.n_steps // 10) == 0:
            print(f"\r  {100 * step / args.n_steps:.0f}%", end="", flush=True)

        # Ground truth (no SHAKE)
        step_ground_truth(state_gt, dt, gravity)
        state_gt.pos[0] = anchor
        state_gt.vel[0] = 0.0

        # GNN
        step_gnn(state_gnn, dt, model, x_mean, x_std, y_mean, y_std, device, args.model_type)
        state_gnn.pos[0] = anchor
        state_gnn.vel[0] = 0.0

        if step % steps_per_frame == 0:
            frames_gt.append(state_gt.pos.copy())
            frames_gnn.append(state_gnn.pos.copy())
            times.append(step * dt)
            rod_errs_gt.append(compute_rod_errors(state_gt))
            rod_errs_gnn.append(compute_rod_errors(state_gnn))
            energies_gt.append(compute_energy(state_gt, gravity)["total"])
            energies_gnn.append(compute_energy(state_gnn, gravity)["total"])

    print("\r  done.       ")
    n_frames = len(frames_gt)

    # --- Animation ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gnn = fig.add_subplot(gs[0, 1])
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

    gnn_title = "BeadMPGNN" if args.model_type == "mpgnn" else "BeadGNN"
    for ax, title, color in [
        (ax_gt, f"Ground Truth ({n_beads} beads)", "black"),
        (ax_gnn, f"{gnn_title} ({n_beads} beads)", "red"),
    ]:
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_aspect("equal")
        ax.set_title(title, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

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

    time_text = ax_gt.text(0.02, 0.98, "", transform=ax_gt.transAxes,
                           va="top", fontsize=10)
    rod_text_gt = ax_gt.text(0.02, 0.90, "", transform=ax_gt.transAxes,
                             va="top", fontsize=9, color="black")
    rod_text_gnn = ax_gnn.text(0.02, 0.98, "", transform=ax_gnn.transAxes,
                               va="top", fontsize=9, color="red")

    # Metrics plot
    ax_metric.set_xlabel("Time (s)")
    ax_metric.set_ylabel("Position Error vs GT (m)")
    ax_metric.set_title("Trajectory Error (mean over beads)")
    ax_metric.grid(True, alpha=0.3)
    (err_line,) = ax_metric.plot([], [], "r-", label="GNN vs GT", alpha=0.8)
    ax_metric.legend(loc="upper left")

    # Pre-compute position errors
    pos_err = [np.linalg.norm(np.array(frames_gnn[i]) - np.array(frames_gt[i]),
                              axis=-1).mean() for i in range(n_frames)]
    max_err = max(pos_err) * 1.1
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

        time_text.set_text(f"t = {times[frame]:.3f} s")
        rod_text_gt.set_text(f"rod err: {rod_errs_gt[frame]:.2e}")
        rod_text_gnn.set_text(f"rod err: {rod_errs_gnn[frame]:.2e}")

        err_line.set_data(times[:frame+1], pos_err[:frame+1])

        return (gt_lines + gnn_lines +
                [gt_nodes, gnn_nodes,
                 gt_anchor, gnn_anchor,
                 time_text, rod_text_gt, rod_text_gnn,
                 err_line])

    print(f"Animating {n_frames} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=args.interval_ms, blit=True,
    )

    if args.save:
        out_path = os.path.join(os.path.dirname(args.model_path), "bead_eval_anim.gif")
        print(f"Saving to {out_path}...")
        anim.save(out_path, writer="pillow", fps=1000 // args.interval_ms)
        print(f"Saved.")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BeadGNN vs ground truth")
    parser.add_argument("--model_path", type=str, default="outputs/bead_best.pt")
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--speed", type=int, default=50,
                        help="Simulation steps per animation frame")
    parser.add_argument("--interval_ms", type=int, default=33,
                        help="Milliseconds between animation frames (~30fps)")
    parser.add_argument("--save", action="store_true",
                        help="Save as GIF instead of showing live")
    parser.add_argument("--model_type", type=str, default="mlp",
                        choices=["mlp", "mpgnn"],
                        help="Model type: 'mlp' for BeadGNN, 'mpgnn' for BeadMPGNN")
    parser.add_argument("--n_beads", type=int, default=16,
                        help="Number of beads (MPGNN supports any N)")
    parser.add_argument("--init_angle", type=float, default=0.0,
                        help="Initial chain angle in degrees (0=horizontal, -90=straight down)")
    parser.add_argument("--init_vx", type=float, default=0.0,
                        help="Initial x-velocity for all non-fixed beads (m/s)")
    parser.add_argument("--init_vy", type=float, default=0.0,
                        help="Initial y-velocity for all non-fixed beads (m/s)")
    args = parser.parse_args()

    run_and_visualize(args)
