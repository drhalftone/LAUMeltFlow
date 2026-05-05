"""Head-to-head comparison: BeadGNN (MLP) vs BeadMPGNN on the same chain.

Runs ground truth, original MLP model, and message-passing GNN simultaneously
from identical initial conditions, plots position error over time for both
models, and shows a 3-panel side-by-side animation.

Usage:
    python compare_models.py
    python compare_models.py --n_beads 16 --n_steps 30000
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
from simulation import create_chain, compute_forces


def load_mlp(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = BeadGNN(input_dim=16, output_dim=4,
                    hidden_dim=a["hidden_dim"], n_layers=a["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["x_mean"].to(device), ckpt["x_std"].to(device), \
           ckpt["y_mean"].to(device), ckpt["y_std"].to(device)


def load_mpgnn(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = BeadMPGNN(node_dim=4, edge_dim=6, output_dim=4,
                      hidden_dim=a["hidden_dim"],
                      n_message_passes=a.get("n_message_passes", 1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def step_gt(state, dt, gravity):
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def step_mlp(state, model, x_mean, x_std, y_mean, y_std, device):
    N = len(state.mass)
    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest = torch.from_numpy(state.rest_lengths).float().to(device)

    left_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    right_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    l_rest = torch.zeros(N, device=device)
    r_rest = torch.zeros(N, device=device)
    for e, (i, j) in enumerate(edges):
        left_idx[j] = i
        right_idx[i] = j
        l_rest[j] = rest[e]
        r_rest[i] = rest[e]
    has_left = left_idx >= 0
    has_right = right_idx >= 0
    sl = left_idx.clamp(min=0)
    sr = right_idx.clamp(min=0)
    l_dpos = torch.where(has_left.unsqueeze(1), pos[sl] - pos, torch.zeros_like(pos))
    l_dvel = torch.where(has_left.unsqueeze(1), vel[sl] - vel, torch.zeros_like(vel))
    l_mass = torch.where(has_left, mass[sl], torch.zeros_like(mass))
    r_dpos = torch.where(has_right.unsqueeze(1), pos[sr] - pos, torch.zeros_like(pos))
    r_dvel = torch.where(has_right.unsqueeze(1), vel[sr] - vel, torch.zeros_like(vel))
    r_mass = torch.where(has_right, mass[sr], torch.zeros_like(mass))

    x = torch.cat([
        vel, mass.unsqueeze(1), is_fixed.float().unsqueeze(1),
        l_dpos, l_dvel, l_mass.unsqueeze(1), l_rest.unsqueeze(1),
        r_dpos, r_dvel, r_mass.unsqueeze(1), r_rest.unsqueeze(1),
    ], dim=1)
    x_norm = (x - x_mean) / x_std
    with torch.no_grad():
        delta = model(x_norm) * y_std + y_mean
    state.pos += delta[:, :2].cpu().numpy()
    state.vel += delta[:, 2:].cpu().numpy()


def step_mpgnn(state, model, device):
    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest = torch.from_numpy(state.rest_lengths).float().to(device)
    with torch.no_grad():
        pos_new, vel_new = model.step_chain(pos, vel, mass, is_fixed, edges, rest)
    state.pos = pos_new.cpu().numpy()
    state.vel = vel_new.cpu().numpy()


def main(args):
    device = torch.device("cpu")

    print(f"Loading models...")
    mlp, x_mean, x_std, y_mean, y_std = load_mlp(args.mlp_path, device)
    mpgnn = load_mpgnn(args.mpgnn_path, device)
    print(f"  BeadGNN (MLP):    {sum(p.numel() for p in mlp.parameters()):,} params")
    print(f"  BeadMPGNN:        {sum(p.numel() for p in mpgnn.parameters()):,} params")

    params = dict(
        n_nodes=args.n_beads, total_length=2.0, total_mass=0.5,
        stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=10.0,
    )
    dt = 0.0001
    gravity = 9.81

    state_gt = create_chain(**params)
    state_mlp = create_chain(**params)
    state_mpgnn = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    sample_every = max(1, args.n_steps // 200)
    times, err_mlp, err_mpgnn = [0.0], [0.0], [0.0]
    frames_gt, frames_mlp, frames_mpgnn = [state_gt.pos.copy()], \
                                           [state_mlp.pos.copy()], \
                                           [state_mpgnn.pos.copy()]

    print(f"Running {args.n_steps} steps on N={args.n_beads}...")
    for step in range(1, args.n_steps + 1):
        if step % max(1, args.n_steps // 20) == 0:
            print(f"\r  {100*step/args.n_steps:.0f}%", end="", flush=True)

        step_gt(state_gt, dt, gravity)
        state_gt.pos[0] = anchor; state_gt.vel[0] = 0.0

        step_mlp(state_mlp, mlp, x_mean, x_std, y_mean, y_std, device)
        state_mlp.pos[0] = anchor; state_mlp.vel[0] = 0.0

        step_mpgnn(state_mpgnn, mpgnn, device)
        state_mpgnn.pos[0] = anchor; state_mpgnn.vel[0] = 0.0

        if step % sample_every == 0:
            times.append(step * dt)
            err_mlp.append(np.linalg.norm(state_mlp.pos - state_gt.pos, axis=-1).mean())
            err_mpgnn.append(np.linalg.norm(state_mpgnn.pos - state_gt.pos, axis=-1).mean())
            frames_gt.append(state_gt.pos.copy())
            frames_mlp.append(state_mlp.pos.copy())
            frames_mpgnn.append(state_mpgnn.pos.copy())

    print("\r  done.       ")

    # --- Print summary ---
    print(f"\n{'='*55}")
    print(f"Final position error (mean over beads, m):")
    print(f"  BeadGNN (MLP):  {err_mlp[-1]:.4e}")
    print(f"  BeadMPGNN:      {err_mpgnn[-1]:.4e}")
    if err_mlp[-1] > 0:
        ratio = err_mpgnn[-1] / err_mlp[-1]
        print(f"  Ratio (MPGNN/MLP): {ratio:.2f}x  "
              f"({'MPGNN better' if ratio < 1 else 'MLP better'})")
    print(f"{'='*55}\n")

    # --- Static comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: error over time (semilog)
    ax = axes[0, 0]
    ax.semilogy(times, err_mlp, label="BeadGNN (MLP)", color="tab:blue", linewidth=2)
    ax.semilogy(times, err_mpgnn, label="BeadMPGNN", color="tab:red", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title(f"Error vs Ground Truth (N={args.n_beads} beads)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: error over time (linear)
    ax = axes[0, 1]
    ax.plot(times, err_mlp, label="BeadGNN (MLP)", color="tab:blue", linewidth=2)
    ax.plot(times, err_mpgnn, label="BeadMPGNN", color="tab:red", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title("Same data, linear scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: final chain positions
    ax = axes[1, 0]
    gt_final = np.array(frames_gt[-1])
    mlp_final = np.array(frames_mlp[-1])
    mpgnn_final = np.array(frames_mpgnn[-1])
    ax.plot(gt_final[:, 0], gt_final[:, 1], "k-o", label="Ground Truth", linewidth=2, markersize=4)
    ax.plot(mlp_final[:, 0], mlp_final[:, 1], "b-o", label="BeadGNN", linewidth=1.5, markersize=3, alpha=0.7)
    ax.plot(mpgnn_final[:, 0], mpgnn_final[:, 1], "r-o", label="BeadMPGNN", linewidth=1.5, markersize=3, alpha=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Final positions at t={times[-1]:.2f}s")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: bar comparison of final + max errors
    ax = axes[1, 1]
    metrics = {
        "Final err": [err_mlp[-1], err_mpgnn[-1]],
        "Max err":   [max(err_mlp), max(err_mpgnn)],
        "Mean err":  [np.mean(err_mlp), np.mean(err_mpgnn)],
    }
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, [v[0] for v in metrics.values()], width, label="MLP", color="tab:blue")
    ax.bar(x + width/2, [v[1] for v in metrics.values()], width, label="MPGNN", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error metrics summary")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir = os.path.dirname(args.mpgnn_path)
    plot_path = os.path.join(out_dir, f"compare_N{args.n_beads}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to {plot_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BeadGNN vs BeadMPGNN")
    parser.add_argument("--mlp_path", type=str, default="outputs/bead_best.pt")
    parser.add_argument("--mpgnn_path", type=str, default="outputs/mpgnn_best.pt")
    parser.add_argument("--n_beads", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=30000)
    args = parser.parse_args()
    main(args)
