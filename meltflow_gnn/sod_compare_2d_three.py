"""Three-way 2D Sod comparison: analytical Roe + 2D MLP baseline + SodMPGNN_2D.

Runs all three on the 2D Sod tube, produces:
  - Static figure: density heatmaps for all three + error maps + 1D slice
  - Animation: density evolution side by side
  - MAE table

Usage:
    python sod_compare_2d_three.py
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from sod_mpgnn_2d import SodMPGNN_2D
from sod_train_mpgnn_2d import build_edge_attrs
from sod_simulate_mpgnn_2d import (
    init_sod_2d, cons_to_prim, step_roe_2d, step_mpgnn_2d,
    load_mpgnn_2d, run_sim,
)
from train_uniform_2d_cuda import FluxMLP2D


def load_mlp_2d_npz(npz_path, device):
    """Load the 2D MLP from an .npz checkpoint (saved by train_uniform_2d_cuda)."""
    data = np.load(npz_path, allow_pickle=True)
    hidden_dim = int(data["hidden_dim"]) if "hidden_dim" in data.files else 256
    n_layers = int(data["n_layers"]) if "n_layers" in data.files else 5
    net = FluxMLP2D(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    sd = {}
    for k in data.files:
        if k.startswith("network."):
            sd[k] = torch.from_numpy(data[k]).to(device)
    if not sd:
        # Maybe the .pt has the state — try loading the .pt next to it
        pt_path = os.path.splitext(npz_path)[0] + ".pt"
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
    net.load_state_dict(sd)
    net.eval()
    stats = {
        "X_mean": torch.from_numpy(data["X_mean"]).float().to(device),
        "X_std": torch.from_numpy(data["X_std"]).float().to(device),
        "Y_mean": torch.from_numpy(data["Y_mean"]).float().to(device),
        "Y_std": torch.from_numpy(data["Y_std"]).float().to(device),
    }
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Loaded 2D MLP baseline: {hidden_dim}x{n_layers}, {n_params:,} params")
    return net, stats


def step_mlp_2d(rho, u, v, p, dx, dy, dt, gamma, net, stats, device):
    """One FV timestep using the 2D MLP for per-cell 4-flux prediction."""
    nx, ny = rho.shape

    rho_p = np.pad(rho, 1, mode="edge")
    u_p = np.pad(u, 1, mode="edge")
    v_p = np.pad(v, 1, mode="edge")
    p_p = np.pad(p, 1, mode="edge")

    states = np.stack([rho_p, u_p, v_p, p_p], axis=-1)
    Wn = states[0:nx,     1:ny+1]
    Cn = states[1:nx+1,   1:ny+1]
    En = states[2:nx+2,   1:ny+1]
    Sn = states[1:nx+1,   0:ny]
    Nn = states[1:nx+1,   2:ny+2]
    # MLP expects: W, C, E, S, N concatenated -> 20 features
    stencils = np.concatenate([Wn, Cn, En, Sn, Nn], axis=-1)  # (nx, ny, 20)
    stencils_flat = stencils.reshape(-1, 20).astype(np.float32)

    X = torch.from_numpy(stencils_flat).to(device)
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    with torch.no_grad():
        Y_norm = net(X_norm)
    Y = Y_norm * stats["Y_std"] + stats["Y_mean"]
    flux = Y.cpu().numpy().reshape(nx, ny, 4, 4)
    # MLP output ordering matches sampler: F_w, F_e, G_s, G_n (each 4-D)

    F_w = flux[:, :, 0, :]
    F_e = flux[:, :, 1, :]
    G_s = flux[:, :, 2, :]
    G_n = flux[:, :, 3, :]

    F_x = np.zeros((nx + 1, ny, 4))
    F_x[0, :, :] = F_w[0, :, :]
    F_x[1:, :, :] = F_e[:, :, :]
    F_x = np.transpose(F_x, (2, 0, 1))

    G_y = np.zeros((nx, ny + 1, 4))
    G_y[:, 0, :] = G_s[:, 0, :]
    G_y[:, 1:, :] = G_n[:, :, :]
    G_y = np.transpose(G_y, (2, 0, 1))

    E_arr = p / (gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2)
    W_arr = np.stack([rho, rho * u, rho * v, E_arr], axis=0)
    W_new = W_arr.copy()
    W_new -= dt / dx * (F_x[:, 1:, :] - F_x[:, :-1, :])
    W_new -= dt / dy * (G_y[:, :, 1:] - G_y[:, :, :-1])
    return cons_to_prim(W_new, gamma)


def plot_three_way(roe, mlp, mpgnn, save_path, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "legend.fontsize": 10})

    x = roe["x"]; y = roe["y"]
    extent = [x[0], x[-1], y[0], y[-1]]

    rho_min = min(roe["rho"].min(), mlp["rho"].min(), mpgnn["rho"].min())
    rho_max = max(roe["rho"].max(), mlp["rho"].max(), mpgnn["rho"].max())

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    # Row 1: density heatmaps for Roe, MLP, MPGNN
    for ax, (data, title) in zip(axes[0],
                                  [(roe["rho"], "Density - Roe (reference)"),
                                   (mlp["rho"], "Density - MLP (baseline)"),
                                   (mpgnn["rho"], "Density - MPGNN")]):
        im = ax.imshow(data.T, origin="lower", extent=extent, aspect="auto",
                       cmap="viridis", vmin=rho_min, vmax=rho_max)
        ax.set_title(title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax)

    # Row 2: |Error| heatmaps (MLP - Roe) and (MPGNN - Roe), plus shared colorbar
    err_mlp = np.abs(mlp["rho"] - roe["rho"])
    err_mp = np.abs(mpgnn["rho"] - roe["rho"])
    err_max = max(err_mlp.max(), err_mp.max())
    axes[1, 0].axis("off")  # empty
    for ax, (data, title) in zip(axes[1, 1:],
                                  [(err_mlp, "|Error| - MLP vs Roe"),
                                   (err_mp, "|Error| - MPGNN vs Roe")]):
        im = ax.imshow(data.T, origin="lower", extent=extent, aspect="auto",
                       cmap="hot", vmin=0, vmax=err_max)
        ax.set_title(title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax)

    # Row 3: 1D slice along y=mid (rho, u, p)
    j_mid = roe["rho"].shape[1] // 2
    for ax, (field, ylabel) in zip(axes[2],
                                    [("rho", "Density [kg/m³]"),
                                     ("u",   "u [m/s]"),
                                     ("p",   "Pressure [Pa]")]):
        ax.plot(x, roe[field][:, j_mid], "k-", linewidth=2,
                label="Roe (reference)")
        ax.plot(x, mlp[field][:, j_mid], ":", color="tab:orange",
                linewidth=2, label="MLP")
        ax.plot(x, mpgnn[field][:, j_mid], "--", color="tab:blue",
                linewidth=2, label="MPGNN")
        ax.set_xlabel("x [m]"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' [')[0]} slice at y={y[j_mid]:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.suptitle(f"2D Sod Shock Tube: Roe vs MLP vs MPGNN at t = "
                 f"{t_final*1e3:.3f} ms", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def animate_three(roe, mlp, mpgnn, save_path, fps, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    n_frames = min(roe["rho_frames"].shape[0],
                   mlp["rho_frames"].shape[0],
                   mpgnn["rho_frames"].shape[0])
    x = roe["x"]; y = roe["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    rho_min = min(roe["rho_frames"].min(), mlp["rho_frames"].min(),
                  mpgnn["rho_frames"].min())
    rho_max = max(roe["rho_frames"].max(), mlp["rho_frames"].max(),
                  mpgnn["rho_frames"].max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ims = []
    for ax, (label, data) in zip(axes,
                                  [("Roe (reference)", roe),
                                   ("MLP (baseline)", mlp),
                                   ("MPGNN", mpgnn)]):
        im = ax.imshow(data["rho_frames"][0].T, origin="lower",
                       extent=extent, aspect="auto", cmap="viridis",
                       vmin=rho_min, vmax=rho_max)
        ax.set_title(label)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax)
        ims.append(im)

    suptitle = fig.suptitle("", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        ims[0].set_data(roe["rho_frames"][frame].T)
        ims[1].set_data(mlp["rho_frames"][frame].T)
        ims[2].set_data(mpgnn["rho_frames"][frame].T)
        suptitle.set_text(
            f"2D Sod density at t = {roe['t'][frame]*1e3:.3f} ms"
        )
        return (*ims, suptitle)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved {save_path}")
    except Exception as e:
        gif = os.path.splitext(save_path)[0] + ".gif"
        print(f"ffmpeg unavailable ({e}); writing .gif")
        anim.save(gif, writer="pillow", fps=fps, dpi=100)
        print(f"Saved {gif}")
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mpgnn = load_mpgnn_2d(args.mpgnn_ckpt, device)
    mlp_net, mlp_stats = load_mlp_2d_npz(args.mlp_npz, device)

    _, _, dx, dy, _, _, _, _ = init_sod_2d(args.nx, args.ny)
    edge_cache = build_edge_attrs(args.nx * args.ny, dx, dy)

    def step_r(rho, u, v, p, dx_, dy_, dt):
        return step_roe_2d(rho, u, v, p, dx_, dy_, dt, args.gamma)

    def step_mp(rho, u, v, p, dx_, dy_, dt):
        return step_mpgnn_2d(rho, u, v, p, dx_, dy_, dt, args.gamma,
                             mpgnn, edge_cache, device)

    def step_ml(rho, u, v, p, dx_, dy_, dt):
        return step_mlp_2d(rho, u, v, p, dx_, dy_, dt, args.gamma,
                           mlp_net, mlp_stats, device)

    print("\nRoe (reference):")
    roe = run_sim(step_r, args.nx, args.ny, args.t_final, args.cfl,
                  args.gamma, "Roe", sample_every=args.sample_every)
    print("\nMLP (baseline):")
    mlp = run_sim(step_ml, args.nx, args.ny, args.t_final, args.cfl,
                  args.gamma, "MLP", sample_every=args.sample_every)
    print("\nMPGNN:")
    mp = run_sim(step_mp, args.nx, args.ny, args.t_final, args.cfl,
                 args.gamma, "MPGNN", sample_every=args.sample_every)

    print("\n=== MAE vs analytical Roe (at t_final) ===")
    print(f"{'method':<22}  {'rho [kg/m^3]':<14}  {'u [m/s]':<10}  "
          f"{'v [m/s]':<10}  {'p [Pa]':<10}")
    for name, r in [("MLP (baseline)", mlp), ("MPGNN", mp)]:
        rho_mae = float(np.mean(np.abs(r["rho"] - roe["rho"])))
        u_mae = float(np.mean(np.abs(r["u"] - roe["u"])))
        v_mae = float(np.mean(np.abs(r["v"] - roe["v"])))
        p_mae = float(np.mean(np.abs(r["p"] - roe["p"])))
        print(f"{name:<22}  {rho_mae:<14.4e}  {u_mae:<10.4e}  "
              f"{v_mae:<10.4e}  {p_mae:<10.4e}")

    os.makedirs(os.path.dirname(args.static_out) or ".", exist_ok=True)
    plot_three_way(roe, mlp, mp, args.static_out, args.t_final)
    animate_three(roe, mlp, mp, args.anim_out, args.fps, args.t_final)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mpgnn_ckpt", type=str,
                   default="outputs/sod_mpgnn_2d_best.pt")
    p.add_argument("--mlp_npz", type=str,
                   default="outputs/flux_model_2d_baseline.npz")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=50)
    p.add_argument("--t_final", type=float, default=4.0e-4)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--static_out", type=str,
                   default="outputs/sod2d_three_way.png")
    p.add_argument("--anim_out", type=str,
                   default="outputs/sod2d_three_way.mp4")
    args = p.parse_args()
    main(args)
