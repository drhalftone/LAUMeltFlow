"""Three-way Sod comparison: analytical Roe + existing MLP (.pt) + SodMPGNN.

Loads the existing models/flux_mlp_medium_3x128.pt as the baseline MLP,
runs three Sod simulations side by side, and produces:
  - Static 3-panel plot at t = t_final
  - Animation (mp4 or gif) of all three evolving over time
  - MAE table (MLP vs Roe, MPGNN vs Roe)

Usage:
    python sod_compare_three.py
    python sod_compare_three.py --mlp_pt ../models/flux_mlp_medium_3x128.pt
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from sod_mpgnn import SodMPGNN
from sod_simulate_mpgnn import (
    load_mpgnn, sod_initial_conditions, step_mpgnn, step_roe,
)


def build_mlp_from_ckpt(ckpt):
    """Reconstruct the SimpleFluxMLP from a .pt checkpoint's stored config."""
    cfg = ckpt["config"]
    layers = []
    in_dim = cfg["input_dim"]
    act = nn.GELU if cfg.get("activation", "gelu") == "gelu" else nn.ReLU
    for h in cfg["hidden_dims"]:
        layers.append(nn.Linear(in_dim, h))
        layers.append(act())
        in_dim = h
    layers.append(nn.Linear(in_dim, cfg["output_dim"]))
    net = nn.Sequential(*layers)
    # State dict was saved from a module with `self.network = Sequential(...)`,
    # so keys are prefixed with "network." — strip that to match our raw Sequential.
    sd = {k.replace("network.", "", 1): v
          for k, v in ckpt["model_state_dict"].items()}
    net.load_state_dict(sd)
    net.eval()
    return net, cfg


def load_mlp_baseline(pt_path, device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    net, cfg = build_mlp_from_ckpt(ckpt)
    net = net.to(device)
    stats = {k: torch.as_tensor(v, device=device).float()
             for k, v in ckpt["stats"].items()
             if k in ("X_mean", "X_std", "Y_mean", "Y_std")}
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Loaded MLP baseline: input_dim={cfg['input_dim']}, "
          f"hidden={cfg['hidden_dims']}, has_gamma={cfg.get('has_gamma', False)}, "
          f"{n_params:,} params")
    return net, stats, cfg


def step_mlp(rho, u, p, dx, dt, gamma, net, stats, cfg, device):
    """One FV timestep using the baseline MLP for flux."""
    nx = rho.shape[0]
    # Form (nx+1, 6 or 7) of (L, R) states at each interface, with
    # transmissive boundary cells duplicated.
    rho_pad = np.concatenate([[rho[0]], rho, [rho[-1]]])
    u_pad = np.concatenate([[u[0]], u, [u[-1]]])
    p_pad = np.concatenate([[p[0]], p, [p[-1]]])

    states_L = np.stack([rho_pad[:-1], u_pad[:-1], p_pad[:-1]], axis=1)  # (nx+1, 3)
    states_R = np.stack([rho_pad[1:], u_pad[1:], p_pad[1:]], axis=1)
    X = np.concatenate([states_L, states_R], axis=1)  # (nx+1, 6)

    if cfg.get("has_gamma", False):
        gam_col = np.full((nx + 1, 1), gamma, dtype=np.float32)
        X = np.concatenate([X, gam_col], axis=1)  # (nx+1, 7)

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    X_norm = (X_t - stats["X_mean"]) / stats["X_std"]
    with torch.no_grad():
        Y_norm = net(X_norm)
    Y = Y_norm * stats["Y_std"] + stats["Y_mean"]
    flux = Y.cpu().numpy()  # (nx+1, 3)

    E = p / (gamma - 1) + 0.5 * rho * u ** 2
    W = np.stack([rho, rho * u, E], axis=1)
    W_new = W - dt / dx * (flux[1:] - flux[:-1])
    rho_n = W_new[:, 0]
    u_n = W_new[:, 1] / rho_n
    E_n = W_new[:, 2]
    p_n = (gamma - 1) * (E_n - 0.5 * rho_n * u_n ** 2)
    return rho_n, u_n, p_n


def run_and_record(stepper, nx, t_final, cfl, gamma, sample_every, label):
    x, dx, rho, u, p = sod_initial_conditions(nx)
    t, n_steps = 0.0, 0
    times, rhos, us, ps = [t], [rho.copy()], [u.copy()], [p.copy()]
    t0 = time.time()
    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + a)
        if t + dt > t_final:
            dt = t_final - t
        rho, u, p = stepper(rho, u, p, dx, dt)
        if np.any(rho <= 0) or np.any(p <= 0) or np.any(np.isnan(rho)):
            print(f"  [{label}] invalid state at t={t:.4e}, stopping")
            break
        t += dt
        n_steps += 1
        if n_steps % sample_every == 0 or t >= t_final:
            times.append(t)
            rhos.append(rho.copy())
            us.append(u.copy())
            ps.append(p.copy())
    print(f"  [{label}] {n_steps} steps, {time.time()-t0:.2f} s, "
          f"{len(times)} frames")
    return dict(x=x, t=np.array(times),
                rho=np.stack(rhos), u=np.stack(us), p=np.stack(ps))


def static_plot(results, save_path, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                         "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = [("rho", "Density [kg/m³]"),
              ("u", "Velocity [m/s]"),
              ("p", "Pressure [Pa]")]
    styles = {"Roe (reference)": ("-", "k", 2.0),
              "MLP (baseline)": (":", "tab:orange", 2.0),
              "MPGNN": ("--", "tab:blue", 2.0)}
    for ax, (field, ylabel) in zip(axes, fields):
        for name, r in results.items():
            style, color, lw = styles.get(name, ("-", None, 1.5))
            ax.plot(r["x"], r[field][-1], style, color=color,
                    label=name, linewidth=lw)
        ax.set_xlabel("x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" [")[0])
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.suptitle(
        f"Sod shock tube at t = {t_final*1e3:.3f} ms", fontsize=15
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def animate_three(results, save_path, fps, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                         "legend.fontsize": 11})

    n_frames = min(r["t"].shape[0] for r in results.values())
    x = next(iter(results.values()))["x"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = [("rho", "Density [kg/m³]"),
              ("u", "Velocity [m/s]"),
              ("p", "Pressure [Pa]")]
    styles = {"Roe (reference)": ("-", "k", 2.0),
              "MLP (baseline)": (":", "tab:orange", 2.0),
              "MPGNN": ("--", "tab:blue", 2.0)}

    lines = {name: [] for name in results}
    for ax, (field, ylabel) in zip(axes, fields):
        for name, r in results.items():
            style, color, lw = styles[name]
            ln, = ax.plot(x, r[field][0], style, color=color,
                          label=name, linewidth=lw)
            lines[name].append(ln)
        ax.set_xlabel("x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" [")[0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ymin = min(r[field].min() for r in results.values())
        ymax = max(r[field].max() for r in results.values())
        pad = 0.05 * (ymax - ymin + 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)

    suptitle = fig.suptitle("", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        for name, r in results.items():
            for ax_idx, (field, _) in enumerate(fields):
                lines[name][ax_idx].set_ydata(r[field][frame])
        suptitle.set_text(
            f"Sod shock tube: Roe vs MLP vs MPGNN  —  "
            f"t = {next(iter(results.values()))['t'][frame]*1e3:.3f} ms"
        )
        return tuple(ln for lns in lines.values() for ln in lns) + (suptitle,)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved {save_path}")
    except Exception as e:
        gif_path = os.path.splitext(save_path)[0] + ".gif"
        print(f"ffmpeg unavailable ({e}); falling back to .gif")
        anim.save(gif_path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved {gif_path}")
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mpgnn = load_mpgnn(args.mpgnn_ckpt, device)
    mlp_net, mlp_stats, mlp_cfg = load_mlp_baseline(args.mlp_pt, device)

    def step_r(rho, u, p, dx, dt):
        return step_roe(rho, u, p, dx, dt, args.gamma)

    def step_mp(rho, u, p, dx, dt):
        return step_mpgnn(rho, u, p, dx, dt, args.gamma, mpgnn, device)

    def step_ml(rho, u, p, dx, dt):
        return step_mlp(rho, u, p, dx, dt, args.gamma,
                        mlp_net, mlp_stats, mlp_cfg, device)

    results = {}
    print("\nRoe (reference):")
    results["Roe (reference)"] = run_and_record(
        step_r, args.nx, args.t_final, args.cfl, args.gamma,
        args.sample_every, "Roe")
    print("\nMLP (baseline):")
    results["MLP (baseline)"] = run_and_record(
        step_ml, args.nx, args.t_final, args.cfl, args.gamma,
        args.sample_every, "MLP")
    print("\nMPGNN:")
    results["MPGNN"] = run_and_record(
        step_mp, args.nx, args.t_final, args.cfl, args.gamma,
        args.sample_every, "MPGNN")

    print("\n=== MAE vs analytical Roe (at t_final) ===")
    ref = results["Roe (reference)"]
    print(f"{'method':<22}  {'rho [kg/m^3]':<14}  {'u [m/s]':<10}  {'p [Pa]':<10}")
    for name in ("MLP (baseline)", "MPGNN"):
        r = results[name]
        rho_mae = float(np.mean(np.abs(r["rho"][-1] - ref["rho"][-1])))
        u_mae = float(np.mean(np.abs(r["u"][-1] - ref["u"][-1])))
        p_mae = float(np.mean(np.abs(r["p"][-1] - ref["p"][-1])))
        print(f"{name:<22}  {rho_mae:<14.4e}  {u_mae:<10.4e}  {p_mae:<10.4e}")

    os.makedirs(os.path.dirname(args.static_out) or ".", exist_ok=True)
    static_plot(results, args.static_out, args.t_final)
    animate_three(results, args.anim_out, args.fps, args.t_final)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mpgnn_ckpt", type=str,
                   default="outputs/sod_mpgnn_best.pt")
    p.add_argument("--mlp_pt", type=str,
                   default="../models/flux_mlp_medium_3x128.pt")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--t_final", type=float, default=7.5e-4)
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--static_out", type=str,
                   default="outputs/sod_three_way.png")
    p.add_argument("--anim_out", type=str,
                   default="outputs/sod_three_way.mp4")
    args = p.parse_args()
    main(args)
