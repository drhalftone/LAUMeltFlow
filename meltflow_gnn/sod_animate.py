"""Animate the 1D Sod shock tube: MPGNN vs analytical Roe over time.

Runs both simulations side by side, records every Nth timestep, and
writes an MP4 with 3 panels (density / velocity / pressure) evolving
from t = 0 to t = t_final.

Usage:
    python sod_animate.py
    python sod_animate.py --output sod_evolution.mp4 --fps 30
"""

import os
import argparse
import time
import numpy as np
import torch

from sod_mpgnn import SodMPGNN
from sod_simulate_mpgnn import (
    load_mpgnn, sod_initial_conditions, step_mpgnn, step_roe
)


def run_and_record(stepper, nx, t_final, cfl, gamma, sample_every, label):
    """Run simulation, snapshotting every `sample_every` steps. Returns snapshots."""
    x, dx, rho, u, p = sod_initial_conditions(nx)
    t = 0.0
    n_steps = 0

    times = [t]
    rhos = [rho.copy()]
    us = [u.copy()]
    ps = [p.copy()]

    t0 = time.time()
    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed
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

    elapsed = time.time() - t0
    print(f"  [{label}] {n_steps} steps in {elapsed:.2f} s, "
          f"{len(times)} frames")
    return dict(x=x, t=np.array(times),
                rho=np.stack(rhos), u=np.stack(us), p=np.stack(ps))


def animate(roe, mpgnn, save_path, fps, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                         "axes.labelsize": 13, "legend.fontsize": 11})

    n_frames = min(roe["t"].shape[0], mpgnn["t"].shape[0])
    x = roe["x"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = [("rho", "Density [kg/m³]"),
              ("u", "Velocity [m/s]"),
              ("p", "Pressure [Pa]")]

    lines_roe = []
    lines_mpgnn = []
    for ax, (field, ylabel) in zip(axes, fields):
        line_r, = ax.plot(x, roe[field][0], "k-", linewidth=2,
                          label="Roe (reference)")
        line_m, = ax.plot(x, mpgnn[field][0], "--", color="tab:blue",
                          linewidth=2, label="MPGNN")
        lines_roe.append(line_r)
        lines_mpgnn.append(line_m)
        ax.set_xlabel("x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" [")[0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Fix y-limits across all frames so axes don't jump around
        ymin = min(roe[field].min(), mpgnn[field].min())
        ymax = max(roe[field].max(), mpgnn[field].max())
        pad = 0.05 * (ymax - ymin + 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)

    suptitle = fig.suptitle("", fontsize=15)

    def update(frame):
        for ax_idx, (field, _) in enumerate(fields):
            lines_roe[ax_idx].set_ydata(roe[field][frame])
            lines_mpgnn[ax_idx].set_ydata(mpgnn[field][frame])
        suptitle.set_text(f"{title}  —  t = {roe['t'][frame]*1e3:.3f} ms "
                          f"(frame {frame+1}/{n_frames})")
        return (*lines_roe, *lines_mpgnn, suptitle)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )

    # Try ffmpeg first, fall back to gif via pillow if not available
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved {save_path} (mp4, fps={fps})")
    except Exception as e:
        print(f"ffmpeg failed ({e}), falling back to .gif")
        gif_path = os.path.splitext(save_path)[0] + ".gif"
        anim.save(gif_path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved {gif_path}")

    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_mpgnn(args.mpgnn_ckpt, device)

    def step_mp(rho, u, p, dx, dt):
        return step_mpgnn(rho, u, p, dx, dt, args.gamma, model, device)

    def step_r(rho, u, p, dx, dt):
        return step_roe(rho, u, p, dx, dt, args.gamma)

    print("\nRoe (reference):")
    roe = run_and_record(step_r, args.nx, args.t_final, args.cfl,
                         args.gamma, args.sample_every, "Roe")

    print("\nMPGNN:")
    mp = run_and_record(step_mp, args.nx, args.t_final, args.cfl,
                        args.gamma, args.sample_every, "MPGNN")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    animate(roe, mp, args.output, args.fps,
            title="Sod shock tube: MPGNN vs analytical Roe")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mpgnn_ckpt", type=str,
                   default="outputs/sod_mpgnn_best.pt")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--t_final", type=float, default=7.5e-4)
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--sample_every", type=int, default=2,
                   help="Record a frame every N timesteps")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output", type=str, default="outputs/sod_evolution.mp4")
    args = p.parse_args()
    main(args)
