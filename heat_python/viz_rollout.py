"""Visualize a GNN rollout vs the solver (from eval_rollout's saved .npz).

Produces:
  1. A static 3-panel space-time heatmap PNG: solver T(x,t), GNN T(x,t), and
     |error|(x,t) -- the "paired error-map" figure.
  2. An animated temperature profile (GIF): T vs depth, solver vs GNN, evolving
     over the trajectory -- the heat-shield analog of the whip animation. You
     can watch the GNN track early then drift.

    python -m heat_python.viz_rollout --rollout heat_python/data/rollout_aw1.npz
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .case import load_case


def _depth_axis(case_dir="heat_2026-04-11_1837/examples/aw1", n=None) -> np.ndarray:
    """Interior cell-center positions [mm]. Uses the case's L0 and either its
    NBRN or an explicit n (for downsampled/overridden meshes). x=0 is the cold
    back; x=L0 is the hot wall."""
    repo = Path(__file__).resolve().parents[1]
    case = load_case(repo / case_dir / "heat.case")
    n = n if n is not None else case.nbrn
    dx = case.l0 / n
    return (np.arange(n) + 0.5) * dx * 1e3      # mm


def static_heatmap(time, T_gt, T_pred, x_mm, out_png, label=""):
    gt = T_gt[:, 1:-1]                            # interior (S, n)
    pr = T_pred[:, 1:-1]
    err = np.abs(pr - gt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    extent = [x_mm[0], x_mm[-1], time[-1], time[0]]
    vmax = max(gt.max(), pr.max())
    for ax, data, title, cmap, vm in [
            (axes[0], gt, "Solver (truth)", "inferno", vmax),
            (axes[1], pr, "GNN surrogate", "inferno", vmax),
            (axes[2], err, "|error|", "viridis", None)]:
        im = ax.imshow(data, aspect="auto", extent=extent, cmap=cmap,
                       vmin=0 if vm else None, vmax=vm)
        ax.set_title(title)
        ax.set_xlabel("position x [mm]  (hot wall at right)")
        fig.colorbar(im, ax=ax, label="T [K]" if vm else "|ΔT| [K]")
    axes[0].set_ylabel("time [s]")
    title = (f"{label}: rollout vs solver (space-time temperature)" if label
             else "Heat-shield rollout: GNN vs solver (space-time temperature)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {out_png}")


def profile_animation(time, T_gt, T_pred, x_mm, out_gif, n_frames=160, fps=20,
                      label=""):
    gt = T_gt[:, 1:-1]
    pr = T_pred[:, 1:-1]
    S = gt.shape[0]
    idx = np.linspace(0, S - 1, min(n_frames, S)).astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    (l_gt,) = ax.plot([], [], "k-", lw=2.5, label="solver (truth)")
    (l_pr,) = ax.plot([], [], "r--", lw=2.0, label="GNN surrogate")
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(min(gt.min(), pr.min()) - 30, max(gt.max(), pr.max()) + 50)
    ax.set_xlabel("position x [mm]  (hot wall at right)")
    ax.set_ylabel("temperature [K]")
    ax.legend(loc="upper left")
    if label:
        ax.set_title(label)
    txt = ax.text(0.02, 0.90, "", transform=ax.transAxes)

    def update(k):
        s = idx[k]
        l_gt.set_data(x_mm, gt[s])
        l_pr.set_data(x_mm, pr[s])
        e = np.abs(pr[s] - gt[s])
        txt.set_text(f"t = {time[s]:5.1f} s   mean |ΔT| = {e.mean():.1f} K   "
                     f"max = {e.max():.1f} K")
        return l_gt, l_pr, txt

    anim = FuncAnimation(fig, update, frames=len(idx), blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {out_gif}")


def main(args):
    d = dict(np.load(args.rollout))
    n = d["T_gt"].shape[1] - 2                    # interior cells from the data
    x_mm = _depth_axis(args.case_dir, n)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    static_heatmap(d["time"], d["T_gt"], d["T_pred"], x_mm,
                   out / "spacetime.png", label=args.label)
    profile_animation(d["time"], d["T_gt"], d["T_pred"], x_mm,
                      out / "profile.gif", label=args.label)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", default="heat_python/data/rollout_aw1.npz")
    p.add_argument("--outdir", default="heat_python/figs")
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1")
    p.add_argument("--label", default="",
                   help="title prefix identifying the case/model, e.g. "
                        "'aw2 gas surrogate (conservative flux-form)'")
    main(p.parse_args())
