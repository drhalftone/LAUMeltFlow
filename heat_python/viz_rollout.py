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
from matplotlib.colors import TwoSlopeNorm

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


def _char_front(rho_int, x_mm, thresh):
    """Char-front position [mm] per snapshot: deepest cell below `thresh`
    (from the hot/outer face). Same definition as viz_rollout_anim."""
    f = []
    for s in range(rho_int.shape[0]):
        c = np.where(rho_int[s] < thresh)[0]
        f.append(x_mm[c.min()] if c.size else x_mm[-1])
    return np.array(f)


def _even_ticks(top, step=None, n=5):
    """Even-increment ticks 0..top, last snapped to `top` (may be odd)."""
    if step is None:
        step = next((s for s in (5, 10, 20, 25, 50, 100) if s >= top / n), 100)
    ticks = list(np.arange(0, top, step))
    if top - ticks[-1] < 0.4 * step:
        ticks[-1] = top
    else:
        ticks.append(top)
    return ticks


def _nice_ceil(v, n=5):
    """Round v up to a clean value (5/10/20/25/50/100 grid)."""
    step = next((s for s in (5, 10, 20, 25, 50, 100) if s >= v / n), 100)
    return float(np.ceil(v / step) * step)


def _signed_ticks(mn, mx, n=5):
    """Even ticks from the true min (mn<0) to the true max (mx>0), both ends
    snapped to the actual extremes, 0 included, same step on both sides."""
    top = max(-mn, mx)
    step = next((s for s in (5, 10, 20, 25, 50, 100) if s >= top / n), 100)
    pos = _even_ticks(mx, step=step)                 # [0 .. mx]
    neg = _even_ticks(-mn, step=step)                # [0 .. -mn]
    return [-t for t in reversed(neg) if t > 0] + pos


def static_heatmap(time, T_gt, T_pred, x_mm, out_png, label="",
                   rho_gt=None, rho_pred=None, rhov_bulk=None, virgin_frac=0.98,
                   signed=False):
    gt = T_gt[:, 1:-1]                            # interior (S, n)
    pr = T_pred[:, 1:-1]
    err = (pr - gt) if signed else np.abs(pr - gt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    extent = [x_mm[0], x_mm[-1], time[-1], time[0]]
    vmax = max(gt.max(), pr.max())
    # Char fronts (drawn on both temperature panels if density is available).
    fronts = None
    if rho_gt is not None and rho_pred is not None and rhov_bulk is not None:
        thresh = virgin_frac * float(rhov_bulk)
        fg = _char_front(rho_gt[:, 1:-1], x_mm, thresh)
        fp = _char_front(rho_pred[:, 1:-1], x_mm, thresh)
        fronts = (fg, fp)
    emin, emax = float(err.min()), float(err.max())
    err_extend = None
    if signed:
        # Clip to a robust symmetric range so isolated outlier pixels (e.g. the
        # first-step wall shock) do not wash out the real structure. Values
        # beyond the clip render at the end colors and get a colorbar arrow.
        R = _nice_ceil(float(np.percentile(np.abs(err), 99.5)))
        vlo, vhi = -R, R
        if emin < vlo and emax > vhi:
            err_extend = "both"
        elif emin < vlo:
            err_extend = "min"
        elif emax > vhi:
            err_extend = "max"
        err_title = "error"
        err_cmap, err_lab = "RdBu_r", "surrogate $-$ solver [K]"
        err_norm = TwoSlopeNorm(vcenter=0, vmin=vlo, vmax=vhi)
        err_ticks = _signed_ticks(vlo, vhi)
    else:
        err_title = f"|error|  (max {emax:.0f} K)"
        err_cmap, err_lab = "viridis", "|ΔT| [K]"
        err_norm = None
        err_ticks = _even_ticks(emax)
    for ax, data, title, cmap, kind in [
            (axes[0], gt, "Solver (truth)", "inferno", "solver"),
            (axes[1], pr, "GNN surrogate", "inferno", "gnn"),
            (axes[2], err, err_title, err_cmap, "err")]:
        is_temp = kind != "err"
        if is_temp:
            im = ax.imshow(data, aspect="auto", extent=extent, cmap=cmap,
                           vmin=0, vmax=vmax)
        elif signed:
            im = ax.imshow(data, aspect="auto", extent=extent, cmap=cmap,
                           norm=err_norm)
        else:
            im = ax.imshow(data, aspect="auto", extent=extent, cmap=cmap,
                           vmin=0, vmax=emax)
        if fronts is not None and kind == "solver":  # solver front only
            ax.plot(fronts[0], time, "w-", lw=1.8, label="char front: solver")
            ax.set_xlim(x_mm[0], x_mm[-1])
            ax.legend(loc="lower left", fontsize=13)
        elif fronts is not None and kind == "gnn":    # both, to show tracking
            ax.plot(fronts[0], time, "w-", lw=1.8, label="char front: solver")
            ax.plot(fronts[1], time, "--", color="#39ff14", lw=1.6,
                    label="char front: GNN")
            ax.set_xlim(x_mm[0], x_mm[-1])
            ax.legend(loc="lower left", fontsize=13)
        elif fronts is not None:                      # surrogate front on |error|
            ax.plot(fronts[1], time, "--", color="#39ff14", lw=1.6,
                    label="char front: GNN")
            ax.set_xlim(x_mm[0], x_mm[-1])
            ax.legend(loc="lower left", fontsize=13)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel("position x [mm]  (hot wall at right)", fontsize=18)
        ax.tick_params(axis="both", labelsize=15)
        cbar = fig.colorbar(im, ax=ax, label="T [K]" if is_temp else err_lab,
                            extend=("neither" if is_temp else
                                    (err_extend or "neither")))
        cbar.ax.yaxis.label.set_size(17)
        cbar.ax.tick_params(labelsize=14)
        if not is_temp:
            cbar.set_ticks(err_ticks)
            cbar.set_ticklabels([f"{t:.0f}" for t in err_ticks])
    axes[0].set_ylabel("time [s]", fontsize=18)
    title = (f"{label}: rollout vs solver (space-time temperature)" if label
             else "Heat-shield rollout: GNN vs solver (space-time temperature)")
    fig.suptitle(title, fontsize=21)
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
                   out / "spacetime.png", label=args.label,
                   rho_gt=d.get("rho_gt"), rho_pred=d.get("rho_pred"),
                   rhov_bulk=d.get("rhov_bulk"))
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
