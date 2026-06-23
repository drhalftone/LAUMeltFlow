"""Animated rollout: watch the thermal wave + char front advance over time.

A two-panel GIF (both axes fixed; the plot fills in as time advances):
  - Left:  instantaneous T(depth) profile, solver vs GNN, with a fading trail of
           past profiles (the penetrating thermal wave) and char-front markers.
  - Right: the GNN's space-time T(x,t) map revealed top-to-bottom over time (the
           animated form of rollout_spacetime.png), with the solver and GNN
           char-front curves drawn progressively so divergence is visible.

The char front is the deepest interior cell (from the hot/outer face) whose
density has dropped below `virgin_frac` of the virgin density -- same definition
used by viz_compare.char_front_compare. Needs rho_gt/rho_pred in the rollout npz
(eval_rollout_traj saves them).

    python -m heat_python.viz_rollout_anim \
        --rollout heat_python/data/rollout_flux_aw2_best.npz \
        --case-dir heat_2026-04-11_1837/examples/aw2_tc21 \
        --label "aw2 gas surrogate (flux-form, best seed)" \
        --out heat_python/figs/aw2_best_rollout.gif
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .viz_rollout import _depth_axis

TRAIL = 8                      # number of fading past-profile ghosts (left panel)


def _fronts(rho_int, x_mm, thresh):
    """Char-front position [mm] per snapshot: deepest cell below `thresh`."""
    f = []
    for s in range(rho_int.shape[0]):
        c = np.where(rho_int[s] < thresh)[0]
        f.append(x_mm[c.min()] if c.size else x_mm[-1])
    return np.array(f)


def _build(d, x_mm, label, virgin_frac, with_solver=False):
    """Build the figure and an update(s) drawer; return (fig, update, S, idxmap)."""
    time = d["time"]
    Tg, Tp = d["T_gt"][:, 1:-1], d["T_pred"][:, 1:-1]
    rg, rp = d["rho_gt"][:, 1:-1], d["rho_pred"][:, 1:-1]
    thresh = virgin_frac * float(d["rhov_bulk"])
    S = Tg.shape[0]
    fg, fp = _fronts(rg, x_mm, thresh), _fronts(rp, x_mm, thresh)
    vmin, vmax = float(min(Tg.min(), Tp.min())), float(max(Tg.max(), Tp.max()))

    extent = [x_mm[0], x_mm[-1], time[-1], time[0]]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.12")                     # unfilled (future) = dark

    def map_panel(ax, title):
        """A space-time T-map panel that fills top-to-bottom over time, with
        both char-front curves drawn progressively. Returns (im, c_gt, c_pr,
        now)."""
        im = ax.imshow(np.full_like(Tp, np.nan), aspect="auto", extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        (c_gt,) = ax.plot([], [], "w-", lw=2.0, label="char front: solver")
        (c_pr,) = ax.plot([], [], "r--", lw=1.8, label="char front: GNN")
        now = ax.axhline(time[0], color="cyan", lw=1.2, alpha=0.8)
        ax.set_xlim(x_mm[0], x_mm[-1]); ax.set_ylim(time[-1], time[0])
        ax.set_xlabel("position x [mm]  (hot wall at right)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        return im, c_gt, c_pr, now

    ncol = 3 if with_solver else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 5.4))
    axL = axes[0]
    if label:
        fig.suptitle(label, fontsize=12)

    # --- Left: profile + trail + char fronts ---
    ghosts = [axL.plot([], [], "-", color="0.6", lw=1.0, alpha=a)[0]
              for a in np.linspace(0.08, 0.45, TRAIL)]
    (l_gt,) = axL.plot([], [], "k-", lw=2.6, label="solver (truth)")
    (l_pr,) = axL.plot([], [], "r--", lw=2.0, label="GNN surrogate")
    vf_gt = axL.axvline(np.nan, color="0.35", ls=":", lw=1.6,
                        label="char front: solver")
    vf_pr = axL.axvline(np.nan, color="red", ls=":", lw=1.6,
                        label="char front: GNN")
    axL.set_xlim(x_mm[0], x_mm[-1])
    axL.set_ylim(vmin - 30, vmax + 60)
    axL.set_xlabel("position x [mm]  (hot wall at right)")
    axL.set_ylabel("temperature [K]")
    axL.legend(loc="upper left", fontsize=8)

    # --- Map panel(s): solver truth (optional) + GNN prediction ---
    panels = []          # (im, fields, fg/fp not needed: same fronts both)
    if with_solver:
        axes[1].set_ylabel("time [s]")
        im_s, cs_gt, cs_pr, now_s = map_panel(axes[1], "Fortran solver T(x,t)")
        panels.append((im_s, Tg, cs_gt, cs_pr, now_s))
        ax_g = axes[2]
    else:
        ax_g = axes[1]
        ax_g.set_ylabel("time [s]")
    im_g, cg_gt, cg_pr, now_g = map_panel(
        ax_g, "GNN surrogate prediction T(x,t)")
    panels.append((im_g, Tp, cg_gt, cg_pr, now_g))
    fig.colorbar(im_g, ax=ax_g, label="T [K]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def update(s):
        l_gt.set_data(x_mm, Tg[s])
        l_pr.set_data(x_mm, Tp[s])
        past = np.linspace(max(0, s - 60), s, TRAIL + 1).astype(int)[:-1]
        for g, ps in zip(ghosts, past):
            g.set_data(x_mm, Tg[ps])
        vf_gt.set_xdata([fg[s], fg[s]])
        vf_pr.set_xdata([fp[s], fp[s]])
        e = np.abs(Tp[s] - Tg[s])
        axL.set_title(f"t = {time[s]:.1f} s    mean |dT| = {e.mean():.1f} K"
                      f"    max = {e.max():.1f} K", fontsize=10)
        artists = [l_gt, l_pr, vf_gt, vf_pr, *ghosts]
        for im, field, c_gt, c_pr, now in panels:
            disp = np.full_like(field, np.nan)
            disp[:s + 1] = field[:s + 1]
            im.set_data(disp)
            c_gt.set_data(fg[:s + 1], time[:s + 1])
            c_pr.set_data(fp[:s + 1], time[:s + 1])
            now.set_ydata([time[s], time[s]])
            artists += [im, c_gt, c_pr, now]
        return tuple(artists)

    return fig, update, S


def main(args):
    d = dict(np.load(args.rollout))
    n = d["T_gt"].shape[1] - 2
    x_mm = _depth_axis(args.case_dir, n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, update, S = _build(d, x_mm, args.label, args.virgin_frac,
                            args.with_solver)
    idx = np.linspace(0, S - 1, min(args.n_frames, S)).astype(int)
    anim = FuncAnimation(fig, lambda k: update(idx[k]), frames=len(idx),
                         blit=False)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"  saved {out}")

    # verification filmstrip: early / mid / late composite frames as one PNG
    if args.filmstrip:
        for tag, frac in [("early", 0.12), ("mid", 0.5), ("late", 0.99)]:
            fig, update, S = _build(d, x_mm, f"{args.label} -- {tag}",
                                    args.virgin_frac, args.with_solver)
            update(int(frac * (S - 1)))
            p = out.with_name(out.stem + f"_frame_{tag}.png")
            fig.savefig(p, dpi=110); plt.close(fig)
            print(f"  saved {p}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", default="heat_python/data/rollout_flux_aw2_best.npz")
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--out", default="heat_python/figs/best/aw2/aw2_best_rollout.gif")
    p.add_argument("--label", default="aw2 gas surrogate (flux-form, best seed)")
    p.add_argument("--n-frames", type=int, default=160)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--virgin-frac", type=float, default=0.98)
    p.add_argument("--filmstrip", action="store_true",
                   help="also dump early/mid/late composite frames as PNGs")
    p.add_argument("--with-solver", action="store_true",
                   help="add a Fortran-solver-truth filling map beside the GNN one")
    main(p.parse_args())
