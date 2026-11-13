"""One contrast figure for the paper: same held-out aw2 case, three space-time
temperature maps side by side.

    solver truth  |  purely-learned (diverges)  |  physics-informed (matches)

Each map carries the char-front curves (solver white, GNN red dashed). The color
scale is pinned to the SOLVER range, so the purely-learned panel saturates to
off-scale green where it diverges, while the physics-informed panel stays on scale.
Replaces the separate diverged + physics-informed 3-panel figures (their solver
panels are identical).

    python -m heat_python.viz_contrast
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .viz_rollout import _depth_axis

REPO = Path(__file__).resolve().parents[1]


def _fronts(rho_int, x_mm, thresh):
    f = []
    for s in range(rho_int.shape[0]):
        c = np.where(rho_int[s] < thresh)[0]
        f.append(x_mm[c.min()] if c.size else x_mm[-1])
    return np.array(f)


def main(args):
    solver = dict(np.load(REPO / args.solver_src))     # provides T_gt, rho_gt
    direct = dict(np.load(REPO / args.direct))         # purely-learned
    phys = dict(np.load(REPO / args.phys))             # physics-informed

    time = solver["time"]
    Tg = solver["T_gt"][:, 1:-1]
    n = Tg.shape[1]
    x_mm = _depth_axis(args.case_dir, n)
    thresh = args.virgin_frac * float(solver["rhov_bulk"])

    Td = direct["T_pred"][:, 1:-1]
    fg = _fronts(solver["rho_gt"][:, 1:-1], x_mm, thresh)
    fd = _fronts(direct["rho_pred"][:, 1:-1], x_mm, thresh)

    vmin, vmax = float(Tg.min()), float(Tg.max())
    extent = [x_mm[0], x_mm[-1], time[-1], time[0]]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_over("#39ff14")                            # above solver max = diverged

    # Two panels, sized to match the per-panel shape of the aw1/aw2 rollout figures.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = [
        (axes[0], Tg, None, "Solver (truth)", "neither"),
        (axes[1], Td, fd, "Purely-learned GNN (diverges)", "max"),
    ]
    for ax, field, gnn_front, title, ext in panels:
        im = ax.imshow(field, aspect="auto", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.plot(fg, time, "w-", lw=1.8, label="char front: solver")
        if gnn_front is not None:
            ax.plot(gnn_front, time, "r--", lw=1.6, label="char front: GNN")
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("position x [mm]  (hot wall at right)")
        ax.legend(loc="lower left", fontsize=8)
        fig.colorbar(im, ax=ax, label="T [K]", extend=ext)
    axes[0].set_ylabel("time [s]")

    fig.tight_layout()                                 # match the rollout figures
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for ext in (".pdf", ".png"):
        fig.savefig(f"{stem}{ext}", dpi=130)
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--solver-src", default="heat_python/data/rollout_cond_aw2_fine_n24.npz")
    p.add_argument("--direct", default="heat_python/data/rollout_direct_aw2_fine.npz")
    p.add_argument("--phys", default="heat_python/data/rollout_cond_aw2_fine_n24.npz")
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--virgin-frac", type=float, default=0.98)
    p.add_argument("--out", default="heat_python/figs/aw2_conductance/aw2_contrast_3map.png")
    main(p.parse_args())
