"""Show how the moving char front maps onto the FIXED graph nodes.

Top panel: the same row of nodes (cell centers) repeated at several times, each
node colored by how charred it is (char fraction beta: 0 = virgin, 1 = fully
char). The nodes never move; the color boundary -- the char front -- sweeps
across them from the hot wall inward. A marker tracks where the front sits each
time.

Bottom panel: the classic char-front depth vs time (same data, the view from
char_front.png), with the snapshot times marked, so the two pictures line up.

    python -m heat_python.viz_front_on_nodes
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def front_x(beta, xc):
    """Interpolated x (mm) where beta crosses 0.5, scanning from the hot wall
    (right) inward. Returns L0 if nothing has charred yet."""
    for i in range(len(xc) - 1, 0, -1):
        if beta[i] >= 0.5 and beta[i - 1] < 0.5:
            t = (0.5 - beta[i - 1]) / (beta[i] - beta[i - 1])
            return xc[i - 1] + t * (xc[i] - xc[i - 1])
    return xc[-1] if beta[-1] >= 0.5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="heat_python/data/rollout_aw1_n100.npz")
    ap.add_argument("--l0-mm", type=float, default=50.0)
    ap.add_argument("--n-snap", type=int, default=6)
    ap.add_argument("--out", default="heat_python/figs/front_on_nodes.png")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    d = np.load(repo / args.rollout)
    time = d["time"]
    rho = d["rho_gt"][:, 1:-1]                 # interior, drop ghosts -> (S, n)
    rhov, rhoc = float(d["rhov_bulk"]), float(d["rhoc_bulk"])
    S, n = rho.shape
    beta_all = np.clip((rhov - rho) / (rhov - rhoc), 0, 1)   # char fraction

    dx = args.l0_mm / n
    xc = (np.arange(n) + 0.5) * dx             # fixed cell centers (mm)

    snaps = np.linspace(0, S - 1, args.n_snap).astype(int)

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=[2.4, 1], constrained_layout=True)

    cmap = plt.cm.coolwarm
    # --- top: node strips over time ---
    for row, s in enumerate(snaps):
        y = row
        beta = beta_all[s]
        axT.scatter(xc, np.full(n, y), c=beta, cmap=cmap, vmin=0, vmax=1,
                    s=48, edgecolors="0.4", linewidths=0.3, zorder=3)
        xf = front_x(beta, xc)
        if xf is not None:
            axT.annotate("", xy=(xf, y), xytext=(xf, y + 0.42),
                         arrowprops=dict(arrowstyle="-|>", color="#111", lw=2))
            axT.text(xf, y + 0.5, "front", color="#111", fontsize=8,
                     ha="center", va="bottom")
        axT.text(-2.5, y, f"t = {time[s]:4.1f} s", ha="right", va="center", fontsize=9)

    axT.set_xlim(-9, args.l0_mm + 2)
    axT.set_ylim(-0.6, len(snaps) - 0.2)
    axT.set_yticks([])
    axT.set_xlabel("x  [mm]   (same fixed nodes every row)")
    axT.set_title("The char front is a color boundary sweeping across FIXED nodes\n"
                  "each dot = one node; color = char fraction (0 virgin → 1 char)")
    axT.text(args.l0_mm, len(snaps) - 0.4, "hot wall →", color="#a8421f",
             fontsize=9, ha="right", va="bottom")
    axT.text(0, len(snaps) - 0.4, "← cold wall", color="#16607a",
             fontsize=9, ha="left", va="bottom")
    for sp in ["left", "right", "top"]:
        axT.spines[sp].set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axT, shrink=0.7,
                      pad=0.01, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("char fraction β")
    cb.ax.set_yticklabels(["virgin", "front", "char"])

    # --- bottom: char-front depth vs time (the char_front.png view) ---
    depth = np.array([
        (args.l0_mm - (front_x(beta_all[s], xc) or args.l0_mm))
        for s in range(S)])
    axB.plot(time, depth, color="#c0392b", lw=2)
    for s in snaps:
        axB.axvline(time[s], color="0.7", ls=":", lw=1)
    axB.set_xlabel("time  [s]")
    axB.set_ylabel("char depth\nfrom hot wall [mm]")
    axB.set_title("Same data as a front-depth curve (dotted lines = the snapshots above)",
                  fontsize=10)
    axB.invert_yaxis()
    axB.margins(x=0.01)
    for sp in ["right", "top"]:
        axB.spines[sp].set_visible(False)

    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
