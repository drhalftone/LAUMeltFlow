"""Draw the 1D finite-volume mesh as a row of cells. Shows the aw1 slab at a
few NBRN values so 'mesh-resolution generalization' is concrete: same physical
slab, different number of cells (and ghost cells at each end carrying forcing).

    python -m heat_python.viz_mesh                 # default: aw1 at 40/100/200
    python -m heat_python.viz_mesh --nbrn 20 60    # custom resolutions
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .case import load_case
from .domain import setup_domain


def draw_mesh(ax, x_faces, l0, y, h, label, max_show=60):
    """Draw one mesh as a row of cell boxes between y and y+h."""
    n = len(x_faces) - 1
    # ghost cells: one before x[0]=0 and one after x[-1]=l0, width = first/last cell
    dx0 = x_faces[1] - x_faces[0]
    dxN = x_faces[-1] - x_faces[-2]
    ghost_l = (-dx0, 0.0)
    ghost_r = (l0, l0 + dxN)

    show_all = n <= max_show
    for i in range(n):
        xl, xr = x_faces[i], x_faces[i + 1]
        edge = "0.4" if (show_all or i % max(1, n // max_show) == 0) else "none"
        ax.add_patch(Rectangle((xl, y), xr - xl, h, facecolor="#cfe3f7",
                               edgecolor=edge, linewidth=0.4))
    # outline the whole interior so coarse vs fine reads even when edges are hidden
    ax.add_patch(Rectangle((0, y), l0, h, facecolor="none", edgecolor="0.2", linewidth=1.0))

    # ghost cells (hatched) — these carry the boundary forcing in the GNN graph
    for (gl, gr), side in [(ghost_l, "L"), (ghost_r, "R")]:
        ax.add_patch(Rectangle((gl, y), gr - gl, h, facecolor="#f7d9cf",
                               edgecolor="0.4", linewidth=0.6, hatch="///"))

    ax.text(-dx0 - 0.04 * l0, y + h / 2, label, ha="right", va="center", fontsize=10)
    ax.text(l0 / 2, y + h + 0.12 * h, f"{n} cells, dx = {l0 / n * 1e3:.2f} mm",
            ha="center", va="bottom", fontsize=8, color="0.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1")
    ap.add_argument("--nbrn", type=int, nargs="+", default=[40, 100, 200])
    ap.add_argument("--out", default="heat_python/figs/mesh.png")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case = load_case(repo / args.case_dir / "heat.case")
    l0 = case.l0

    fig, ax = plt.subplots(figsize=(11, 1.1 * len(args.nbrn) + 1.2))
    h = 0.6
    gap = 0.5
    for k, n in enumerate(args.nbrn):
        case.nbrn = n
        d = setup_domain(case)
        x_faces = d.x[1:n + 2]  # n+1 interior faces, x[1]=0 .. x[n+1]=l0
        y = k * (h + gap)
        draw_mesh(ax, x_faces, l0, y, h, f"NBRN={n}")

    # annotate boundaries
    ytop = len(args.nbrn) * (h + gap)
    ax.annotate("inner wall\nx=0", (0, ytop - gap), ha="center", fontsize=8, color="#16607a")
    ax.annotate("outer wall (aero heating)\nx=L0", (l0, ytop - gap), ha="center",
                fontsize=8, color="#a8421f")
    ax.text(-0.06 * l0, -0.5, "hatched = ghost cells (boundary forcing)",
            fontsize=8, color="0.4")

    ax.set_xlim(-0.12 * l0, 1.12 * l0)
    ax.set_ylim(-0.7, ytop + 0.3)
    ax.set_xlabel("x  [m]")
    ax.set_yticks([])
    ax.set_title(f"aw1 slab (L0 = {l0*1e3:.0f} mm, Cartesian) at three mesh resolutions")
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
