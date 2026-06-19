"""Draw the GNN graph on top of the finite-volume mesh: one NODE per cell
(at the cell center) and one EDGE per interior face (connecting neighboring
cell centers). Shows how the same physical slab maps to a different graph at
each mesh resolution -- the basis for mesh-resolution generalization.

    python -m heat_python.viz_graph
    python -m heat_python.viz_graph --nbrn 12 24
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .case import load_case
from .domain import setup_domain


def draw(ax, x_faces, l0, y, h, label):
    n = len(x_faces) - 1
    centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    dx0 = x_faces[1] - x_faces[0]
    dxN = x_faces[-1] - x_faces[-2]
    yc = y + h / 2

    # cells
    for i in range(n):
        ax.add_patch(Rectangle((x_faces[i], y), x_faces[i + 1] - x_faces[i], h,
                               facecolor="#eef4fb", edgecolor="0.7", linewidth=0.5))
    # ghost cells
    ghost_c = [-dx0 / 2, l0 + dxN / 2]
    for gx, x0, w in [(ghost_c[0], -dx0, dx0), (ghost_c[1], l0, dxN)]:
        ax.add_patch(Rectangle((x0, y), w, h, facecolor="#f7e3da",
                               edgecolor="0.7", linewidth=0.5, hatch="///"))

    # EDGES: interior faces connect neighbor centers; plus ghost->edge cells
    allc = [ghost_c[0]] + list(centers) + [ghost_c[1]]
    for a, b in zip(allc[:-1], allc[1:]):
        ax.plot([a, b], [yc, yc], "-", color="#c0392b", lw=1.3, zorder=2)
    # NODES: interior = blue dots, ghosts = orange squares
    ax.plot(centers, [yc] * n, "o", color="#1f6fb2", ms=5, zorder=3)
    ax.plot(ghost_c, [yc, yc], "s", color="#d35400", ms=6, zorder=3)

    ax.text(-dx0 - 0.04 * l0, yc, label, ha="right", va="center", fontsize=10)
    ax.text(l0 / 2, y + h + 0.18 * h, f"{n} nodes, {n + 1} edges",
            ha="center", va="bottom", fontsize=8, color="0.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1")
    ap.add_argument("--nbrn", type=int, nargs="+", default=[12, 24, 48])
    ap.add_argument("--out", default="heat_python/figs/graph.png")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case = load_case(repo / args.case_dir / "heat.case")
    l0 = case.l0 * 1e3   # display in mm

    fig, ax = plt.subplots(figsize=(11, 1.15 * len(args.nbrn) + 1.4))
    h, gap = 0.5, 0.7
    for k, n in enumerate(args.nbrn):
        case.nbrn = n
        d = setup_domain(case)
        draw(ax, d.x[1:n + 2] * 1e3, l0, k * (h + gap), h, f"NBRN={n}")

    ytop = len(args.nbrn) * (h + gap)

    # Inline callouts on the top (finest shown) mesh, pointing at a node, an
    # edge, and a ghost node, so the figure is self-explaining.
    case.nbrn = args.nbrn[-1]
    dtop = setup_domain(case)
    xf = dtop.x[1:args.nbrn[-1] + 2] * 1e3
    ctop = 0.5 * (xf[:-1] + xf[1:])
    ytop_row = (len(args.nbrn) - 1) * (h + gap)
    ycen = ytop_row + h / 2
    dx0 = xf[1] - xf[0]
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.8)

    # node callout
    ni = len(ctop) // 3
    ax.annotate("NODE = one cell center\ncarries [T, ρ, ρᵢ, porosity]",
                xy=(ctop[ni], ycen), xytext=(ctop[ni], ytop_row + h + 0.9),
                ha="center", va="bottom", fontsize=8.5, bbox=box, color="#1f6fb2",
                arrowprops=dict(arrowstyle="->", color="#1f6fb2", lw=1.2))
    # edge callout
    ei = 2 * len(ctop) // 3
    xmid = 0.5 * (ctop[ei] + ctop[ei + 1])
    ax.annotate("EDGE = shared cell face\nattr: Δx, ΔT, Δρ to neighbor",
                xy=(xmid, ycen), xytext=(xmid, ytop_row + h + 0.9),
                ha="center", va="bottom", fontsize=8.5, bbox=box, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    # ghost callout
    ax.annotate("GHOST NODE = boundary forcing\n(value imposed, not predicted)",
                xy=(l0 + dx0 / 2, ycen), xytext=(l0 - 0.02 * l0, ytop_row - 0.95),
                ha="center", va="top", fontsize=8.5, bbox=box, color="#d35400",
                arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2))

    ax.set_xlim(-0.14 * l0, 1.14 * l0)
    ax.set_ylim(-1.3, ytop + 1.7)
    ax.set_xlabel("x  [mm]")
    ax.set_yticks([])
    ax.set_title("GNN graph over the aw1 slab — coarse meshes shown so nodes/edges are visible")
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
