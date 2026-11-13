"""The conductance GNN we actually use, drawn to match viz_fcn_alt.py.

Companion figure: same mesh, same job, but the update is a bounded three-cell
stencil applied with shared weights at every cell, so the parameter count does
not scale with the mesh and the physics is structural rather than learned.

    python -m heat_python.viz_gnn_alt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "heat_python/figs/architecture/gnn_alt.png"

NCELL, NFEAT, NOUT, SUB = 502, 8, 5, 8
BLUE, ORANGE = "#3B82C4", "#E07A28"
GREEN = "#2f8f6b"
INK, MUTED, GRID, SURF = "#22262b", "#6b7280", "#dfe1e5", "#fcfcfb"
DEAD = "#b9bdc3"


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def main():
    fig, ax = plt.subplots(figsize=(11.4, 6.2), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ytop, ybot = 0.80, 0.30

    # ---- mesh in --------------------------------------------------------
    mx, mw = 0.035, 0.085
    for j in range(NFEAT):
        y = ybot + (ytop - ybot) * j / NFEAT
        ax.add_patch(Rectangle((mx, y), mw, (ytop - ybot) / NFEAT * 0.86,
                               fc=BLUE, alpha=0.16 + 0.05 * (j % 2),
                               ec=BLUE, lw=0.5))
    ax.text(mx + mw / 2, ytop + 0.045, "mesh state", ha="center", fontsize=10,
            color=INK, fontweight="medium")
    ax.text(mx + mw / 2, ytop + 0.015, f"{NCELL} cells x {NFEAT} feats",
            ha="center", fontsize=8.5, color=MUTED)
    ax.text(mx + mw / 2, ybot - 0.035, "T, rho, rho_i, phi,\npg, mdotf",
            ha="center", fontsize=8, color=MUTED, va="top", linespacing=1.4)

    ax.annotate("", xy=(0.205, 0.55), xytext=(mx + mw + 0.008, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.172, 0.585, "one cell\nat a time", ha="center", fontsize=8.6,
            color=INK, style="italic", linespacing=1.35)

    # ---- the three-cell stencil ----------------------------------------
    cw, ch, cy = 0.062, 0.10, 0.60
    cxs = [0.222, 0.292, 0.362]
    labs = [r"$u_{i-1}$", r"$u_i$", r"$u_{i+1}$"]
    fcs = ["#e8eef5", "#cfe0ef", "#e8eef5"]
    for x, l, f in zip(cxs, labs, fcs):
        rbox(ax, x, cy, cw, ch, f, BLUE, lw=1.2)
        ax.text(x + cw / 2, cy + ch / 2, l, ha="center", va="center",
                fontsize=12, color=INK)
    ax.text(cxs[1] + cw / 2, cy + ch + 0.062, "the stencil", ha="center",
            fontsize=9.5, color=INK, fontweight="medium")
    ax.text(cxs[1] + cw / 2, cy + ch + 0.022, "conduction in from both faces",
            ha="center", fontsize=8.4, color=GREEN)

    for a, b in ((0, 1), (2, 1)):
        x0 = cxs[a] + (cw if a == 0 else 0)
        x1 = cxs[b] + (0 if a == 0 else cw)
        ax.add_patch(FancyArrowPatch((x0, cy + ch * 0.5), (x1, cy + ch * 0.5),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=GREEN, lw=1.6, zorder=5,
                                     shrinkA=1, shrinkB=1))

    ax.add_patch(FancyArrowPatch((cxs[1] + cw / 2, cy - 0.075),
                                 (cxs[1] + cw / 2, cy - 0.010),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=ORANGE, lw=1.6, zorder=5))
    ax.text(cxs[1] + cw / 2, cy - 0.088, "pyrolysis source", ha="center",
            fontsize=8.4, color=ORANGE, va="top")

    # ---- the update ------------------------------------------------------
    ux = 0.455
    rbox(ax, ux, 0.540, 0.245, 0.215, "#f6f7f8", GRID, lw=1.0)
    ax.text(ux + 0.1225, 0.718, "the update", ha="center", fontsize=9.5,
            color=INK, fontweight="medium")
    ax.text(ux + 0.1225, 0.663,
            r"$u_i \leftarrow u_i + \Sigma_j D\,(u_j - u_i) + S_i$",
            ha="center", fontsize=12.5, color=INK)
    ax.text(ux + 0.1225, 0.602,
            r"$D \geq 0$,     $\Sigma_j D \leq 1$", ha="center", fontsize=10,
            color=GREEN)
    ax.text(ux + 0.1225, 0.562, "a convex combination, so it cannot blow up",
            ha="center", fontsize=8.2, color=MUTED)

    rbox(ax, ux, 0.372, 0.117, 0.138, "#eaf2ea", GREEN, lw=1.2)
    ax.text(ux + 0.0585, 0.478, "conductance", ha="center", fontsize=8.8,
            color=GREEN, fontweight="bold")
    ax.text(ux + 0.0585, 0.446, r"$D = \frac{1}{2}\sigma(\theta)$",
            ha="center", fontsize=11, color=INK)
    ax.text(ux + 0.0585, 0.390, "5 learned", ha="center", fontsize=8.4,
            color=GREEN)

    rbox(ax, ux + 0.128, 0.372, 0.117, 0.138, "#fdf0e6", ORANGE, lw=1.2)
    ax.text(ux + 0.1865, 0.478, "source", ha="center", fontsize=8.8,
            color=ORANGE, fontweight="bold")
    ax.text(ux + 0.1865, 0.443, "Arrhenius, fixed", ha="center", fontsize=8.2,
            color=INK)
    ax.text(ux + 0.1865, 0.413, r"$+\ c \cdot \dot{m}$", ha="center",
            fontsize=10, color=INK)
    ax.text(ux + 0.1865, 0.390, "5 learned", ha="center", fontsize=8.4,
            color=ORANGE)

    ax.annotate("", xy=(ux + 0.075, 0.540), xytext=(ux + 0.055, 0.514),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.annotate("", xy=(ux + 0.175, 0.540), xytext=(ux + 0.190, 0.514),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))

    ax.text(ux + 0.1225, 0.782, f"x {SUB} substeps per super-step",
            ha="center", fontsize=8.8, color=MUTED, style="italic")
    ax.add_patch(FancyArrowPatch((0.293, 0.345), (0.293, 0.242),
                                 arrowstyle="-", color=MUTED, lw=1.0,
                                 ls=(0, (3, 2))))
    ax.text(0.310, 0.212,
            f"the same weights are applied at all {NCELL} cells",
            ha="center", fontsize=9, color=INK, style="italic")

    # ---- mesh out --------------------------------------------------------
    ax.annotate("", xy=(0.885, 0.55), xytext=(0.712, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ox, ow = 0.895, 0.075
    for j in range(NOUT):
        y = ybot + (ytop - ybot) * j / NOUT
        ax.add_patch(Rectangle((ox, y), ow, (ytop - ybot) / NOUT * 0.86,
                               fc=ORANGE, alpha=0.18 + 0.05 * (j % 2),
                               ec=ORANGE, lw=0.5))
    ax.text(ox + ow / 2, ytop + 0.045, "next state", ha="center", fontsize=10,
            color=INK, fontweight="medium")
    ax.text(ox + ow / 2, ytop + 0.015, f"{NCELL} x {NOUT} deltas", ha="center",
            fontsize=8.5, color=MUTED)

    # ---- dead machinery --------------------------------------------------
    ax.text(ux + 0.1225, 0.330,
            "node_encoder, geom_encoder, cond_mlp, source_mlp",
            ha="center", fontsize=8.2, color=DEAD)
    ax.text(ux + 0.1225, 0.300,
            "110,289 more parameters, allocated but never called",
            ha="center", fontsize=8.6, color=DEAD, fontweight="bold")

    ax.text(0.5, 0.150, "10 parameters", ha="center", fontsize=15,
            color=INK, fontweight="bold")
    ax.text(0.5, 0.116,
            "one three-cell rule, shared by every cell, at any mesh size",
            ha="center", fontsize=10, color=MUTED)

    notes = [
        ("any mesh size", "the same weights run 100 or 502 cells"),
        ("local by construction", "a cell only ever sees its two neighbours"),
        ("bounded by construction", "convex update, stable at any time step"),
    ]
    for i, (h, sub) in enumerate(notes):
        x = 0.055 + i * 0.315
        rbox(ax, x, 0.005, 0.295, 0.066, "#f2f3f4", GRID, lw=0.8, z=2)
        ax.text(x + 0.1475, 0.052, h, ha="center", fontsize=9,
                color=GREEN, fontweight="bold")
        ax.text(x + 0.1475, 0.024, sub, ha="center", fontsize=8, color=MUTED)

    ax.text(0.0, 0.965, "The conductance GNN we use", fontsize=14,
            color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.928, "aw2 mesh, uniform-cond + real-kinetics relax model",
            fontsize=9.5, color=MUTED, transform=ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
