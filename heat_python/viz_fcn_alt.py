"""A fully-connected alternative to the mesh GNN, drawn to scale in parameters.

The whole mesh state is flattened into one long vector, pushed through dense
layers, and reshaped back. Contrast figure for the surrogate write-up: it shows
what the FCN commits to that the GNN does not.

    python -m heat_python.viz_fcn_alt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "heat_python/figs/architecture/fcn_alt.png"

NCELL, NFEAT, NOUT, HID = 502, 8, 5, 256
BLUE, ORANGE = "#3B82C4", "#E07A28"
INK, MUTED, GRID, SURF = "#22262b", "#6b7280", "#dfe1e5", "#fcfcfb"


def col(ax, x, ytop, ybot, n_show, color, lw=1.0, r=0.011):
    ys = np.linspace(ytop, ybot, n_show)
    for y in ys:
        ax.add_patch(plt.Circle((x, y), r, fc=color, ec="none", zorder=4))
    return ys


def connect(ax, x0, y0s, x1, y1s, color, alpha, lw=0.35):
    for a in y0s:
        for b in y1s:
            ax.plot([x0, x1], [a, b], color=color, lw=lw, alpha=alpha, zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(11.4, 6.2), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ytop, ybot = 0.80, 0.30

    # ---- mesh strip in --------------------------------------------------
    mx, mw = 0.035, 0.085
    for j in range(NFEAT):
        y = ybot + (ytop - ybot) * j / NFEAT
        ax.add_patch(Rectangle((mx, y), mw, (ytop - ybot) / NFEAT * 0.86,
                               fc=BLUE, alpha=0.16 + 0.05 * (j % 2), ec=BLUE, lw=0.5))
    ax.text(mx + mw / 2, ytop + 0.045, "mesh state", ha="center",
            fontsize=10, color=INK, fontweight="medium")
    ax.text(mx + mw / 2, ytop + 0.015, f"{NCELL} cells x {NFEAT} feats",
            ha="center", fontsize=8.5, color=MUTED)
    ax.text(mx + mw / 2, ybot - 0.035, "T, rho, rho_i, phi,\npg, mdotf",
            ha="center", fontsize=8, color=MUTED, va="top", linespacing=1.4)

    ax.annotate("", xy=(0.175, 0.55), xytext=(mx + mw + 0.008, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.152, 0.585, "flatten", ha="center", fontsize=9,
            color=INK, style="italic")

    # ---- layers ---------------------------------------------------------
    xs = [0.235, 0.415, 0.585, 0.765]
    sizes = [NCELL * NFEAT, HID, HID, NCELL * NOUT]
    labels = [f"{sizes[0]:,}", f"{HID}", f"{HID}", f"{sizes[3]:,}"]
    names = ["input", "hidden 1", "hidden 2", "output"]
    colors = [BLUE, "#8a8f98", "#8a8f98", ORANGE]
    shows = [13, 9, 9, 13]

    ycols = []
    for x, ns, c in zip(xs, shows, colors):
        ycols.append(col(ax, x, ytop, ybot, ns, c))

    for i in range(3):
        connect(ax, xs[i], ycols[i], xs[i + 1], ycols[i + 1], "#9aa0a8",
                alpha=0.30 if i else 0.22)

    for x, lab, nm, c in zip(xs, labels, names, colors):
        ax.text(x, ytop + 0.045, nm, ha="center", fontsize=10,
                color=INK, fontweight="medium")
        ax.text(x, ytop + 0.015, lab, ha="center", fontsize=9, color=c,
                fontweight="bold")

    # weight-matrix annotations
    for i in range(3):
        xm = (xs[i] + xs[i + 1]) / 2
        p = sizes[i] * sizes[i + 1] + sizes[i + 1]
        ax.text(xm, ybot - 0.048, f"{sizes[i]:,} x {sizes[i+1]:,}\n= {p:,}",
                ha="center", fontsize=8.2, color=MUTED, va="top",
                linespacing=1.4)

    # ---- mesh strip out -------------------------------------------------
    ax.annotate("", xy=(0.885, 0.55), xytext=(xs[3] + 0.022, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.862, 0.585, "reshape", ha="center", fontsize=9,
            color=INK, style="italic")
    ox, ow = 0.895, 0.075
    for j in range(NOUT):
        y = ybot + (ytop - ybot) * j / NOUT
        ax.add_patch(Rectangle((ox, y), ow, (ytop - ybot) / NOUT * 0.86,
                               fc=ORANGE, alpha=0.18 + 0.05 * (j % 2),
                               ec=ORANGE, lw=0.5))
    ax.text(ox + ow / 2, ytop + 0.045, "next state", ha="center",
            fontsize=10, color=INK, fontweight="medium")
    ax.text(ox + ow / 2, ytop + 0.015, f"{NCELL} x {NOUT} deltas",
            ha="center", fontsize=8.5, color=MUTED)

    total = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(3))
    ax.text(0.5, 0.150, f"{total:,} parameters", ha="center", fontsize=15,
            color=INK, fontweight="bold")
    ax.text(0.5, 0.116, "every cell wired to every cell, at one fixed mesh size",
            ha="center", fontsize=10, color=MUTED)

    notes = [
        ("locked to 502 cells", "another mesh needs another network"),
        ("no locality", "cell 12 wires to cell 400, and must unlearn it"),
        ("no weight sharing", "one conduction law, relearned 502 times"),
    ]
    bx = 0.055
    for i, (h, sub) in enumerate(notes):
        x = bx + i * 0.315
        ax.add_patch(FancyBboxPatch((x, 0.005), 0.295, 0.066,
                                    boxstyle="round,pad=0.008,rounding_size=0.012",
                                    fc="#f2f3f4", ec=GRID, lw=0.8))
        ax.text(x + 0.1475, 0.052, h, ha="center", fontsize=9,
                color="#b35f18", fontweight="bold")
        ax.text(x + 0.1475, 0.024, sub, ha="center", fontsize=8, color=MUTED)

    ax.text(0.0, 0.965, "The fully-connected alternative", fontsize=14,
            color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.928, f"aw2 mesh, hidden width {HID}", fontsize=9.5,
            color=MUTED, transform=ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"total params {total:,}   (GNN allocates 110,299, trains 10)")
    for h in (64, 128, 256, 512):
        s = [NCELL * NFEAT, h, h, NCELL * NOUT]
        t = sum(s[i] * s[i + 1] + s[i + 1] for i in range(3))
        print(f"   hidden {h:4d} -> {t:12,} params")


if __name__ == "__main__":
    main()
