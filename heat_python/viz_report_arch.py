"""Clean landscape architecture figure for Goodwin_report.tex.

Beats matching Section 3:
  1. finite-volume mesh IS a graph (cells -> nodes, faces -> edges)
  2. one update = three pieces summed (fixed conduction [relaxed], fixed
     Arrhenius pyrolysis, small learned gas coupling), a bounded neighbor exchange
  2b. transfer-learning training (relax the anchor + focus on the transient): 25 K -> 18 K
  3. bounded exchange -> cannot exceed its surroundings -> cannot blow up

-> docs/heat_TPS/figs/gnn_arch.{pdf,png}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

FIX = "#3b8f5a"     # fixed to known physics (green)
LRN = "#c0611f"     # learned by the GNN (orange)
STR = "#6d4fa8"     # structural guarantee (purple)
NODE = "#333333"    # graph nodes (neutral)
GRY = "#555555"

fig, ax = plt.subplots(figsize=(14.5, 5.6))
ax.set_xlim(0, 14.5); ax.set_ylim(1.95, 7.6); ax.axis("off")


def box(cx, cy, w, h, color, fill=None, lw=2.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 linewidth=lw, edgecolor=color,
                 facecolor=(fill if fill else color + "12")))


def arrow(x0, y0, x1, y1, color=GRY, lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=18, color=color, lw=lw))


# ---------------------------------------------------------------- 1. mesh->graph
ax.text(2.2, 7.25, "1.  the finite-volume mesh is a graph", ha="center",
        fontsize=15, fontweight="bold", color="0.1")
n = 5
x0, cellw, cy = 0.6, 0.66, 5.7
for i in range(n):
    cx = x0 + i * cellw
    ax.add_patch(Rectangle((cx, cy - 0.36), cellw, 0.72, fill=False,
                 edgecolor="0.6", lw=1.4))
centers = [x0 + (i + 0.5) * cellw for i in range(n)]
for i in range(n - 1):                       # edges = faces
    ax.plot([centers[i], centers[i + 1]], [cy, cy], color=NODE, lw=2.0, zorder=1)
for cx in centers:                           # nodes = cells
    ax.add_patch(Circle((cx, cy), 0.12, color=NODE, zorder=2))
ax.text(2.2, 4.82, "each cell $\\rightarrow$ node", ha="center", fontsize=13,
        color="0.25")
ax.text(2.2, 4.46, "each face $\\rightarrow$ edge  (carries the flux)",
        ha="center", fontsize=13, color="0.25")

arrow(4.05, 5.65, 4.75, 5.65)

# ---------------------------------------------------------------- cell state
box(5.75, 5.65, 1.95, 1.6, GRY, fill="#efefef")
ax.text(5.75, 6.2, "cell state", ha="center", fontsize=13.5,
        fontweight="bold", color="0.1")
ax.text(5.75, 5.72, "$T,\\ \\rho,\\ \\rho_i$", ha="center", fontsize=13,
        color="0.2")
ax.text(5.75, 5.34, "$\\varphi,\\ p_g,\\ \\dot m$", ha="center", fontsize=13,
        color="0.2")
ax.text(5.75, 4.98, "(8 features)", ha="center", fontsize=10.5, color="0.45",
        style="italic")

# ---------------------------------------------------------------- 2. three pieces
ax.text(9.55, 7.25, "2.  one update = three pieces, summed", ha="center",
        fontsize=15, fontweight="bold", color="0.1")
bx, bw, bh = 9.55, 3.9, 0.9
ys = [6.55, 5.6, 4.65]
chans = [
    (FIX, "Conduction", "anchored, relaxed to a learnable $D$"),
    (FIX, "Pyrolysis", "Arrhenius kinetics from the file, fixed"),
    (LRN, "Gas coupling", "the only learned part, $\\sim$3 parameters"),
]
for (col, name, sub), y in zip(chans, ys):
    box(bx, y, bw, bh, col)
    ax.text(bx - bw / 2 + 0.3, y + 0.15, name, ha="left", fontsize=13.5,
            fontweight="bold", color=col)
    ax.text(bx - bw / 2 + 0.3, y - 0.22, sub, ha="left", fontsize=11.5,
            color="0.22")
for y in ys:
    arrow(6.75, 5.65, bx - bw / 2 - 0.04, y, color="0.7", lw=1.5)
for y in ys:
    arrow(bx + bw / 2 + 0.04, y, 11.85, 5.65, color="0.7", lw=1.5)

# ---------------------------------------------------------------- bounded update
arrow(11.95, 5.65, 12.3, 5.65)
box(13.3, 5.65, 2.2, 1.25, STR)
ax.text(13.3, 6.02, "bounded", ha="center", fontsize=13,
        fontweight="bold", color=STR)
ax.text(13.3, 5.72, "update", ha="center", fontsize=13,
        fontweight="bold", color=STR)
ax.text(13.3, 5.36, "$\\times$8 sub-steps", ha="center", fontsize=11, color="0.25")

# ---------------------------------------------------------------- 2b. transfer learning
box(7.25, 3.35, 13.6, 1.15, LRN, fill="#fbf0e7", lw=2.0)
ax.text(7.25, 3.68, "Transfer-learning training", ha="center", fontsize=13.5,
        fontweight="bold", color=LRN)
ax.text(7.25, 3.28, "relax the conduction anchor $+$ weight the loss toward the "
        "sharp flux transient", ha="center", fontsize=12, color="0.2")
ax.text(7.25, 2.94, "physics-informed 25 K  $\\rightarrow$  18 K held-out",
        ha="center", fontsize=12, fontweight="bold", color="0.1")

# ---------------------------------------------------------------- legend
ax.text(0.55, 2.2, "color key:", ha="left", va="center", fontsize=10.5,
        fontstyle="italic", color="0.35")
for lx, col, lab in [(2.2, FIX, "fixed to known physics"),
                     (6.1, LRN, "learned by the GNN"),
                     (9.6, STR, "structural guarantee")]:
    ax.add_patch(Rectangle((lx, 2.1), 0.28, 0.2, color=col))
    ax.text(lx + 0.4, 2.2, lab, ha="left", va="center", fontsize=11,
            color="0.3")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs/heat_TPS/figs/gnn_arch"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
print(f"saved {out}.pdf / .png")
