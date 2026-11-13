"""Generate standalone MLP architecture diagram matching MPGNN font sizes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

def box(ax, x, y, w, h, text, fc, ec, fontsize=14, bold=True):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.15",
                       fc=fc, ec=ec, lw=2, zorder=3)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color="black", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="gray"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.15",
                                color=color, lw=2.5))

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3, 4.5)
ax.set_title("Shared MLP", fontsize=24, fontweight="bold", pad=12)
ax.axis("off")

# Three input sources — same font sizes as MPGNN (13pt for inputs)
box(ax, -2, 3.5, 2.2, 1.0, "Left Neighbor\n6 features", "#E8F5E9", "#4CAF50", fontsize=13)
box(ax, 0, 3.5, 2.2, 1.0, "Bead $i$\n4 features", "#E3F2FD", "#2196F3", fontsize=13)
box(ax, 2, 3.5, 2.2, 1.0, "Right Neighbor\n6 features", "#F3E5F5", "#9C27B0", fontsize=13)

# Arrows to concat
arrow(ax, -2, 2.95, -0.3, 2.3, "#4CAF50")
arrow(ax, 0, 2.95, 0, 2.3, "#2196F3")
arrow(ax, 2, 2.95, 0.3, 2.3, "#9C27B0")

# Concatenate — same as MPGNN concat box (13pt)
box(ax, 0, 1.7, 5, 0.9, "Concatenate into 16 features", "#FFF3E0", "#FF9800", fontsize=13)

# Arrow
arrow(ax, 0, 1.2, 0, 0.5)

# MLP — same as MPGNN Node Update MLP (13pt)
box(ax, 0, -0.1, 4.5, 1.0, "MLP\n16 → 64 → 64 → 4", "#E3F2FD", "#2196F3", fontsize=13)

# Arrow
arrow(ax, 0, -0.65, 0, -1.3)

# Output — same as MPGNN output (15pt)
box(ax, 0, -1.9, 4.5, 0.9, "$\\Delta x,\\ \\Delta y,\\ \\Delta v_x,\\ \\Delta v_y$", "#E8F5E9", "#4CAF50", fontsize=15)

plt.tight_layout()
plt.savefig("architecture_mlp_only.png", dpi=150, bbox_inches="tight")
print("Saved architecture_mlp_only.png")
