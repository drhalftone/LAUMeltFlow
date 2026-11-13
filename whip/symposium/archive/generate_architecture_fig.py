"""Simple side-by-side architecture comparison: MLP vs MPGNN.
Clean block diagram a grad student can follow in 30 seconds."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 16})

fig, (ax_mlp, ax_gnn) = plt.subplots(2, 1, figsize=(10, 18))

def box(ax, x, y, w, h, text, fc, ec, fontsize=14, textcolor="black", bold=True):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.15",
                       fc=fc, ec=ec, lw=2, zorder=3)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="gray"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.15",
                                color=color, lw=2.5))

# ============================================================
# LEFT: MLP
# ============================================================
ax_mlp.set_xlim(-3, 3)
ax_mlp.set_ylim(-5.5, 5)
ax_mlp.set_title("Shared MLP", fontsize=24, fontweight="bold", pad=15)
ax_mlp.axis("off")

# Three input sources at top
box(ax_mlp, -1.8, 4, 2.2, 0.9, "Left Neighbor\n6 features", "#E8F5E9", "#4CAF50", fontsize=13)
box(ax_mlp, 0, 4, 2.2, 0.9, "Bead $i$\n4 features", "#E3F2FD", "#2196F3", fontsize=13)
box(ax_mlp, 1.8, 4, 2.2, 0.9, "Right Neighbor\n6 features", "#F3E5F5", "#9C27B0", fontsize=13)

# Arrows down to concat
arrow(ax_mlp, -1.8, 3.5, -0.3, 2.7, "#4CAF50")
arrow(ax_mlp, 0, 3.5, 0, 2.7, "#2196F3")
arrow(ax_mlp, 1.8, 3.5, 0.3, 2.7, "#9C27B0")

# Concatenate
box(ax_mlp, 0, 2.2, 4, 0.8, "Concatenate into 16 features", "#FFF3E0", "#FF9800", fontsize=14)

# Arrow
arrow(ax_mlp, 0, 1.75, 0, 1.0)

# MLP
box(ax_mlp, 0, 0.4, 3.5, 1.0, "MLP\n16 → 64 → 64 → 4", "#E3F2FD", "#2196F3", fontsize=15)

# Arrow
arrow(ax_mlp, 0, -0.15, 0, -0.9)

# Output
box(ax_mlp, 0, -1.5, 3.5, 0.8, "$\\Delta x,\\ \\Delta y,\\ \\Delta v_x,\\ \\Delta v_y$", "#E8F5E9", "#4CAF50", fontsize=15)

# Note
ax_mlp.text(0, -3.5, "All features mixed together.\nFixed to one chain length.",
            ha="center", fontsize=14, style="italic", color="#C62828",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="#EF9A9A", lw=1.5))


# ============================================================
# RIGHT: MPGNN
# ============================================================
ax_gnn.set_xlim(-4, 4)
ax_gnn.set_ylim(-5.5, 5)
ax_gnn.set_title("BeadMPGNN", fontsize=24, fontweight="bold", pad=15)
ax_gnn.axis("off")

# Three input sources at top
box(ax_gnn, -2.5, 4, 2.5, 0.9, "Left Edge\n6 features", "#E8F5E9", "#4CAF50", fontsize=13)
box(ax_gnn, 0, 4, 2.5, 0.9, "Bead $i$\n4 features", "#E3F2FD", "#2196F3", fontsize=13)
box(ax_gnn, 2.5, 4, 2.5, 0.9, "Right Edge\n6 features", "#F3E5F5", "#9C27B0", fontsize=13)

# Arrows to encoders
arrow(ax_gnn, -2.5, 3.5, -2.5, 2.7, "#4CAF50")
arrow(ax_gnn, 0, 3.5, 0, 2.7, "#2196F3")
arrow(ax_gnn, 2.5, 3.5, 2.5, 2.7, "#9C27B0")

# Separate encoders
box(ax_gnn, -2.5, 2.2, 2.5, 0.8, "Edge Encoder\n6 → 64", "#E8F5E9", "#4CAF50", fontsize=13)
box(ax_gnn, 0, 2.2, 2.5, 0.8, "Node Encoder\n4 → 64", "#E3F2FD", "#2196F3", fontsize=13)
box(ax_gnn, 2.5, 2.2, 2.5, 0.8, "Edge Encoder\n6 → 64", "#F3E5F5", "#9C27B0", fontsize=13)

# Arrows to message MLPs
arrow(ax_gnn, -2.5, 1.75, -2.5, 1.0, "#4CAF50")
arrow(ax_gnn, 2.5, 1.75, 2.5, 1.0, "#9C27B0")

# Message MLPs (only for edges)
box(ax_gnn, -2.5, 0.5, 2.5, 0.7, "Message MLP\n64 → 64", "#E8F5E9", "#4CAF50", fontsize=13)
box(ax_gnn, 2.5, 0.5, 2.5, 0.7, "Message MLP\n64 → 64", "#F3E5F5", "#9C27B0", fontsize=13)

# Ghost mask labels
ax_gnn.text(-2.5, -0.15, "× ghost mask", ha="center", fontsize=12, style="italic", color="gray")
ax_gnn.text(2.5, -0.15, "× ghost mask", ha="center", fontsize=12, style="italic", color="gray")

# Arrows converging to concat
arrow(ax_gnn, -2.5, -0.4, -0.5, -0.9, "#4CAF50")
arrow(ax_gnn, 0, 1.75, 0, -0.9, "#2196F3")
arrow(ax_gnn, 2.5, -0.4, 0.5, -0.9, "#9C27B0")

# Concatenate
box(ax_gnn, 0, -1.5, 5, 0.8, "Concatenate  [self | left msg | right msg]\n192 features", "#FFF3E0", "#FF9800", fontsize=13)

# Arrow
arrow(ax_gnn, 0, -1.95, 0, -2.6)

# Node update
box(ax_gnn, 0, -3.1, 4.5, 0.8, "Node Update MLP → Output Head\n192 → 64 → 4", "#E3F2FD", "#2196F3", fontsize=13)

# Arrow
arrow(ax_gnn, 0, -3.55, 0, -4.2)

# Output
box(ax_gnn, 0, -4.7, 3.5, 0.7, "$\\Delta x,\\ \\Delta y,\\ \\Delta v_x,\\ \\Delta v_y$", "#E8F5E9", "#4CAF50", fontsize=15)

# Note
ax_gnn.text(4.3, 0.5, "Shared\nweights", ha="center", fontsize=13, color="#1565C0", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9", lw=1.5))

ax_gnn.annotate("", xy=(3.8, 2.2), xytext=(4.3, 1.0),
                arrowprops=dict(arrowstyle="->, head_width=0.12", color="#90CAF9", lw=1.5))
ax_gnn.annotate("", xy=(3.8, 0.5), xytext=(4.3, 0.8),
                arrowprops=dict(arrowstyle="->, head_width=0.12", color="#90CAF9", lw=1.5))

plt.tight_layout(h_pad=4)
plt.savefig("architecture_comparison.png", dpi=150, bbox_inches="tight")
print("Saved architecture_comparison.png")
