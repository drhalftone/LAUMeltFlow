"""Generate architecture diagram for the HGNN."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_diagram():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Colors
    C_INPUT = "#E3F2FD"
    C_GNN = "#C8E6C9"
    C_HAM = "#FFF9C4"
    C_CONSTRAINT = "#FFCCBC"
    C_INTEGRATOR = "#E1BEE7"
    C_OUTPUT = "#F0F0F0"
    C_KNOWN = "#B3E5FC"
    C_BORDER = "#555555"

    def box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor=C_BORDER, linewidth=1.2)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#333333", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5))

    def label(x, y, text, fontsize=8, color="#666666", ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color)

    # =========================================================
    # Title
    # =========================================================
    ax.text(8, 10.6, "Hamiltonian Graph Neural Network — Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # =========================================================
    # Row 1: Inputs (top)
    # =========================================================
    box(1, 9.2, 2.8, 0.8, "Positions q\n(B, 16, 2)", C_INPUT, 9)
    box(4.5, 9.2, 2.8, 0.8, "Momenta p\n(B, 16, 2)", C_INPUT, 9)
    box(8, 9.2, 2.8, 0.8, "Masses m\n(16,) known", C_KNOWN, 9)
    box(11.5, 9.2, 2.8, 0.8, "Rest lengths L\u2080\n(15,) known", C_KNOWN, 9)

    # =========================================================
    # Row 2: Two branches — T (known) and V (learned)
    # =========================================================

    # --- T branch (right side) ---
    box(4.5, 7.4, 2.8, 0.9, "Kinetic Energy T\n|p|² / (2m)\nHARDCODED", C_KNOWN, 9, bold=True)
    arrow(5.9, 9.2, 5.9, 8.3)  # p -> T

    # --- V branch (left side) — the GNN ---
    # GNN box (large)
    gnn_rect = FancyBboxPatch((0.5, 3.6), 3.6, 4.5, boxstyle="round,pad=0.2",
                               facecolor=C_GNN, edgecolor="#388E3C", linewidth=2,
                               alpha=0.3)
    ax.add_patch(gnn_rect)
    ax.text(2.3, 7.85, "PotentialEnergyGNN (LEARNED)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2E7D32")

    # Node encoder
    box(0.8, 7.0, 3.0, 0.6, "Node Encoder\n[q_i | m_i | fixed_i] → h_i", C_GNN, 8)
    arrow(2.4, 9.2, 2.3, 7.6)  # q -> node encoder

    # Edge encoder
    box(0.8, 6.1, 3.0, 0.6, "Edge Encoder\n[Δq | dist | L₀] → h_edge", C_GNN, 8)

    # Message passing (×3)
    box(0.8, 4.8, 3.0, 1.0,
        "Message Passing ×3\n\n"
        "left_msg = MLP([h_left | h_edge_L])\n"
        "right_msg = MLP([h_right | h_edge_R])\n"
        "h_i = MLP([h_i | left_msg | right_msg])",
        C_GNN, 7)
    arrow(2.3, 6.1, 2.3, 5.8)
    arrow(2.3, 7.0, 2.3, 6.7)

    # Energy readout
    box(0.8, 3.9, 3.0, 0.6, "Energy Readout\nh_i → scalar e_i,  V = Σ e_i", C_GNN, 8)
    arrow(2.3, 4.8, 2.3, 4.5)

    # =========================================================
    # Row 3: Hamiltonian
    # =========================================================
    box(3.2, 2.6, 4.4, 0.8, "H(q, p) = T(p) + V(q)\nscalar", C_HAM, 10, bold=True)
    arrow(2.3, 3.9, 4.5, 3.4)   # V -> H
    arrow(5.9, 7.4, 5.9, 3.4)   # T -> H

    # =========================================================
    # Row 4: Autograd
    # =========================================================
    box(8.5, 5.5, 3.5, 1.2,
        "Hamilton's Equations\n(via autograd)\n\n"
        "dq/dt =  ∂H/∂p\n"
        "dp/dt = −∂H/∂q",
        C_HAM, 9, bold=False)
    arrow(7.6, 3.0, 8.5, 5.8)   # H -> Hamilton's eqs

    # =========================================================
    # Row 5: Constraint projection
    # =========================================================
    box(8.5, 3.6, 3.5, 1.2,
        "Constraint Projection\n(Lagrange multipliers)\n\n"
        "Φ = ||q_j−q_i||² − L₀² = 0\n"
        "Project dz/dt onto Φ=0",
        C_CONSTRAINT, 8)
    arrow(10.25, 5.5, 10.25, 4.8)  # eqs -> constraint

    # =========================================================
    # Row 6: Integrator
    # =========================================================
    box(8.5, 2.0, 3.5, 1.0,
        "Leapfrog Integrator\n(symplectic)\n"
        "p½ → q_new → p_new",
        C_INTEGRATOR, 9, bold=True)
    arrow(10.25, 3.6, 10.25, 3.0)  # constraint -> integrator

    # =========================================================
    # Row 7: Output
    # =========================================================
    box(8.5, 0.6, 3.5, 0.8, "Next state\n(q_{n+1}, p_{n+1})", C_OUTPUT, 10)
    arrow(10.25, 2.0, 10.25, 1.4)

    # Feedback loop arrow (output -> input)
    ax.annotate("", xy=(14, 9.6), xytext=(14, 1.0),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.5,
                                connectionstyle="arc3,rad=0.0",
                                linestyle="dashed"))
    ax.annotate("", xy=(12, 9.6), xytext=(14, 9.6),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.5,
                                linestyle="dashed"))
    label(14.8, 5.3, "loop for\neach\ntimestep", fontsize=8, color="#999999")

    # =========================================================
    # Bead chain diagram (bottom-left)
    # =========================================================
    # Draw a small chain of beads
    chain_y = 1.3
    chain_x0 = 0.5
    n_show = 8
    spacing = 0.4

    # Title
    ax.text(chain_x0 + n_show * spacing / 2, 2.3, "Chain topology (16 beads)",
            ha="center", fontsize=8, fontstyle="italic", color="#666666")

    for i in range(n_show):
        cx = chain_x0 + i * spacing
        # Rod
        if i < n_show - 1:
            ax.plot([cx, cx + spacing], [chain_y, chain_y], color="#888888", lw=2)
        # Bead
        color = "#F44336" if i == 0 else "#2196F3"
        circle = plt.Circle((cx, chain_y), 0.1, color=color, ec="black", lw=1, zorder=5)
        ax.add_patch(circle)
        if i == 0:
            ax.text(cx, chain_y - 0.3, "fixed", ha="center", fontsize=6, color="#F44336")

    # Dots for remaining beads
    ax.text(chain_x0 + n_show * spacing, chain_y, "···", fontsize=12, va="center")

    # Message arrows
    ax.text(chain_x0 + 1.5 * spacing, chain_y + 0.35, "← left msg    right msg →",
            ha="center", fontsize=6, color="#2E7D32")

    # =========================================================
    # Legend
    # =========================================================
    legend_x = 12.5
    legend_y = 3.8
    ax.text(legend_x, legend_y + 1.8, "Legend", fontsize=9, fontweight="bold")
    items = [
        (C_GNN, "Learned (GNN)"),
        (C_KNOWN, "Known / hardcoded"),
        (C_HAM, "Hamiltonian"),
        (C_CONSTRAINT, "Constraint projection"),
        (C_INTEGRATOR, "Symplectic integrator"),
    ]
    for idx, (color, text) in enumerate(items):
        y = legend_y + 1.2 - idx * 0.4
        rect = FancyBboxPatch((legend_x, y - 0.12), 0.3, 0.24,
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor=C_BORDER, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y, text, fontsize=8, va="center")

    plt.tight_layout()
    out_path = "architecture.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    draw_diagram()
