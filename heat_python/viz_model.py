"""One-page architecture schematic of the heat-shield surrogate (HeatMPGNN).

Draws: the graph (mesh as nodes+edges), one cell's K=1 message-passing update
(encoders -> messages -> node update -> output head), the ResMLPBlock building
block, and the time-rollout recurrence. Saved as a standalone figure.

    python -m heat_python.viz_model
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

C_IO = "#cfe8ff"      # inputs/outputs
C_ENC = "#d7f0d0"     # encoders
C_MLP = "#ffe2b3"     # MLPs
C_HID = "#eadcff"     # hidden / update
C_NODE = "#5b8def"    # mesh nodes


def box(ax, cx, cy, w, h, text, fc, fs=9, ec="#333"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.4,rounding_size=2",
                 fc=fc, ec=ec, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=5)


def arr(ax, x1, y1, x2, y2, color="#333", lw=1.4, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 98, "HeatMPGNN — heat-shield state surrogate",
            ha="center", fontsize=15, weight="bold")

    # ---------- Panel A: the graph ----------
    ax.text(2, 92, "1.  The graph: 1D mesh as nodes + edges  (weights shared "
            "across all cells → any mesh length)", fontsize=10.5, weight="bold")
    ys = 86
    xs = [10, 22, 34, 46, 58, 70, 82]
    for i, x in enumerate(xs):
        ghost = i in (0, len(xs) - 1)
        ax.add_patch(Circle((x, ys), 2.6, fc="white" if ghost else C_NODE,
                     ec="#333", lw=1.3, ls="--" if ghost else "-", zorder=4))
        if not ghost:
            ax.text(x, ys, f"c{i}", ha="center", va="center", fontsize=8,
                    color="white", zorder=5)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.plot([x1 + 2.6, x2 - 2.6], [ys, ys], "-", color="#777", lw=1.4, zorder=3)
    ax.text(xs[0], ys - 5.5, "ghost\n(inner BC)", ha="center", fontsize=7.5)
    ax.text(xs[-1], ys - 5.5, "ghost\n(outer/hot BC)", ha="center", fontsize=7.5)
    arr(ax, 92, ys, 85, ys, color="#c0392b", lw=2)
    ax.text(93.5, ys, "boundary\nforcing", ha="left", va="center",
            fontsize=8, color="#c0392b")
    ax.text(10, ys + 4.5, "inner (x=0, cool)", fontsize=7.5, ha="center")
    ax.text(82, ys + 4.5, "outer (x=L0, hot)", fontsize=7.5, ha="center")

    # ---------- Panel B: one cell's update ----------
    ax.text(2, 75, "2.  One cell's update  —  K=1 message passing "
            "(each cell hears from its 2 neighbors)", fontsize=10.5, weight="bold")
    # inputs
    box(ax, 12, 66, 17, 6, "node features\n[T, ρ, ρᵢ, φ]", C_IO, 8.5)
    box(ax, 12, 54, 17, 6, "left edge\n[ΔT, Δρ, …, dx]", C_IO, 8.5)
    box(ax, 12, 44, 17, 6, "right edge\n[ΔT, Δρ, …, dx]", C_IO, 8.5)
    # encoders
    box(ax, 35, 66, 15, 6, "Node\nEncoder", C_ENC, 8.5)
    box(ax, 35, 49, 15, 6, "Edge Encoder\n(shared)", C_ENC, 8.5)
    arr(ax, 20.5, 66, 27.5, 66)
    arr(ax, 20.5, 54, 27.5, 50.5); arr(ax, 20.5, 44, 27.5, 47.5)
    # messages
    box(ax, 55, 54, 14, 5.5, "Message MLP\n→ left msg", C_MLP, 8)
    box(ax, 55, 44, 14, 5.5, "Message MLP\n→ right msg", C_MLP, 8)
    arr(ax, 42.5, 50, 48, 53); arr(ax, 42.5, 48, 48, 45)
    # node update
    box(ax, 76, 57, 16, 8, "Node-Update MLP\n[self | left | right]", C_HID, 8.5)
    arr(ax, 42.5, 66, 68, 60)         # self hidden
    arr(ax, 62, 54, 68, 57.5)         # left msg
    arr(ax, 62, 44, 68, 54.5)         # right msg
    # output
    box(ax, 76, 44, 16, 6, "Output Head\n→ Δ[T, ρᵢ0, ρᵢ1]", C_IO, 8.5)
    arr(ax, 76, 53, 76, 47)
    ax.text(50, 37.5, "self-attention is the same idea on a fully-connected "
            "graph; here the graph is restricted to physical neighbors "
            "(local-physics inductive bias).", fontsize=8, style="italic",
            ha="center", color="#444")

    # ResMLPBlock inset
    bx, by = 92, 70
    ax.text(bx, by + 9, "ResMLPBlock", ha="center", fontsize=8.5, weight="bold")
    ax.text(bx, by + 6.6, "(every box above)", ha="center", fontsize=7,
            color="#555")
    for j, (lbl, c) in enumerate([("LayerNorm", "#eee"), ("Linear→GELU", C_MLP),
                                  ("Linear", C_MLP)]):
        box(ax, bx, by + 3 - j * 3.2, 13, 2.6, lbl, c, 7.5)
    arr(ax, bx + 7, by + 5, bx + 9.5, by + 5, color="#888", lw=1)
    arr(ax, bx + 9.5, by + 5, bx + 9.5, by - 6.6, color="#888", lw=1, style="-")
    arr(ax, bx + 9.5, by - 6.6, bx + 7, by - 6.6, color="#888", lw=1)
    ax.text(bx + 10.3, by - 1, "+\nresidual", fontsize=7, color="#888", va="center")
    ax.text(bx, by - 10, "≈ a transformer\nFFN block", ha="center", fontsize=7,
            style="italic", color="#444")

    # ---------- Panel C: rollout ----------
    ax.text(2, 30, "3.  Rollout in time  —  a recurrence (trained by "
            "backprop-through-time over M steps)", fontsize=10.5, weight="bold")
    box(ax, 16, 20, 20, 8, "state(t)\nall cells", C_IO, 9)
    box(ax, 47, 20, 24, 8, "HeatMPGNN\n(apply to every cell)\n+ Δ",
        C_HID, 8.5)
    box(ax, 78, 20, 20, 8, "state(t+Δt)", C_IO, 9)
    arr(ax, 26.5, 20, 34.5, 20)
    arr(ax, 59.5, 20, 67.5, 20)
    # feedback loop
    arr(ax, 78, 16, 78, 9, color="#c0392b", lw=1.6, style="-")
    ax.plot([16, 78], [9, 9], "-", color="#c0392b", lw=1.6)
    arr(ax, 16, 9, 16, 16, color="#c0392b", lw=1.6)
    ax.text(47, 7, "feed prediction back in (autoregressive, like an RNN)",
            ha="center", fontsize=8, color="#c0392b")
    ax.text(47, 27.5, "ghost forcing re-imposed each step; "
            "ρ/porosity derived from predicted species",
            ha="center", fontsize=7.5, style="italic", color="#444")

    out = Path(__file__).resolve().parents[1] / "heat_python" / "figs" / "model_onepager.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
