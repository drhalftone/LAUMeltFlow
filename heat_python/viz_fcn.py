"""Show the HeatMPGNN as (left) a GNN dataflow and (right) the fully-connected
MLP inside one block. Answers: it's a GNN on the outside (wiring that gathers a
cell + its two neighbors) built from fully-connected networks on the inside.

    python -m heat_python.viz_fcn
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="0.3", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color="0.35", lw=1.3,
                                 shrinkA=2, shrinkB=2))


def panel_gnn(ax):
    ax.set_title("(a) GNN dataflow for one cell\n(the wiring: gather self + 2 neighbors)",
                 fontsize=10)
    enc = "#cfe3f7"; mp = "#d7f0d8"; out = "#f7d9cf"
    # inputs
    box(ax, 0.02, 0.78, 0.26, 0.1, "node feat\n[T,ρ,ρᵢ,por]  (6)", "#eef4fb")
    box(ax, 0.02, 0.50, 0.26, 0.1, "left edge\n[Δstate,Δx]  (7)", "#eef4fb")
    box(ax, 0.02, 0.22, 0.26, 0.1, "right edge\n[Δstate,Δx]  (7)", "#eef4fb")
    # encoders
    box(ax, 0.36, 0.78, 0.22, 0.1, "node\nencoder", enc)
    box(ax, 0.36, 0.50, 0.22, 0.1, "edge\nencoder", enc)
    box(ax, 0.36, 0.22, 0.22, 0.1, "edge\nencoder", enc)
    ax.text(0.47, 0.14, "(shared weights)", ha="center", fontsize=7, color="0.4")
    # message MLPs
    box(ax, 0.64, 0.50, 0.18, 0.1, "msg\nMLP", mp)
    box(ax, 0.64, 0.22, 0.18, 0.1, "msg\nMLP", mp)
    # node update + output
    box(ax, 0.64, 0.74, 0.18, 0.14, "node\nupdate\nMLP", mp)
    box(ax, 0.86, 0.74, 0.12, 0.14, "output\nhead", out)
    ax.text(0.92, 0.66, "Δ[T,ρᵢ]\n(3)", ha="center", fontsize=8)

    for y in (0.83, 0.55, 0.27):
        arrow(ax, 0.28, y, 0.36, y)
    arrow(ax, 0.58, 0.55, 0.64, 0.55)      # edge enc -> msg
    arrow(ax, 0.58, 0.27, 0.64, 0.27)
    arrow(ax, 0.58, 0.83, 0.64, 0.81)      # node enc -> node update
    arrow(ax, 0.73, 0.60, 0.71, 0.74)      # left msg -> node update
    arrow(ax, 0.73, 0.32, 0.715, 0.74)     # right msg -> node update
    arrow(ax, 0.82, 0.81, 0.86, 0.81)      # node update -> output
    ax.text(0.5, 0.025, "Every box is a fully-connected MLP  →  see (b)",
            ha="center", fontsize=8.5, style="italic", color="#a8421f")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def panel_fcn(ax):
    ax.set_title("(b) Inside one block: a fully-connected MLP\n"
                 "(each line = one learned weight; all-to-all between layers)",
                 fontsize=10)
    # representative layers (drawn small; real widths labeled)
    layers = [(0.12, 6, "in"), (0.40, 9, "hidden\n(128)"), (0.68, 6, "out\n(64)")]
    pos = []
    for lx, nshow, lbl in layers:
        ys = np.linspace(0.18, 0.82, nshow)
        pos.append([(lx, y) for y in ys])
        ax.text(lx, 0.06, lbl, ha="center", fontsize=8.5)
    # full connections
    for a, b in [(0, 1), (1, 2)]:
        for (x1, y1) in pos[a]:
            for (x2, y2) in pos[b]:
                ax.plot([x1, x2], [y1, y2], "-", color="0.75", lw=0.4, zorder=1)
    # neurons
    cols = ["#1f6fb2", "#2e7d32", "#c0392b"]
    for li, layer in enumerate(pos):
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), 0.022, fc=cols[li], ec="0.3",
                                    lw=0.6, zorder=3))
    ax.annotate("", xy=(0.86, 0.5), xytext=(0.74, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3))
    ax.text(0.90, 0.5, "Δ", fontsize=12, va="center")
    ax.text(0.5, 0.92, "(dots shown reduced; true widths 64–128)",
            ha="center", fontsize=7.5, color="0.4")
    ax.set_xlim(0.02, 0.98); ax.set_ylim(0, 1); ax.axis("off")


def eq_strip(fig):
    """Ordered equations for the pipeline, across the bottom of the figure."""
    y0 = 0.235
    fig.text(0.5, y0, "What each part computes (per cell $i$, one step):",
             ha="center", fontsize=10.5, weight="bold")
    lines = [
        r"$\mathbf{1.\ encode:}\ \ h_i=f_{node}(x_i)\ \ \ \ "
        r"m_L=f_{edge}([\,x_{i-1}-x_i,\ \Delta x\,])\ \ \ \ m_R=f_{edge}([\,x_{i+1}-x_i,\ \Delta x\,])$",
        r"$\mathbf{2.\ message:}\ \ a_L=g(m_L)\ \ \ \ a_R=g(m_R)$",
        r"$\mathbf{3.\ update:}\ \ h_i'=u([\,h_i\ |\ a_L\ |\ a_R\,])$",
        r"$\mathbf{4.\ output:}\ \ \Delta=W\,\mathrm{LayerNorm}(h_i')\ \ \ \ "
        r"x_i^{\,new}=x_i+\Delta$",
        r"Each $f,\,g,\,u,\,W$ is a fully-connected MLP. One MLP layer (panel b):"
        r"$\ \ h_{out}=\mathrm{GELU}(W\,h_{in}+b)$  — each neuron = weighted sum of all inputs, then a nonlinearity.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.06, y0 - 0.035 - i * 0.038, ln, ha="left", fontsize=9.2,
                 color="#222" if i < 4 else "#a8421f")


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 7.2))
    panel_gnn(axA)
    panel_fcn(axB)
    # leave the bottom third for equations
    for ax in (axA, axB):
        ax.set_position([ax.get_position().x0, 0.42,
                         ax.get_position().width, 0.5])
    eq_strip(fig)
    fig.suptitle("HeatMPGNN — a GNN built from fully-connected MLPs (80,451 params, width 64, K=1)",
                 fontsize=12, y=0.99)
    out = Path(__file__).resolve().parents[1] / "heat_python" / "figs" / "fcn.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
