"""What the conservative flux-form changes, architecturally.

The graph, node/edge encoders, and message passing are IDENTICAL to the direct
HeatMPGNN (viz_model.py). Only the output head differs. This figure contrasts the
two heads so the change is visible at a glance.

    python -m heat_python.viz_flux_model
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_IO = "#cfe8ff"
C_MLP = "#ffe2b3"
C_HID = "#eadcff"
C_BAD = "#f6c9c4"
C_GOOD = "#cdebcd"


def box(ax, cx, cy, w, h, text, fc, fs=9, ec="#333"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.4,rounding_size=2", fc=fc, ec=ec, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=5)


def arr(ax, x1, y1, x2, y2, color="#333", lw=1.4, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97, "Flux-form variant: only the output head changes",
            ha="center", fontsize=15, weight="bold")
    ax.text(50, 91.5, "The graph, node/edge encoders, and message passing are "
            "identical to the direct model; the cell's hidden state is the same input.",
            ha="center", fontsize=9.5, style="italic", color="#444")

    # Shared input
    box(ax, 17, 60, 24, 12, "shared front-end\n\nencoded cell $h_i$\n+ encoded "
        "neighbors\n$h_{i-1}, h_{i+1}$", C_HID, 9)
    ax.text(17, 50.5, "(message passing, K=1)", ha="center", fontsize=8,
            color="#555")

    # ---- Direct head (top lane) ----
    ax.text(40, 82, "DIRECT  (HeatMPGNN)", fontsize=11, weight="bold", color="#9b3b32")
    box(ax, 52, 76, 18, 8, "Node-Update MLP\n[self | left | right]", C_HID, 8.5)
    box(ax, 78, 76, 16, 8, "Output Head\n$\\rightarrow \\Delta_i$", C_IO, 9)
    arr(ax, 29, 64, 43, 76)
    arr(ax, 61, 76, 70, 76)
    box(ax, 93, 76, 9, 8, "$\\Delta_i$", C_BAD, 11)
    arr(ax, 86, 76, 88.5, 76)
    ax.text(78, 69.5, "predicts the change directly", ha="center", fontsize=8,
            style="italic", color="#555")
    ax.text(93, 70, "no conservation;\nstill cell can leak\na bias $\\rightarrow$ drift",
            ha="center", fontsize=7.5, color="#9b3b32")

    # divider
    ax.plot([30, 100], [55, 55], ":", color="#bbb", lw=1)

    # ---- Flux-form head (bottom lane) ----
    ax.text(40, 48, "FLUX-FORM  (HeatFluxMPGNN)", fontsize=11, weight="bold",
            color="#2e7d32")
    # antisymmetric face flux
    box(ax, 52, 40, 22, 10,
        "Flux MLP  $\\psi$  (per face)\n$\\Phi(a,b)=\\psi(a,b)-\\psi(b,a)$",
        C_MLP, 8.5)
    ax.text(52, 33.5, "antisymmetric: $\\Phi(a,b)=-\\Phi(b,a)$", ha="center",
            fontsize=7.5, color="#2e7d32")
    # source
    box(ax, 52, 22, 22, 7, "Source MLP $\\rightarrow S_i$\n(local pyrolysis)", C_MLP, 8.5)
    arr(ax, 29, 58, 41, 41)        # to flux
    arr(ax, 29, 56, 41, 23)        # to source
    # combine
    box(ax, 82, 31, 20, 12,
        "$\\Delta_i =$\n$(F_{i-1/2}-F_{i+1/2})$\n$+\\, S_i$", C_GOOD, 9.5)
    arr(ax, 63, 40, 72, 33)
    arr(ax, 63, 22, 72, 29)

    # callouts
    ax.text(50, 11, "built in by construction:", ha="center", fontsize=9,
            weight="bold", color="#2e7d32")
    ax.text(50, 7, "one flux per face, used with opposite signs in the two cells "
            "$\\Rightarrow$ CONSERVATION", ha="center", fontsize=8.5, color="#2e7d32")
    ax.text(50, 3.5, "uniform field $\\Rightarrow \\Phi=0 \\Rightarrow$ a quiescent "
            "cell changes by exactly zero (no bias to accumulate)",
            ha="center", fontsize=8.5, color="#2e7d32")

    out = Path(__file__).resolve().parents[1] / "heat_python" / "figs" / "flux_model_onepager.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
