"""Explainer: two ways to build known physics into a learned PDE surrogate.

Left  -- PINN (Cao 2026): physics as a SOFT LOSS PENALTY. The net maps (Z,t)->u
         and is penalized for violating the PDE/BC. It CAN violate them, and at
         sharp/discontinuous boundaries the error accumulates and corrupts the field.
Right -- ours: physics as a HARD ARCHITECTURAL CONSTRAINT. Fixed conduction, fixed
         Arrhenius, and a maximum-principle-bounded update. The net learns only the
         residual gas coupling and CANNOT blow up by construction.

-> figs/architecture/approach_compare.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PINN = "#6d4fa8"     # soft / loss-based (purple)
FIX = "#3b8f5a"      # hard physics (green)
LRN = "#c0611f"      # learned (orange)
GRY = "0.45"

fig, ax = plt.subplots(figsize=(13, 9.6))
ax.set_xlim(0, 14); ax.set_ylim(-1.3, 12.4); ax.axis("off")

ax.text(7, 11.9, "Two ways to build known physics into a learned surrogate",
        ha="center", fontsize=17, fontweight="bold")
ax.text(7, 11.45, "both embed the governing physics; they differ in WHERE it goes "
        "and whether the network can violate it",
        ha="center", fontsize=10.8, color="0.35", style="italic")

# column headers
ax.text(3.4, 10.7, "Physics in the LOSS", ha="center", fontsize=13.5,
        fontweight="bold", color=PINN)
ax.text(3.4, 10.34, "PINN  (Cao 2026, transfer-learning)", ha="center",
        fontsize=10, color=PINN, style="italic")
ax.text(10.6, 10.7, "Physics in the ARCHITECTURE", ha="center", fontsize=13.5,
        fontweight="bold", color=FIX)
ax.text(10.6, 10.34, "ours  (conductance GNN)", ha="center", fontsize=10,
        color=FIX, style="italic")

# vertical divider
ax.plot([7, 7], [-0.9, 10.15], color="0.8", lw=1.3, ls="--")


def box(cx, y, w, h, col, lines, fill=None, fs=10.2, lead=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, y - h), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=2, edgecolor=col,
                 facecolor=(fill if fill else col + "14")))
    if isinstance(lines, str):
        lines = [lines]
    y0 = y - 0.33
    for i, (txt, bold, tc) in enumerate(lines):
        ax.text(cx, y0 - i * (lead or 0.35), txt, ha="center", va="top",
                fontsize=fs, fontweight=("bold" if bold else "normal"), color=tc)


def arrow(cx, y0, y1, col="0.5"):
    ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="-|>",
                 mutation_scale=15, color=col, lw=1.7))


# ---------------- LEFT: PINN ----------------
cx = 3.4
box(cx, 9.9, 5.2, 0.95, PINN,
    [("Neural net:  (Z, t)  ->  u", True, "0.1"),
     ("fully-connected, mesh-free", False, "0.35")])
arrow(cx, 8.95, 8.55)
box(cx, 8.55, 5.2, 1.72, PINN,
    [("Physics as a LOSS PENALTY", True, PINN),
     (r"$L = w_f\,\Vert$PDE residual$\Vert + w_b\,\Vert$BC$\Vert$", False, "0.15"),
     ("the net is punished for violating", False, "0.35"),
     ("the physics, but it still can", False, "0.35")])
arrow(cx, 6.83, 6.43)
box(cx, 6.43, 5.2, 1.25, "0.55",
    [("Soft constraint", True, "0.1"),
     ("physics can be broken, and error", False, "0.3"),
     ("builds up at sharp boundaries", False, "0.3")], fill="0.92")
arrow(cx, 5.18, 4.78)
box(cx, 4.78, 5.2, 1.25, PINN,
    [("Sharp transient (rain on/off):", True, "0.1"),
     ("staged transfer learning", False, "0.3"),
     ("(freeze layers, reweight loss)", False, "0.3")])
arrow(cx, 3.53, 3.13)
box(cx, 3.13, 5.2, 0.78, LRN,
    [("Learns the ENTIRE solution field", True, "0.1")])

# ---------------- RIGHT: ours ----------------
cx = 10.6
box(cx, 9.9, 5.2, 0.95, FIX,
    [("GNN on the mesh graph:", True, "0.1"),
     ("one node per cell, learns a local update", False, "0.35")])
arrow(cx, 8.95, 8.55)
box(cx, 8.55, 5.2, 1.72, FIX,
    [("Physics in the UPDATE", True, FIX),
     ("fixed conduction D  +  fixed Arrhenius", False, "0.15"),
     ("+ maximum-principle bound", False, "0.15"),
     ("the physics is hard-wired, not penalized", False, "0.35")])
arrow(cx, 6.83, 6.43)
box(cx, 6.43, 5.2, 1.25, "0.55",
    [("Hard constraint", True, "0.1"),
     ("physics cannot be broken. Each", False, "0.3"),
     ("update is a bounded neighbor blend", False, "0.3")], fill="0.92")
arrow(cx, 5.18, 4.78)
box(cx, 4.78, 5.2, 1.25, FIX,
    [("Sharp transient (flux on/off):", True, "0.1"),
     ("sub-stepping + bounded update", False, "0.3"),
     ("(+ temporal loss weight, from Cao)", False, "0.3")])
arrow(cx, 3.53, 3.13)
box(cx, 3.13, 5.2, 0.78, LRN,
    [("Learns ONLY the gas coupling  (~3 params)", True, "0.1")])

# ---------------- bottom contrast bar ----------------
ax.add_patch(FancyBboxPatch((0.6, -0.9), 12.8, 2.05,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             linewidth=1.6, edgecolor="0.5", facecolor="0.965"))
ax.text(7, 0.85, "Same goal, different guarantee", ha="center", fontsize=12,
        fontweight="bold", color="0.15")
ax.text(3.4, 0.35,
        "PINN: physics is a soft target in the loss.\n"
        "Flexible and mesh-free, but the solution can\n"
        "drift, and sharp boundaries need extra tricks\n"
        "(transfer learning) to stay stable.",
        ha="center", va="top", fontsize=9.6, color="0.2", linespacing=1.4)
ax.text(10.6, 0.35,
        "Ours: physics is a hard part of the update.\n"
        "Less flexible, but the field CANNOT blow up and\n"
        "the char front is exact by construction. The net\n"
        "only has to learn the small unknown piece.",
        ha="center", va="top", fontsize=9.6, color="0.2", linespacing=1.4)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "heat_python/figs/architecture/approach_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
