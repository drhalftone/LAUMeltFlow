"""Architecture diagram: the physics-informed CONDUCTANCE model (aw2, working).
Explains what is FIXED physics vs. what the GNN LEARNS, and the two structural
guarantees (maximum principle -> unconditional stability; conservation + zero
baseline). Config/numbers pulled from heat_cond_aw2_fine_phys3.pt.
-> figs/architecture/model_conductance_arch.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIX = "#3b8f5a"    # fixed physics (green)
LRN = "#c0611f"    # learned by the GNN (orange)
STR = "#6d4fa8"    # structural guarantee (purple)
GRY = "0.45"

fig, ax = plt.subplots(figsize=(10.2, 14.6))
ax.set_xlim(0, 12); ax.set_ylim(-6.7, 13.0); ax.axis("off")

ax.text(6, 12.5, "Physics-informed conductance GNN (aw2 model)", ha="center",
        fontsize=17, fontweight="bold")
ax.text(6, 12.05, "known physics is built into the update; the GNN learns only the gas coupling",
        ha="center", fontsize=10.8, color="0.35", style="italic")

# ---------- input row ----------
def box(x, y, w, h, col, title, sub, fill=None, dark=False):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=2, edgecolor=col,
                 facecolor=(fill if fill else col + "16")))
    tcol = "white" if dark else "0.08"
    scol = "0.88" if dark else "0.18"
    ax.text(x + w / 2, y - 0.30, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y - 0.66, sub, ha="center", va="top",
            fontsize=9.0, color=scol, style="italic")

def arrow(x0, y0, x1, y1, col="0.5", lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=15, color=col, lw=lw))

# Input state
ax.add_patch(FancyBboxPatch((1.6, 10.55), 8.8, 1.05,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=2, edgecolor=GRY, facecolor=GRY))
ax.text(6.0, 11.42, "Cell state (per node)", ha="center", va="top",
        fontsize=11.5, fontweight="bold", color="white")
ax.text(6.0, 11.10,
        "x = [ T, rho, rho_i0, rho_i1, rho_i2, porosity, pg, mdotf ]",
        ha="center", va="top", fontsize=9.6, family="monospace", color="white")
ax.text(6.0, 10.82,
        "8 features: temperature, densities, 3 char-species, porosity, gas pressure, gas flux",
        ha="center", va="top", fontsize=8.2, color="0.9", style="italic")

# ---- edges note ----
ax.text(6.0, 10.28,
        "edges = 1D path (each cell <-> its left/right neighbor);   edge attribute = cell spacing dx",
        ha="center", va="top", fontsize=8.6, color="0.35", style="italic")

# ---- split into three physical channels ----
ax.text(6, 9.95, "one update = these three pieces, summed", ha="center",
        fontsize=10, color="0.4", fontweight="bold")

# ---------- CONDUCTION (fixed) ----------
cy = 9.5
box(0.4, cy, 3.5, 2.0, FIX, "1.  Conduction",
    "")
ax.text(2.15, cy - 0.72,
        r"$\Phi = D\,(T_{self}-T_{nbr})$", ha="center", fontsize=12, color=FIX)
ax.text(2.15, cy - 1.15, "D = 0.375  (FIXED to the", ha="center", fontsize=8.8,
        color="0.25")
ax.text(2.15, cy - 1.42, "measured diffusion rate,", ha="center", fontsize=8.8,
        color="0.25")
ax.text(2.15, cy - 1.69, "NOT learned)", ha="center", fontsize=8.8, color="0.25",
        fontweight="bold")

# ---------- PYROLYSIS (fixed) ----------
box(4.25, cy, 3.5, 2.0, FIX, "2.  Pyrolysis source",
    "")
ax.text(6.0, cy - 0.72,
        r"$W' = W\,e^{-A\,\Delta t\,e^{-E/T}}$", ha="center", fontsize=11, color=FIX)
ax.text(6.0, cy - 1.15, "exact Arrhenius kinetics", ha="center", fontsize=8.8,
        color="0.25")
ax.text(6.0, cy - 1.42, "from the solver's material", ha="center", fontsize=8.8,
        color="0.25")
ax.text(6.0, cy - 1.69, "file (A, E, psi) - no params", ha="center", fontsize=8.8,
        color="0.25", fontweight="bold")

# ---------- GAS COUPLING (learned) ----------
box(8.1, cy, 3.5, 2.0, LRN, "3.  Gas coupling",
    "")
ax.text(9.85, cy - 0.72, r"$c = \mathrm{MLP}(T, p_g, \dot m)$", ha="center",
        fontsize=10.5, color=LRN)
ax.text(9.85, cy - 1.15, "the ONLY learned part:", ha="center", fontsize=8.8,
        color="0.25", fontweight="bold")
ax.text(9.85, cy - 1.42, "how gas state modulates", ha="center", fontsize=8.8,
        color="0.25")
ax.text(9.85, cy - 1.69, "the decomposition (~3 params)", ha="center",
        fontsize=8.8, color="0.25")

# arrows from state to the three channels
for xt in (2.15, 6.0, 9.85):
    arrow(6, 9.78, xt, cy + 0.02, GRY, 1.5)

# ---------- SUM / EULER STEP ----------
sy = 6.7
arrow(2.15, cy - 2.0, 6, sy + 0.02, GRY, 1.5)
arrow(6.0, cy - 2.0, 6, sy + 0.02, GRY, 1.5)
arrow(9.85, cy - 2.0, 6, sy + 0.02, GRY, 1.5)
box(3.3, sy, 5.4, 0.95, STR,
    r"$\Delta x = -(\Phi_R + \Phi_L) + S_{pyro}$", "net conduction flux + pyrolysis")

# ---------- SUB-STEP LOOP ----------
ly = 5.15
box(2.8, ly, 6.4, 1.1, "0.35",
    r"repeat $\times\,8$ sub-steps  $\Rightarrow$  one data step",
    "bounded sub-steps let heat physically reach across the mesh", dark=True)
arrow(6, sy - 0.95, 6, ly + 0.02, GRY, 1.5)
# loop-back arrow
ax.add_patch(FancyArrowPatch((9.2, ly - 0.55), (10.4, ly - 0.55),
             arrowstyle="-", color="0.5", lw=1.4))
ax.add_patch(FancyArrowPatch((10.4, ly - 0.55), (10.4, sy - 0.47),
             arrowstyle="-", color="0.5", lw=1.4))
ax.add_patch(FancyArrowPatch((10.4, sy - 0.47), (8.7, sy - 0.47),
             arrowstyle="-|>", mutation_scale=13, color="0.5", lw=1.4))
ax.text(10.6, (ly + sy) / 2 - 0.5, "loop", fontsize=8.5, color="0.5", rotation=90,
        va="center")

# ---------- OUTPUT ----------
oy = 3.55
box(3.6, oy, 4.8, 0.9, GRY, "next-step field  x + dx",
    "T bounded to the solver's physical range", dark=True)
arrow(6, ly - 1.1, 6, oy + 0.02, GRY, 1.5)

# ---------- the guarantee box ----------
ax.add_patch(FancyBboxPatch((0.4, -6.0), 5.4, 8.4,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             linewidth=1.8, edgecolor=STR, facecolor=STR + "0c"))
ax.text(0.7, 2.05, "Why it cannot blow up", fontsize=12, fontweight="bold",
        color=STR)
g = r"""MAXIMUM PRINCIPLE
Each face weight D is capped at 1/degree = 0.5,
so the update is a convex blend of a cell and
its neighbors:

  T_new  in  [ min(neighbors) , max(neighbors) ]

The field can NEVER exceed its surroundings
=> unconditionally stable at ANY timestep.

CONSERVATION + ZERO BASELINE
  Phi(self->nbr) = -Phi(nbr->self)   (antisymm.)
  identical cells -> Phi = 0  exactly

A uniform field produces no flux, so there is
no drift on quiescent cells.

------------------------------------------------
This is the piece the purely-learned models
lacked. Without it:
  * direct model   -> 141,000 K
  * free flux-form -> checkerboard, +/-10,000 K"""
ax.text(0.7, 1.65, g, fontsize=8.4, family="monospace", va="top", color="0.15",
        linespacing=1.35)

# ---------- what's fixed vs learned box ----------
ax.add_patch(FancyBboxPatch((6.2, -6.0), 5.4, 8.4,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             linewidth=1.8, edgecolor="0.5", facecolor="0.965"))
ax.text(6.5, 2.05, "Fixed physics vs. learned", fontsize=12, fontweight="bold",
        color="0.2")
ax.add_patch(plt.Rectangle((6.5, 1.45), 0.28, 0.2, color=FIX))
ax.text(6.9, 1.55, "FIXED to known physics", va="center", fontsize=9.2,
        color="0.2", fontweight="bold")
fx = r"""  - conduction rate D (measured)
  - Arrhenius pyrolysis (A, E, psi
    from the material file)
  - the max-principle / flux structure"""
ax.text(6.55, 1.28, fx, fontsize=8.6, family="monospace", va="top", color="0.15",
        linespacing=1.4)

ax.add_patch(plt.Rectangle((6.5, -0.15), 0.28, 0.2, color=LRN))
ax.text(6.9, -0.05, "LEARNED by the GNN", va="center", fontsize=9.2, color="0.2",
        fontweight="bold")
ln = r"""  - the gas coupling MLP:
    how (T, p_g, mdot) modulate the
    total decomposition rate
  - ~3 parameters, small residual"""
ax.text(6.55, -0.32, ln, fontsize=8.6, family="monospace", va="top", color="0.15",
        linespacing=1.4)

ax.text(6.5, -2.4, "RESULT (held-out aw2):", fontsize=9.6, fontweight="bold",
        color="0.2")
res = r"""  ~25 K mean  (8.7 K by t = 120 s)
  char front tracks solver to ~1.4 mm
  field bounded to solver's exact range
  and stays NONLINEAR (real physics)"""
ax.text(6.55, -2.65, res, fontsize=8.8, family="monospace", va="top",
        color="0.15", linespacing=1.5)

# ---------- legend (bottom, in the empty band under the boxes) ----------
lx = 2.6
for c, lab in [(FIX, "fixed physics"), (LRN, "learned by GNN"),
               (STR, "structural guarantee")]:
    ax.add_patch(plt.Rectangle((lx, -6.35), 0.24, 0.18, color=c))
    ax.text(lx + 0.34, -6.26, lab, va="center", fontsize=8.6, color="0.25")
    lx += 2.55

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "heat_python/figs/architecture/model_conductance_arch.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
