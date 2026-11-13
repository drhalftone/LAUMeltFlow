"""Study diagram: forward step of the CONSERVATIVE FLUX-FORM model (aw2).
Parallel to viz_6steps.py (direct model). The signature move is the antisymmetric
face flux, which gives conservation + a zero baseline by construction. Worked
example uses REAL trained weights from heat_flux_aw2.pt.
-> figs/architecture/model_6steps_flux.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FCN, FLUX, PHYS = "#3b6fb0", "#6d4fa8", "#3b8f5a"
steps = [
    (1, "Encode each cell", r"$h = \mathrm{MLP}(x)$  [self, left, right]",
     "same weights on all three cells", FCN),
    (2, "Encode face geometry", r"$g = \mathrm{MLP}(\Delta x)$",
     "the cell spacing at each face", FCN),
    (3, "Directed half-flux", r"$\psi(a,b) = \mathrm{MLP}([\,h_a,\ h_b,\ g\,])$",
     "a one-way flux estimate", FCN),
    (4, "Antisymmetric face flux", r"$\Phi(\mathrm{self},\mathrm{nbr}) = \psi(\mathrm{self},\mathrm{nbr}) - \psi(\mathrm{nbr},\mathrm{self})$",
     "=> conservation + zero baseline, BY CONSTRUCTION", FLUX),
    (5, "Local source", r"$S = \mathrm{MLP}(h_{\mathrm{self}})$",
     "pyrolysis: local, not gradient-driven", FCN),
    (6, "Divergence + step", r"$\Delta = -(\Phi_R + \Phi_L) + S,\quad x^{new} = x + \Delta$",
     "net flux in + source, then one Euler step", PHYS),
]

fig, ax = plt.subplots(figsize=(9.0, 14.2))
ax.set_xlim(0, 10); ax.set_ylim(-5.8, 12.2); ax.axis("off")
ax.text(5, 11.7, "Conservative flux-form step (aw2 model)", ha="center",
        fontsize=16.5, fontweight="bold")
ax.text(5, 11.28, "predicts face fluxes + a local source, not a direct state delta",
        ha="center", fontsize=10.5, color="0.35")

h, gap = 1.32, 0.2
for k, (n, title, eq, plain, col) in enumerate(steps):
    y = 10.55 - k * (h + gap)
    ax.add_patch(FancyBboxPatch((1.0, y - h), 8.0, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=2, edgecolor=col, facecolor=col + "18"))
    ax.add_patch(plt.Circle((1.55, y - h / 2), 0.3, color=col, zorder=3))
    ax.text(1.55, y - h / 2, str(n), ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", zorder=4)
    ax.text(2.3, y - 0.32, title, fontsize=12.5, fontweight="bold", va="top")
    ax.text(2.3, y - 0.70, eq, fontsize=11.5, va="top", color=col)
    ax.text(2.3, y - 1.06, plain, fontsize=9.3, va="top", color="0.3", style="italic")
    if k < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((5, y - h), (5, y - h - gap),
                     arrowstyle="-|>", mutation_scale=15, color="0.55", lw=1.6))

# ---- worked example: real trained flux weights ----
ax.add_patch(FancyBboxPatch((1.0, -5.0), 8.0, 6.2,
             boxstyle="round,pad=0.02,rounding_size=0.1",
             linewidth=1.8, edgecolor="0.45", facecolor="0.96"))
ax.text(1.4, 0.95, "Worked example — real trained flux weights (aw2)",
        fontsize=12.5, fontweight="bold")
ax.text(1.4, 0.58, "the two guarantees, shown on two real adjacent cells",
        fontsize=9, color="0.35", style="italic")
body = r"""two real adjacent cells:  hot T=1222K ,  neighbor T=1132K
out channels: [ T , rho_i0 , rho_i1 ]

face flux leaving the hot cell:
  Phi(hot -> nbr)  = [  1.94 , -57.95 , -82.47 ]

CONSERVATION (by construction):
  Phi(nbr -> hot)  = [ -1.94 ,  57.95 ,  82.47 ]  = -Phi
  sum = [ 0 , 0 , 0 ]   what leaves hot enters nbr, exactly

ZERO BASELINE (by construction):
  two identical cells (self == nbr):
  Phi = psi(x,x) - psi(x,x) = [ 0 , 0 , 0 ]  exactly
  a uniform field makes no flux -> no quiescent-cell bias

  (this is why the flux form kills the direct model's
   catastrophic divergence -- the bias channel cannot exist)"""
ax.text(1.4, 0.05, body, fontsize=9.0, family="monospace", va="top",
        ha="left", color="0.15", linespacing=1.4)

lx = 0.9
for c, lab in [(FCN, "fully-connected net (FCN)"),
               (FLUX, "antisymmetric flux (conservation)"),
               (PHYS, "physics update")]:
    ax.add_patch(plt.Rectangle((lx, -5.5), 0.3, 0.22, color=c))
    ax.text(lx + 0.42, -5.39, lab, va="center", fontsize=9.0, color="0.25")
    lx += 3.05
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "heat_python/figs/architecture/model_6steps_flux.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
