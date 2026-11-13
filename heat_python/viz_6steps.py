"""One-off study diagram: the 6 steps of one GNN forward step, as a clean flow.
Not part of the pipeline; a teaching figure. -> figs/architecture/model_6steps.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FCN, GRAPH, PHYS = "#3b6fb0", "#b06a3b", "#3b8f5a"   # step-type colors
steps = [
    (1, "Encode the cell", r"$h_i = \mathrm{MLP}(x_i)$",
     "state [T, ρ, ρᵢ, porosity] → hidden(64)", FCN),
    (2, "Encode each edge", r"$m_L,\ m_R = \mathrm{MLP}(x_{nbr}-x_i,\ \Delta x)$",
     "neighbor-minus-self + spacing → message", FCN),
    (3, "Concatenate neighbors", r"$z_i = [\,h_i,\ m_L,\ m_R\,]$",
     "the cell plus what it heard from both sides", GRAPH),
    (4, "Update", r"$h_i' = \mathrm{MLP}(z_i)$",
     "mix cell + neighbors → new hidden", FCN),
    (5, "Read out the change", r"$\Delta_i = W\,\mathrm{LayerNorm}(h_i')$",
     "→ [ΔT, Δρᵢ₀, Δρᵢ₁]", FCN),
    (6, "Step forward in time", r"$x_i^{new} = x_i + \Delta_i$",
     "add increment (one Euler step); repeat every cell, every step", PHYS),
]

fig, ax = plt.subplots(figsize=(8.8, 14.0))
ax.set_xlim(0, 10); ax.set_ylim(-5.8, 12.2); ax.axis("off")
ax.text(5, 11.7, "GNN forward step: 6 steps", ha="center",
        fontsize=17, fontweight="bold")
ax.text(5, 11.25, "one cell $i$, one time step (same weights for every cell)",
        ha="center", fontsize=10.5, color="0.35")

h, gap = 1.32, 0.2
for k, (n, title, eq, plain, col) in enumerate(steps):
    y = 10.55 - k * (h + gap)
    box = FancyBboxPatch((1.0, y - h), 8.0, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=2, edgecolor=col, facecolor=col + "18")
    ax.add_patch(box)
    ax.add_patch(plt.Circle((1.55, y - h / 2), 0.3, color=col, zorder=3))
    ax.text(1.55, y - h / 2, str(n), ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", zorder=4)
    ax.text(2.3, y - 0.32, title, fontsize=12.5, fontweight="bold", va="top")
    ax.text(2.3, y - 0.70, eq, fontsize=12, va="top", color=col)
    ax.text(2.3, y - 1.06, plain, fontsize=9.3, va="top", color="0.3", style="italic")
    if k < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((5, y - h), (5, y - h - gap),
                     arrowstyle="-|>", mutation_scale=15, color="0.55", lw=1.6))

# ---- worked mini-example panel (toy sizes so the numbers are legible) ----
wbox = FancyBboxPatch((1.0, -5.0), 8.0, 6.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                      linewidth=1.8, edgecolor="0.45", facecolor="0.96")
ax.add_patch(wbox)
ax.text(1.4, 0.95, "Worked example — real trained weights (aw1 model)",
        fontsize=12.5, fontweight="bold")
ax.text(1.4, 0.58, "cell-encoder first layer (Linear 6->64), 3 of 64 units",
        fontsize=9, color="0.35", style="italic")

body = r"""one real cell (raw physical):
  [ T , rho , rhoi0 , rhoi1 , rhoi2 , por ]
    = [ 1008 , 278 , 280 , 900 , 1600 , 0.80 ]

normalize:  x = (raw - mean)/std
  x = [ 1.77 , 0.30 , 0.22 , 0.37 , 0.00 , -0.30 ]

first 3 rows of the 64x6 weight matrix W (+ bias b):
  W0 = [ 0.13 -0.33 -0.19 -0.22 -0.15  0.21 ]  b0 = -0.18
  W1 = [ 0.25  0.14 -0.12  0.34 -0.15 -0.07 ]  b1 = -0.03
  W2 = [ 0.10  0.28 -0.23  0.40 -0.21 -0.24 ]  b2 =  0.09

pre-activation  z = W . x + b :
  z0 = 0.13(1.77) -0.33(0.30) -0.19(0.22) -0.22(0.37)
       +0.21(-0.30) -0.18                    = -0.23
  z1 = 0.57      z2 = 0.52     (same pattern)

GELU:  h = [ -0.10 , 0.40 , 0.36 ]   (first 3 of 64)

then LayerNorm + a 2-layer MLP finish the encode.
these are the model's actual learned weights."""
ax.text(1.4, 0.05, body, fontsize=9.0, family="monospace", va="top",
        ha="left", color="0.15", linespacing=1.4)

# legend (bottom)
lx = 0.9
for c, lab in [(FCN, "fully-connected net (FCN)"),
               (GRAPH, "graph step (mix neighbors)"),
               (PHYS, "physics update")]:
    ax.add_patch(plt.Rectangle((lx, -5.5), 0.3, 0.22, color=c))
    ax.text(lx + 0.42, -5.39, lab, va="center", fontsize=9.3, color="0.25")
    lx += 3.0
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "heat_python/figs/architecture/model_6steps.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
