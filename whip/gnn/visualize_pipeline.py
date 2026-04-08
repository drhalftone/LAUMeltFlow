"""Visualize the generate_volume_data pipeline."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("generate_volume_data.py — Pipeline Visualization", fontsize=16, fontweight="bold", y=0.98)

# ─── Panel 1: The bead chain (the physical system) ───
ax = axes[0, 0]
ax.set_title("1. The Physical System", fontsize=12, fontweight="bold")
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect("equal")
ax.axis("off")

n_beads = 16
x_pos = np.linspace(0, 2, n_beads)
masses = np.linspace(1.0, 1.0/10, n_beads)
masses = masses / masses.sum() * 0.5
sizes = 80 + 400 * (masses / masses.max())

# Draw rods
for i in range(n_beads - 1):
    ax.plot([x_pos[i], x_pos[i+1]], [0, 0], 'b-', linewidth=1.5, alpha=0.5)

# Draw beads
for i in range(n_beads):
    color = 'red' if i == 0 else plt.cm.viridis(masses[i] / masses.max())
    ax.scatter(x_pos[i], 0, s=sizes[i], c=[color], zorder=5, edgecolors='black', linewidth=0.5)

ax.annotate("Fixed\nanchor", (x_pos[0], 0), (x_pos[0], 0.3),
            ha='center', fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.annotate("Light tip", (x_pos[-1], 0), (x_pos[-1], 0.3),
            ha='center', fontsize=8, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.text(1.0, -0.35, "16 beads, tapered mass (heavy → light)\nconnected by spring-rod elements",
        ha='center', fontsize=9, style='italic')

# ─── Panel 2: Random sampling (one bead's perspective) ───
ax = axes[0, 1]
ax.set_title("2. Sample Random Scenarios", fontsize=12, fontweight="bold")
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
ax.set_aspect("equal")

# The bead at center
ax.scatter(0, 0, s=200, c='steelblue', zorder=5, edgecolors='black', linewidth=1.5)
ax.text(0, -0.02, "self", ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Velocity arrow
rng = np.random.default_rng(42)
vx, vy = 0.08, -0.05
ax.annotate("", xy=(vx, vy), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax.text(vx + 0.02, vy, "vel", fontsize=8, color='orange', fontweight='bold')

# Left neighbor — polar sampling ring
rest = 0.1333
theta_ring = np.linspace(0, 2 * np.pi, 100)
r_inner = rest * 0.95
r_outer = rest * 1.05
ax.plot(r_inner * np.cos(theta_ring) - rest, r_inner * np.sin(theta_ring), 'g--', alpha=0.4, linewidth=0.8)
ax.plot(r_outer * np.cos(theta_ring) - rest, r_outer * np.sin(theta_ring), 'g--', alpha=0.4, linewidth=0.8)
ax.fill_between(r_outer * np.cos(theta_ring) - rest,
                r_outer * np.sin(theta_ring),
                r_inner * np.sin(theta_ring) + (r_outer - r_inner) * np.sin(theta_ring) / np.abs(np.sin(theta_ring) + 1e-9) * 0,
                alpha=0.08, color='green')

# Show a few sampled left neighbor positions
for _ in range(8):
    r = rng.uniform(r_inner, r_outer)
    theta = rng.uniform(0, 2 * np.pi)
    lx = -rest + r * np.cos(theta)
    ly = r * np.sin(theta)
    ax.scatter(lx, ly, s=50, c='green', alpha=0.5, zorder=4)

ax.scatter(-rest, 0, s=120, c='green', zorder=5, edgecolors='black', linewidth=1)
ax.text(-rest, 0.035, "left\nneighbor", ha='center', fontsize=7, color='green')

# Right neighbor — polar sampling ring
ax.plot(r_inner * np.cos(theta_ring) + rest, r_inner * np.sin(theta_ring), 'm--', alpha=0.4, linewidth=0.8)
ax.plot(r_outer * np.cos(theta_ring) + rest, r_outer * np.sin(theta_ring), 'm--', alpha=0.4, linewidth=0.8)

for _ in range(8):
    r = rng.uniform(r_inner, r_outer)
    theta = rng.uniform(0, 2 * np.pi)
    rx = rest + r * np.cos(theta)
    ry = r * np.sin(theta)
    ax.scatter(rx, ry, s=50, c='purple', alpha=0.5, zorder=4)

ax.scatter(rest, 0, s=120, c='purple', zorder=5, edgecolors='black', linewidth=1)
ax.text(rest, 0.035, "right\nneighbor", ha='center', fontsize=7, color='purple')

ax.text(0, -0.22, "Polar sampling: r ≈ rest length, θ uniform\n1M independent random scenarios",
        ha='center', fontsize=8, style='italic')
ax.axis("off")

# ─── Panel 3: Force computation ───
ax = axes[0, 2]
ax.set_title("3. Compute Forces (per sample)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.7, 0.5)
ax.axis("off")

# Central bead
ax.scatter(0, 0, s=300, c='steelblue', zorder=5, edgecolors='black', linewidth=1.5)

# Force arrows
forces = [
    ((-0.15, 0.15), "Spring L", "green"),
    ((0.12, 0.1), "Spring R", "purple"),
    ((0.0, -0.2), "Gravity", "brown"),
    ((0.1, -0.05), "Drag", "gray"),
    ((-0.08, 0.05), "Damp L", "green"),
    ((0.05, 0.03), "Damp R", "purple"),
]

for (dx, dy), label, color in forces:
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5, alpha=0.7))
    offset = 0.03 if dy > 0 else -0.04
    ax.text(dx + 0.02, dy + offset, label, fontsize=7, color=color, fontweight='bold')

# Equations
eq_y = -0.40
eqs = [
    "F_spring = k·(|Δr| - L₀)·dir",
    "F_damp  = c·v_along·dir",
    "F_grav  = -m·g·ŷ",
    "F_drag  = -b·v",
]
for i, eq in enumerate(eqs):
    ax.text(0, eq_y - i * 0.07, eq, ha='center', fontsize=8, family='monospace')

# ─── Panel 4: Symplectic Euler integration ───
ax = axes[1, 0]
ax.set_title("4. Symplectic Euler Step", fontsize=12, fontweight="bold")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.6, 0.5)
ax.axis("off")

steps = [
    (0.35, "acc = F_total / mass", "#2196F3"),
    (0.20, "vel_new = vel + dt · acc", "#4CAF50"),
    (0.05, "Δpos = dt · vel_new", "#FF9800"),
    (-0.10, "Δvel = vel_new − vel", "#FF9800"),
    (-0.30, "if fixed: Δpos = Δvel = 0", "#f44336"),
]

for y, text, color in steps:
    bbox = FancyBboxPatch((-0.42, y - 0.05), 0.84, 0.10,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(0, y, text, ha='center', va='center', fontsize=9, family='monospace',
            fontweight='bold', color=color)

# Arrows between steps
for i in range(len(steps) - 1):
    ax.annotate("", xy=(0, steps[i+1][0] + 0.05), xytext=(0, steps[i][0] - 0.05),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.text(0, -0.50, "dt = 0.0001s — one tiny step",
        ha='center', fontsize=9, style='italic', color='gray')

# ─── Panel 5: Input/Output format ───
ax = axes[1, 1]
ax.set_title("5. Pack Input (X) and Target (Y)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.7, 0.5)
ax.axis("off")

# Input X
ax.text(0, 0.42, "Input X — 16 features per sample", ha='center', fontsize=9, fontweight='bold')

sections = [
    (0.28, "Self (4)", "steelblue", "vel_x, vel_y, mass, is_fixed"),
    (0.12, "Left neighbor (6)", "green", "dpos_x, dpos_y, dvel_x, dvel_y, mass, L₀"),
    (-0.04, "Right neighbor (6)", "purple", "dpos_x, dpos_y, dvel_x, dvel_y, mass, L₀"),
]

for y, title, color, fields in sections:
    bbox = FancyBboxPatch((-0.45, y - 0.05), 0.90, 0.12,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(-0.40, y + 0.01, title, fontsize=8, fontweight='bold', color=color, va='center')
    ax.text(0.42, y + 0.01, fields, fontsize=7, ha='right', va='center', family='monospace')

# Target Y
ax.text(0, -0.22, "Target Y — 4 values per sample", ha='center', fontsize=9, fontweight='bold')
bbox = FancyBboxPatch((-0.45, -0.38), 0.90, 0.12,
                      boxstyle="round,pad=0.02",
                      facecolor='#FF9800', alpha=0.15, edgecolor='#FF9800', linewidth=1.5)
ax.add_patch(bbox)
ax.text(0, -0.32, "Δpos_x,  Δpos_y,  Δvel_x,  Δvel_y",
        ha='center', fontsize=9, family='monospace', fontweight='bold', color='#E65100')

ax.text(0, -0.55, "Shape: X=(1M, 16)  Y=(1M, 4)",
        ha='center', fontsize=9, style='italic', color='gray')

# ─── Panel 6: Volume vs trajectory comparison ───
ax = axes[1, 2]
ax.set_title("6. Why Volume Sampling?", fontsize=12, fontweight="bold")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")

# Trajectory data — thin curves
t = np.linspace(0, 10, 500)
for i in range(5):
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.5, 2.0)
    tx = amp * np.sin(t + phase) + rng.normal(0, 0.1, len(t)).cumsum() * 0.02
    ty = amp * np.cos(t * 0.7 + phase) + rng.normal(0, 0.1, len(t)).cumsum() * 0.02
    tx = np.clip(tx, -2.8, 2.8)
    ty = np.clip(ty, -2.8, 2.8)
    ax.plot(tx, ty, 'b-', alpha=0.3, linewidth=1)

ax.text(-1.5, 2.5, "Trajectory data\n(thin paths)", fontsize=9, color='blue',
        fontweight='bold', style='italic')

# Volume data — scattered points
vx = rng.uniform(-2.5, 2.5, 300)
vy = rng.uniform(-2.5, 2.5, 300)
ax.scatter(vx, vy, s=8, c='red', alpha=0.3, zorder=3)
ax.text(1.0, -2.5, "Volume data\n(fills space)", fontsize=9, color='red',
        fontweight='bold', style='italic')

ax.set_xlabel("State dimension 1", fontsize=8)
ax.set_ylabel("State dimension 2", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("pipeline_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to pipeline_visualization.png")
