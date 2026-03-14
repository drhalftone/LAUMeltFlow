"""Run tapered bead chain simulation with live visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulation import create_chain, step_symplectic_euler

# --- Simulation parameters ---
N_NODES = 15
TOTAL_LENGTH = 2.0
TOTAL_MASS = 0.5
GRAVITY = 9.81
STIFFNESS = 1e4
DAMPING = 0.5
DRAG = 0.02
DT = 0.0001
STEPS_PER_FRAME = 50      # simulation steps between each screen update
TAPER_RATIO = 10.0
USE_CONSTRAINTS = True
CONSTRAINT_ITERS = 10

# --- Build chain ---
state = create_chain(
    n_nodes=N_NODES,
    total_length=TOTAL_LENGTH,
    total_mass=TOTAL_MASS,
    stiffness=STIFFNESS,
    damping=DAMPING,
    drag=DRAG,
    taper_ratio=TAPER_RATIO,
)
anchor = state.pos[0].copy()

# Node sizes proportional to mass
node_sizes = 3 + 12 * (state.mass / state.mass.max())

# --- Set up figure ---
fig, ax_sim = plt.subplots(1, 1, figsize=(8, 8))

# Simulation axes — anchor centered
ax_sim.set_xlim(-TOTAL_LENGTH * 1.2, TOTAL_LENGTH * 1.2)
ax_sim.set_ylim(-TOTAL_LENGTH * 1.5, TOTAL_LENGTH * 0.5)
ax_sim.set_aspect("equal")
ax_sim.set_xlabel("x (m)")
ax_sim.set_ylabel("y (m)")
ax_sim.set_title("Bead Chain (tapered mass, gravity)")
ax_sim.grid(True, alpha=0.3)

# Draw rod lines
lines = []
for i in range(N_NODES - 1):
    (line,) = ax_sim.plot([], [], "b-", linewidth=1.5, alpha=0.7)
    lines.append(line)

# Draw nodes as scatter (sized by mass)
scatter = ax_sim.scatter(state.pos[:, 0], state.pos[:, 1],
                         s=node_sizes ** 2, c="black", zorder=5)

# Anchor marker
(anchor_plot,) = ax_sim.plot([], [], "rs", markersize=10, zorder=6)

# Tip trail
trail_x, trail_y = [], []
(trail_plot,) = ax_sim.plot([], [], "r-", linewidth=0.5, alpha=0.4)

time_text = ax_sim.text(0.02, 0.98, "", transform=ax_sim.transAxes,
                        verticalalignment="top", fontsize=10,
                        family="monospace")

sim_time = 0.0
frame_count = 0


def update(frame):
    global sim_time, frame_count

    # Advance simulation by STEPS_PER_FRAME substeps
    for _ in range(STEPS_PER_FRAME):
        step_symplectic_euler(state, DT, GRAVITY,
                              use_constraints=USE_CONSTRAINTS,
                              constraint_iters=CONSTRAINT_ITERS)
        state.pos[0] = anchor
        state.vel[0] = 0.0
        sim_time += DT

    frame_count += 1
    pos = state.pos

    # Update rods
    for i in range(N_NODES - 1):
        lines[i].set_data([pos[i, 0], pos[i + 1, 0]],
                          [pos[i, 1], pos[i + 1, 1]])

    # Update nodes
    scatter.set_offsets(pos)

    # Anchor
    anchor_plot.set_data([pos[0, 0]], [pos[0, 1]])

    # Tip trail
    trail_x.append(pos[-1, 0])
    trail_y.append(pos[-1, 1])
    trail_plot.set_data(trail_x, trail_y)

    # Info text
    tip_speed = np.linalg.norm(state.vel[-1])
    time_text.set_text(f"t = {sim_time:.3f} s\ntip: {tip_speed:.1f} m/s")

    return lines + [scatter, anchor_plot, trail_plot, time_text]


anim = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
