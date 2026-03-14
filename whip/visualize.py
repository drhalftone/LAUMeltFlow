"""
Visualization for bead chain simulation.

Provides animation of the chain dynamics and energy diagnostics
to validate the symplectic integrator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_chain(
    trajectory: dict,
    interval_ms: int = 20,
    save_path: str = None,
    trail: bool = True,
):
    """
    Animate the chain falling under gravity.

    Args:
        trajectory: dict from run_simulation()
        interval_ms: milliseconds between frames
        save_path: if set, save animation to this path (.mp4 or .gif)
        trail: if True, show faded trail of tip node
    """
    positions = trajectory["positions"]  # (n_frames, n_nodes, 2)
    edge_index = trajectory["edge_index"]  # (2, n_edges_bidi)
    n_frames = len(positions)
    n_nodes = positions.shape[1]

    # Use only forward edges for drawing (first half of bidirectional)
    n_edges_uni = edge_index.shape[1] // 2
    edges_fwd = edge_index[:, :n_edges_uni].T  # (n_edges, 2)

    # Compute axis limits from full trajectory
    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]
    margin = 0.3
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    # Make aspect ratio equal
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    taper = trajectory.get("taper_ratio", 1.0)
    title = "Whip Crack" if taper > 1.0 else "Bead Chain"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Draw elements
    lines = []
    for e in range(len(edges_fwd)):
        (line,) = ax.plot([], [], "b-", linewidth=1.5, alpha=0.7)
        lines.append(line)

    # Draw nodes — size proportional to mass if available
    node_mass = trajectory.get("node_mass", None)
    if node_mass is not None:
        node_sizes = 3 + 12 * (node_mass / node_mass.max())
    else:
        node_sizes = np.full(n_nodes, 5.0)
    (nodes_plot,) = ax.plot([], [], "ko", markersize=5, zorder=5)

    # Fixed node marker
    (anchor_plot,) = ax.plot([], [], "rs", markersize=10, zorder=6)

    # Tip trail
    if trail:
        trail_x, trail_y = [], []
        (trail_plot,) = ax.plot([], [], "r-", linewidth=0.5, alpha=0.4)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        verticalalignment="top", fontsize=10)

    def init():
        for line in lines:
            line.set_data([], [])
        nodes_plot.set_data([], [])
        anchor_plot.set_data([], [])
        time_text.set_text("")
        if trail:
            trail_plot.set_data([], [])
            return lines + [nodes_plot, anchor_plot, trail_plot, time_text]
        return lines + [nodes_plot, anchor_plot, time_text]

    def update(frame):
        pos = positions[frame]

        # Update edges
        for e, (i, j) in enumerate(edges_fwd):
            lines[e].set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])

        # Update nodes
        nodes_plot.set_data(pos[:, 0], pos[:, 1])

        # Anchor
        anchor_plot.set_data([pos[0, 0]], [pos[0, 1]])

        # Time and tip speed
        vels = trajectory["velocities"]
        tip_speed = np.linalg.norm(vels[frame, -1])
        time_text.set_text(f"t = {trajectory['times'][frame]:.3f} s  |  tip: {tip_speed:.1f} m/s")

        # Trail
        if trail:
            trail_x.append(pos[-1, 0])
            trail_y.append(pos[-1, 1])
            trail_plot.set_data(trail_x, trail_y)
            return lines + [nodes_plot, anchor_plot, trail_plot, time_text]

        return lines + [nodes_plot, anchor_plot, time_text]

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=interval_ms, blit=True,
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval_ms)
        print("Done.")
    else:
        plt.show()

    plt.close(fig)
    return anim


def plot_energy(trajectory: dict):
    """
    Plot kinetic, potential (gravitational + elastic), and total energy.

    Uses the precomputed energy array from run_simulation().
    Total energy should be approximately conserved (symplectic integrator).
    Small oscillations are expected; drift indicates a problem.
    """
    times = trajectory["times"]
    energy = trajectory["energy"]  # (n_frames, 4): KE, GPE, EPE, Total

    KE = energy[:, 0]
    GPE = energy[:, 1]
    EPE = energy[:, 2]
    TE = energy[:, 3]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Energy components
    ax1.plot(times, KE, label="Kinetic", color="red", alpha=0.8)
    ax1.plot(times, GPE, label="Gravitational PE", color="blue", alpha=0.8)
    ax1.plot(times, EPE, label="Elastic PE", color="green", alpha=0.8)
    ax1.plot(times, TE, label="Total", color="black", linewidth=2)
    ax1.set_ylabel("Energy (J)")
    ax1.set_title("Energy Components")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy drift (relative to initial)
    drift = TE - TE[0]
    ax2.plot(times, drift, "k-", linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy drift (J)")
    ax2.set_title(f"Total Energy Drift (final: {drift[-1]:.4e} J)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
