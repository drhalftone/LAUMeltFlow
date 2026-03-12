"""
FEA-style bead chain simulation.

Models a chain of N mass nodes connected by stiff rod elements.
Uses symplectic Euler (semi-implicit Euler) integration which is
simple, stable, and energy-conserving for conservative systems.

The mesh is inherently a graph: nodes = FE nodes, edges = elements,
making this a natural fit for later GNN surrogate modeling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChainState:
    """State of the bead chain."""
    pos: np.ndarray          # (n_nodes, 2) positions
    vel: np.ndarray          # (n_nodes, 2) velocities
    mass: np.ndarray         # (n_nodes,) mass per node
    fixed: np.ndarray        # (n_nodes,) bool, True = fixed node
    edges: np.ndarray        # (n_edges, 2) element connectivity
    rest_lengths: np.ndarray # (n_edges,) rest length per element
    stiffness: float         # spring stiffness (N/m)
    damping: float           # damping coefficient


def create_chain(
    n_nodes: int = 20,
    total_length: float = 2.0,
    total_mass: float = 1.0,
    stiffness: float = 1e4,
    damping: float = 0.0,
    anchor_pos: np.ndarray = None,
) -> ChainState:
    """
    Create a horizontal chain fixed at one end (the ceiling).

    Node 0 is the anchor (fixed). Nodes 1..N-1 extend horizontally
    to the right from the anchor point.
    """
    if anchor_pos is None:
        anchor_pos = np.array([0.0, 0.0])

    L0 = total_length / (n_nodes - 1)
    mass_per_node = total_mass / n_nodes

    pos = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        pos[i] = anchor_pos + np.array([i * L0, 0.0])

    vel = np.zeros((n_nodes, 2))
    mass = np.full(n_nodes, mass_per_node)
    fixed = np.zeros(n_nodes, dtype=bool)
    fixed[0] = True

    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)])
    rest_lengths = np.full(n_nodes - 1, L0)

    return ChainState(
        pos=pos, vel=vel, mass=mass, fixed=fixed,
        edges=edges, rest_lengths=rest_lengths,
        stiffness=stiffness, damping=damping,
    )


def compute_forces(state: ChainState, gravity: float = 9.81) -> np.ndarray:
    """
    Compute all forces: gravity + elastic rod forces + damping.

    Each rod element acts as a spring:
      F = -k * (|delta| - L0) * (delta / |delta|)
    plus optional viscous damping along the rod axis.
    """
    n = len(state.mass)
    forces = np.zeros((n, 2))

    # Gravity
    for i in range(n):
        if not state.fixed[i]:
            forces[i, 1] = -state.mass[i] * gravity

    # Spring forces from rod elements
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        delta = state.pos[j] - state.pos[i]
        dist = np.linalg.norm(delta)
        if dist < 1e-12:
            continue

        direction = delta / dist
        stretch = dist - state.rest_lengths[e]

        # Elastic force: Hooke's law
        f_spring = state.stiffness * stretch * direction

        # Damping force along rod axis
        if state.damping > 0:
            vel_rel = state.vel[j] - state.vel[i]
            v_along = np.dot(vel_rel, direction)
            f_damp = state.damping * v_along * direction
        else:
            f_damp = 0.0

        forces[i] += f_spring + f_damp
        forces[j] -= f_spring + f_damp

    return forces


def step_symplectic_euler(
    state: ChainState,
    dt: float,
    gravity: float = 9.81,
) -> np.ndarray:
    """
    One symplectic Euler timestep.

    Symplectic Euler (semi-implicit):
      v_{n+1} = v_n + dt * a(x_n)
      x_{n+1} = x_n + dt * v_{n+1}

    This is first-order but symplectic — energy oscillates but
    doesn't drift. Good for stiff spring systems.

    Returns acceleration (n_nodes, 2).
    """
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]

    # Zero out fixed node acceleration
    acc[state.fixed] = 0.0

    # Update velocity first (symplectic)
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0

    # Then update position with new velocity
    state.pos += dt * state.vel
    # Re-pin fixed nodes (shouldn't move, but be safe)
    # (handled by vel=0 and acc=0)

    return acc


def run_simulation(
    n_nodes: int = 20,
    total_length: float = 2.0,
    total_mass: float = 1.0,
    gravity: float = 9.81,
    stiffness: float = 1e4,
    damping: float = 0.5,
    dt: float = 0.0001,
    n_steps: int = 50000,
    save_interval: int = 50,
    anchor_pos: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the full simulation and collect trajectory data.

    Returns a dict with arrays shaped for GNN training:
      positions:      (n_saved, n_nodes, 2)
      velocities:     (n_saved, n_nodes, 2)
      accelerations:  (n_saved, n_nodes, 2)
      edge_index:     (2, 2*n_edges) bidirectional
      rest_lengths:   (n_edges,)
      node_types:     (n_nodes,) 0=free, 1=fixed
      times:          (n_saved,)
    """
    state = create_chain(n_nodes, total_length, total_mass,
                         stiffness, damping, anchor_pos)
    anchor = state.pos[0].copy()

    # Preallocate storage
    n_saved = n_steps // save_interval + 1
    positions = np.zeros((n_saved, n_nodes, 2))
    velocities = np.zeros((n_saved, n_nodes, 2))
    accelerations = np.zeros((n_saved, n_nodes, 2))
    times = np.zeros(n_saved)

    # Save initial state
    positions[0] = state.pos.copy()
    velocities[0] = state.vel.copy()
    accelerations[0] = 0.0
    times[0] = 0.0
    save_idx = 1

    # Time integration loop
    for step in range(1, n_steps + 1):
        acc = step_symplectic_euler(state, dt, gravity)

        # Re-pin anchor (belt and suspenders)
        state.pos[0] = anchor
        state.vel[0] = 0.0

        if step % save_interval == 0 and save_idx < n_saved:
            positions[save_idx] = state.pos.copy()
            velocities[save_idx] = state.vel.copy()
            accelerations[save_idx] = acc.copy()
            times[save_idx] = step * dt
            save_idx += 1

    # Trim
    positions = positions[:save_idx]
    velocities = velocities[:save_idx]
    accelerations = accelerations[:save_idx]
    times = times[:save_idx]

    # Bidirectional edge index for GNN
    fwd = state.edges
    bwd = state.edges[:, ::-1]
    edge_index = np.concatenate([fwd, bwd], axis=0).T  # (2, 2*n_edges)

    # Node types
    node_types = np.zeros(n_nodes, dtype=np.int32)
    node_types[0] = 1  # fixed

    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "times": times,
        "edge_index": edge_index,
        "rest_lengths": state.rest_lengths,
        "node_types": node_types,
        "dt": dt * save_interval,
        "gravity": gravity,
        "n_nodes": n_nodes,
        "total_mass": total_mass,
        "total_length": total_length,
        "stiffness": stiffness,
        "damping": damping,
    }
