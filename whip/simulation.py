"""
FEA-style bead chain / whip simulation.

Models a chain of N mass nodes connected by rod elements.
Supports both spring-based and constraint-based (inextensible) rods,
tapered mass distribution, and prescribed anchor motion for whip cracks.

Uses symplectic Euler integration with optional SHAKE-like constraint
projection to enforce rod inextensibility.

The mesh is inherently a graph: nodes = FE nodes, edges = elements,
making this a natural fit for later GNN surrogate modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


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
    damping: float           # damping coefficient (rod-axis)
    drag: float = 0.0        # global velocity drag coefficient
    taper_ratio: float = 1.0 # mass ratio: base / tip (1.0 = uniform)


def create_chain(
    n_nodes: int = 20,
    total_length: float = 2.0,
    total_mass: float = 1.0,
    stiffness: float = 1e4,
    damping: float = 0.0,
    drag: float = 0.0,
    anchor_pos: np.ndarray = None,
    taper_ratio: float = 1.0,
) -> ChainState:
    """
    Create a horizontal chain fixed at one end.

    Node 0 is the anchor (fixed). Nodes 1..N-1 extend horizontally
    to the right from the anchor point.

    Args:
        taper_ratio: Mass ratio of base (node 0) to tip (node N-1).
            1.0 = uniform mass. 10.0 = base is 10x heavier than tip.
            Mass decreases linearly from base to tip.
    """
    if anchor_pos is None:
        anchor_pos = np.array([0.0, 0.0])

    L0 = total_length / (n_nodes - 1)

    # Tapered mass: linear from base to tip
    if taper_ratio == 1.0:
        mass = np.full(n_nodes, total_mass / n_nodes)
    else:
        # Linear taper: mass[i] = a - b*i, where mass[0]/mass[N-1] = taper_ratio
        # mass[0] = m_base, mass[N-1] = m_base / taper_ratio
        # sum(mass) = total_mass
        t = np.linspace(1.0, 1.0 / taper_ratio, n_nodes)
        mass = t / t.sum() * total_mass

    pos = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        pos[i] = anchor_pos + np.array([i * L0, 0.0])

    vel = np.zeros((n_nodes, 2))
    fixed = np.zeros(n_nodes, dtype=bool)
    fixed[0] = True

    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)])
    rest_lengths = np.full(n_nodes - 1, L0)

    return ChainState(
        pos=pos, vel=vel, mass=mass, fixed=fixed,
        edges=edges, rest_lengths=rest_lengths,
        stiffness=stiffness, damping=damping, drag=drag,
        taper_ratio=taper_ratio,
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

    # Global velocity drag: F = -drag * v
    if state.drag > 0:
        for i in range(n):
            if not state.fixed[i]:
                forces[i] -= state.drag * state.vel[i]

    return forces


def project_constraints(state: ChainState, dt: float, n_iters: int = 10,
                        tol: float = 1e-8):
    """
    SHAKE-like constraint projection to enforce rod inextensibility.

    After an unconstrained position update, iteratively adjusts bead
    positions along each rod so that |r_j - r_i| = L0. Also corrects
    velocities to be consistent with the constrained positions.

    This replaces the need for very stiff springs — rods are exactly
    the right length after projection.
    """
    for _ in range(n_iters):
        max_err = 0.0
        for e in range(len(state.edges)):
            i, j = state.edges[e]
            delta = state.pos[j] - state.pos[i]
            dist = np.linalg.norm(delta)
            if dist < 1e-12:
                continue

            err = dist - state.rest_lengths[e]
            max_err = max(max_err, abs(err))

            # Correction direction
            direction = delta / dist

            # Inverse masses for weighting (fixed nodes have w=0)
            wi = 0.0 if state.fixed[i] else 1.0 / state.mass[i]
            wj = 0.0 if state.fixed[j] else 1.0 / state.mass[j]
            w_total = wi + wj
            if w_total < 1e-12:
                continue

            # Position correction proportional to inverse mass
            correction = (err / w_total) * direction
            if not state.fixed[i]:
                state.pos[i] += wi * correction
            if not state.fixed[j]:
                state.pos[j] -= wj * correction

        if max_err < tol:
            break

    # Correct velocities to be consistent with constrained positions
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        delta = state.pos[j] - state.pos[i]
        dist = np.linalg.norm(delta)
        if dist < 1e-12:
            continue

        direction = delta / dist
        vel_rel = state.vel[j] - state.vel[i]
        v_along = np.dot(vel_rel, direction)

        wi = 0.0 if state.fixed[i] else 1.0 / state.mass[i]
        wj = 0.0 if state.fixed[j] else 1.0 / state.mass[j]
        w_total = wi + wj
        if w_total < 1e-12:
            continue

        # Remove relative velocity component along the rod
        v_correction = (v_along / w_total) * direction
        if not state.fixed[i]:
            state.vel[i] += wi * v_correction
        if not state.fixed[j]:
            state.vel[j] -= wj * v_correction


def step_symplectic_euler(
    state: ChainState,
    dt: float,
    gravity: float = 9.81,
    use_constraints: bool = False,
    constraint_iters: int = 10,
) -> np.ndarray:
    """
    One symplectic Euler timestep with optional constraint projection.

    Symplectic Euler (semi-implicit):
      v_{n+1} = v_n + dt * a(x_n)
      x_{n+1} = x_n + dt * v_{n+1}

    If use_constraints=True, applies SHAKE-like projection after the
    position update to enforce rod inextensibility. This allows using
    lower spring stiffness (or zero) since constraints handle the
    length preservation.

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

    # Project constraints to enforce inextensibility
    if use_constraints:
        project_constraints(state, dt, n_iters=constraint_iters)

    return acc


def whip_crack_motion(t: float, anchor_origin: np.ndarray,
                      amplitude: float = 0.5,
                      duration: float = 0.15) -> np.ndarray:
    """
    Prescribed anchor trajectory for a whip crack.

    Quick forward flick followed by a pull-back, launching a wave
    down the chain. After the flick the anchor returns to rest.

    Args:
        t: Current time.
        anchor_origin: Rest position of the anchor.
        amplitude: Peak displacement of the flick (meters).
        duration: Total duration of the flick motion (seconds).

    Returns:
        (2,) anchor position at time t.
    """
    if t > duration:
        return anchor_origin.copy()

    # Smooth flick: sine pulse in x, small downward dip in y
    phase = np.pi * t / duration
    x_offset = amplitude * np.sin(phase)
    y_offset = -0.2 * amplitude * np.sin(2 * phase)

    return anchor_origin + np.array([x_offset, y_offset])


def compute_energy(state: ChainState, gravity: float = 9.81) -> dict:
    """
    Compute kinetic, potential (gravitational + elastic), and total energy.

    Useful for verifying energy conservation in the constraint-based
    solver and for training energy-conserving GNN surrogates.
    """
    # Kinetic energy: 0.5 * m * |v|^2
    ke = 0.5 * np.sum(state.mass[:, None] * state.vel ** 2)

    # Gravitational PE: m * g * y (y is up)
    gpe = np.sum(state.mass * gravity * state.pos[:, 1])

    # Elastic PE: 0.5 * k * stretch^2
    epe = 0.0
    for e in range(len(state.edges)):
        i, j = state.edges[e]
        delta = state.pos[j] - state.pos[i]
        dist = np.linalg.norm(delta)
        stretch = dist - state.rest_lengths[e]
        epe += 0.5 * state.stiffness * stretch ** 2

    return {"kinetic": ke, "gravitational": gpe, "elastic": epe,
            "total": ke + gpe + epe}


def run_simulation(
    n_nodes: int = 20,
    total_length: float = 2.0,
    total_mass: float = 1.0,
    gravity: float = 9.81,
    stiffness: float = 1e4,
    damping: float = 0.5,
    drag: float = 0.0,
    dt: float = 0.0001,
    n_steps: int = 50000,
    save_interval: int = 50,
    anchor_pos: Optional[np.ndarray] = None,
    taper_ratio: float = 1.0,
    use_constraints: bool = False,
    constraint_iters: int = 10,
    anchor_motion: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
) -> dict:
    """
    Run the full simulation and collect trajectory data.

    Args:
        taper_ratio: Mass ratio base/tip. 1.0 = uniform, >1 = whip taper.
        use_constraints: If True, apply SHAKE projection for inextensibility.
        constraint_iters: Number of SHAKE iterations per timestep.
        anchor_motion: Optional callable(t, anchor_origin) -> (2,) position.
            If provided, the anchor follows this prescribed trajectory.

    Returns a dict with arrays shaped for GNN training:
      positions:      (n_saved, n_nodes, 2)
      velocities:     (n_saved, n_nodes, 2)
      accelerations:  (n_saved, n_nodes, 2)
      energy:         (n_saved, 4) [kinetic, gravitational, elastic, total]
      edge_index:     (2, 2*n_edges) bidirectional
      rest_lengths:   (n_edges,)
      node_mass:      (n_nodes,) per-node mass
      node_types:     (n_nodes,) 0=free, 1=fixed
      times:          (n_saved,)
    """
    state = create_chain(n_nodes, total_length, total_mass,
                         stiffness, damping, drag, anchor_pos,
                         taper_ratio=taper_ratio)
    anchor_origin = state.pos[0].copy()

    # Preallocate storage
    n_saved = n_steps // save_interval + 1
    positions = np.zeros((n_saved, n_nodes, 2))
    velocities = np.zeros((n_saved, n_nodes, 2))
    accelerations = np.zeros((n_saved, n_nodes, 2))
    energy = np.zeros((n_saved, 4))
    times = np.zeros(n_saved)

    # Save initial state
    positions[0] = state.pos.copy()
    velocities[0] = state.vel.copy()
    accelerations[0] = 0.0
    e0 = compute_energy(state, gravity)
    energy[0] = [e0["kinetic"], e0["gravitational"], e0["elastic"], e0["total"]]
    times[0] = 0.0
    save_idx = 1

    # Time integration loop
    print_interval = max(1, n_steps // 10)
    for step_num in range(1, n_steps + 1):
        t = step_num * dt

        if step_num % print_interval == 0:
            pct = 100 * step_num / n_steps
            print(f"\r  Simulating... {pct:.0f}%", end="", flush=True)

        acc = step_symplectic_euler(state, dt, gravity,
                                   use_constraints=use_constraints,
                                   constraint_iters=constraint_iters)

        # Apply anchor motion or re-pin
        if anchor_motion is not None:
            new_anchor = anchor_motion(t, anchor_origin)
            # Compute anchor velocity from position change
            state.vel[0] = (new_anchor - state.pos[0]) / dt
            state.pos[0] = new_anchor
        else:
            state.pos[0] = anchor_origin
            state.vel[0] = 0.0

        if step_num % save_interval == 0 and save_idx < n_saved:
            positions[save_idx] = state.pos.copy()
            velocities[save_idx] = state.vel.copy()
            accelerations[save_idx] = acc.copy()
            e = compute_energy(state, gravity)
            energy[save_idx] = [e["kinetic"], e["gravitational"],
                                e["elastic"], e["total"]]
            times[save_idx] = t
            save_idx += 1

    print("\r  Simulating... done.       ")

    # Trim
    positions = positions[:save_idx]
    velocities = velocities[:save_idx]
    accelerations = accelerations[:save_idx]
    energy = energy[:save_idx]
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
        "energy": energy,
        "times": times,
        "edge_index": edge_index,
        "rest_lengths": state.rest_lengths,
        "node_mass": state.mass.copy(),
        "node_types": node_types,
        "dt": dt * save_interval,
        "gravity": gravity,
        "n_nodes": n_nodes,
        "total_mass": total_mass,
        "total_length": total_length,
        "stiffness": stiffness,
        "damping": damping,
        "taper_ratio": taper_ratio,
    }
