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
    capture_shake: bool = False,
) -> tuple:
    """
    One symplectic Euler timestep with optional constraint projection.

    Symplectic Euler (semi-implicit):
      v_{n+1} = v_n + dt * a(x_n)
      x_{n+1} = x_n + dt * v_{n+1}

    If use_constraints=True, applies SHAKE-like projection after the
    position update to enforce rod inextensibility.

    If capture_shake=True, returns the state before and after SHAKE
    for training data collection.

    Returns (acc, shake_data) where shake_data is None unless
    capture_shake=True, in which case it is a dict with:
      pre_pos, pre_vel:   state after integration, before SHAKE
      post_pos, post_vel: state after SHAKE
      pos_correction:     post_pos - pre_pos
      vel_correction:     post_vel - pre_vel
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

    # Capture pre-SHAKE state if requested
    shake_data = None
    if use_constraints:
        if capture_shake:
            pre_pos = state.pos.copy()
            pre_vel = state.vel.copy()

        project_constraints(state, dt, n_iters=constraint_iters)

        if capture_shake:
            shake_data = {
                "pre_pos": pre_pos,
                "pre_vel": pre_vel,
                "post_pos": state.pos.copy(),
                "post_vel": state.vel.copy(),
                "pos_correction": state.pos - pre_pos,
                "vel_correction": state.vel - pre_vel,
            }

    return acc, shake_data


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

        acc, _ = step_symplectic_euler(state, dt, gravity,
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


def build_tree_edges(n_beads: int) -> dict:
    """
    Build binary tree connectivity for n_beads (must be power of 2).

    Returns a dict with:
      tree_edges:     (n_tree_edges, 2) parent-child pairs
      n_total_nodes:  total nodes (beads + interior)
      node_levels:    (n_total_nodes,) level of each node (0=bead, 1..log2=interior)
      parent_of:      (n_total_nodes,) parent index (-1 for root)
      children_of:    dict mapping interior node -> (left_child, right_child)
    """
    assert n_beads > 0 and (n_beads & (n_beads - 1)) == 0, \
        f"n_beads must be power of 2, got {n_beads}"

    n_levels = int(np.log2(n_beads)) + 1  # including bead level
    n_interior = n_beads - 1
    n_total = n_beads + n_interior

    node_levels = np.zeros(n_total, dtype=np.int32)
    parent_of = np.full(n_total, -1, dtype=np.int32)
    children_of = {}
    tree_edges = []

    # Build bottom-up: pair nodes at each level
    # Level 0 nodes are 0..n_beads-1
    # Level 1 nodes are n_beads..n_beads+n_beads//2-1
    # etc.
    current_level_nodes = list(range(n_beads))
    next_id = n_beads

    for level in range(1, n_levels):
        next_level_nodes = []
        for i in range(0, len(current_level_nodes), 2):
            left = current_level_nodes[i]
            right = current_level_nodes[i + 1]
            parent = next_id
            next_id += 1

            node_levels[parent] = level
            parent_of[left] = parent
            parent_of[right] = parent
            children_of[parent] = (left, right)

            # Bidirectional edges
            tree_edges.append([parent, left])
            tree_edges.append([parent, right])
            tree_edges.append([left, parent])
            tree_edges.append([right, parent])

            next_level_nodes.append(parent)

        current_level_nodes = next_level_nodes

    tree_edges = np.array(tree_edges, dtype=np.int32)

    return {
        "tree_edges": tree_edges,
        "n_total_nodes": n_total,
        "n_interior": n_interior,
        "n_levels": n_levels,
        "node_levels": node_levels,
        "parent_of": parent_of,
        "children_of": children_of,
        "root": n_total - 1,
    }


def generate_training_data(
    n_nodes: int = 16,
    total_length: float = 2.0,
    total_mass: float = 0.5,
    gravity: float = 9.81,
    stiffness: float = 1e4,
    damping: float = 0.5,
    drag: float = 0.02,
    dt: float = 0.0001,
    n_steps: int = 30000,
    save_interval: int = 50,
    taper_ratio: float = 10.0,
    constraint_iters: int = 10,
) -> dict:
    """
    Run simulation and collect SHAKE input/output pairs for U-Net training.

    For each saved frame, captures the state before and after SHAKE
    constraint projection, providing clean input/output pairs.

    Returns a dict with:
      pre_shake_pos:    (n_saved, n_nodes, 2) positions before SHAKE
      pre_shake_vel:    (n_saved, n_nodes, 2) velocities before SHAKE
      post_shake_pos:   (n_saved, n_nodes, 2) positions after SHAKE
      post_shake_vel:   (n_saved, n_nodes, 2) velocities after SHAKE
      pos_corrections:  (n_saved, n_nodes, 2) position deltas from SHAKE
      vel_corrections:  (n_saved, n_nodes, 2) velocity deltas from SHAKE
      chain_edges:      (2, 2*n_edges) bidirectional chain connectivity
      tree:             dict from build_tree_edges()
      node_mass:        (n_nodes,) per-node mass
      node_types:       (n_nodes,) 0=free, 1=fixed
      rest_lengths:     (n_edges,) rod rest lengths
      times:            (n_saved,)
      energy_pre:       (n_saved, 4) energy before SHAKE
      energy_post:      (n_saved, 4) energy after SHAKE
    """
    assert n_nodes > 0 and (n_nodes & (n_nodes - 1)) == 0, \
        f"n_nodes must be power of 2, got {n_nodes}"

    state = create_chain(n_nodes, total_length, total_mass,
                         stiffness, damping, drag,
                         taper_ratio=taper_ratio)
    anchor_origin = state.pos[0].copy()
    tree = build_tree_edges(n_nodes)

    # Preallocate
    n_saved = n_steps // save_interval + 1
    pre_shake_pos = np.zeros((n_saved, n_nodes, 2))
    pre_shake_vel = np.zeros((n_saved, n_nodes, 2))
    post_shake_pos = np.zeros((n_saved, n_nodes, 2))
    post_shake_vel = np.zeros((n_saved, n_nodes, 2))
    pos_corrections = np.zeros((n_saved, n_nodes, 2))
    vel_corrections = np.zeros((n_saved, n_nodes, 2))
    energy_pre = np.zeros((n_saved, 4))
    energy_post = np.zeros((n_saved, 4))
    times = np.zeros(n_saved)

    # Initial state (no SHAKE needed at t=0)
    pre_shake_pos[0] = state.pos.copy()
    pre_shake_vel[0] = state.vel.copy()
    post_shake_pos[0] = state.pos.copy()
    post_shake_vel[0] = state.vel.copy()
    e0 = compute_energy(state, gravity)
    energy_pre[0] = [e0["kinetic"], e0["gravitational"], e0["elastic"], e0["total"]]
    energy_post[0] = energy_pre[0]
    times[0] = 0.0
    save_idx = 1

    print_interval = max(1, n_steps // 10)
    for step_num in range(1, n_steps + 1):
        t = step_num * dt

        if step_num % print_interval == 0:
            pct = 100 * step_num / n_steps
            print(f"\r  Generating training data... {pct:.0f}%", end="", flush=True)

        # Only capture SHAKE data on save frames
        capture = (step_num % save_interval == 0) and (save_idx < n_saved)

        acc, shake_data = step_symplectic_euler(
            state, dt, gravity,
            use_constraints=True,
            constraint_iters=constraint_iters,
            capture_shake=capture,
        )

        # Re-pin anchor
        state.pos[0] = anchor_origin
        state.vel[0] = 0.0

        if capture:
            pre_shake_pos[save_idx] = shake_data["pre_pos"]
            pre_shake_vel[save_idx] = shake_data["pre_vel"]
            post_shake_pos[save_idx] = shake_data["post_pos"]
            post_shake_vel[save_idx] = shake_data["post_vel"]
            pos_corrections[save_idx] = shake_data["pos_correction"]
            vel_corrections[save_idx] = shake_data["vel_correction"]

            # Energy before and after SHAKE
            # Temporarily set state to pre-SHAKE to compute energy
            saved_pos = state.pos.copy()
            saved_vel = state.vel.copy()
            state.pos = shake_data["pre_pos"]
            state.vel = shake_data["pre_vel"]
            e_pre = compute_energy(state, gravity)
            energy_pre[save_idx] = [e_pre["kinetic"], e_pre["gravitational"],
                                    e_pre["elastic"], e_pre["total"]]
            state.pos = saved_pos
            state.vel = saved_vel
            e_post = compute_energy(state, gravity)
            energy_post[save_idx] = [e_post["kinetic"], e_post["gravitational"],
                                     e_post["elastic"], e_post["total"]]

            times[save_idx] = t
            save_idx += 1

    print("\r  Generating training data... done.       ")

    # Trim
    pre_shake_pos = pre_shake_pos[:save_idx]
    pre_shake_vel = pre_shake_vel[:save_idx]
    post_shake_pos = post_shake_pos[:save_idx]
    post_shake_vel = post_shake_vel[:save_idx]
    pos_corrections = pos_corrections[:save_idx]
    vel_corrections = vel_corrections[:save_idx]
    energy_pre = energy_pre[:save_idx]
    energy_post = energy_post[:save_idx]
    times = times[:save_idx]

    # Chain edges (bidirectional)
    fwd = state.edges
    bwd = state.edges[:, ::-1]
    chain_edges = np.concatenate([fwd, bwd], axis=0).T

    # Node types
    node_types = np.zeros(n_nodes, dtype=np.int32)
    node_types[0] = 1

    return {
        "pre_shake_pos": pre_shake_pos,
        "pre_shake_vel": pre_shake_vel,
        "post_shake_pos": post_shake_pos,
        "post_shake_vel": post_shake_vel,
        "pos_corrections": pos_corrections,
        "vel_corrections": vel_corrections,
        "chain_edges": chain_edges,
        "tree_edges": tree.pop("tree_edges"),
        "tree_meta": tree,
        "node_mass": state.mass.copy(),
        "node_types": node_types,
        "rest_lengths": state.rest_lengths.copy(),
        "times": times,
        "energy_pre": energy_pre,
        "energy_post": energy_post,
        "dt": dt,
        "gravity": gravity,
        "n_nodes": n_nodes,
        "stiffness": stiffness,
        "damping": damping,
        "drag": drag,
        "taper_ratio": taper_ratio,
    }
