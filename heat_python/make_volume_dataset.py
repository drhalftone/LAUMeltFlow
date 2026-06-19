"""Volume (state-space) sampling for heat-shield training data.

Applies the symposium paper's headline finding: trajectory-recorded data traces
a thin path through state space and causes rollout drift; sampling the reachable
state space directly ("volume sampling") gives stable rollouts. We synthesize
random physically-plausible local field windows, advance them with the REAL
solver physics, and emit single-step (center-cell + 2 neighbors -> center delta)
training samples covering state space far more broadly than 24 trajectories.

Heat-shield twist: the GNN's step is K_SUPER=10 solver steps, whose true domain
of dependence is +-10 cells. So each sample is a WIDE window (W cells); we run
the solver K_SUPER steps, and read only the central cell (kept >K_SUPER cells
from the window edges, so block-boundary errors never reach it) plus its two
immediate neighbors at t=0. Many windows are tiled into one big mesh and stepped
together (conduction is nearest-neighbor + pyrolysis is per-cell, so each
window's center depends only on its own cells).

Sampling distribution (the crucial part -- must match the manifold the system
lives on, per the paper's Cartesian-diverges / polar-works finding):
  - smooth T profiles (linear + curvature), realistic neighbor gradients
  - char "virginity" w in [0,1] per cell -> rho_i = char + w*(virgin-char),
    smoothly varying; T and w sampled broadly but as SMOOTH fields, not
    independent per-cell uniform noise (which would be off-manifold).

Output matches the full-mesh dataset format with m=3, n=1 (each sample is a
3-cell mini-mesh), so train_gnn.py consumes it unchanged.

    python -m heat_python.make_volume_dataset --n 12000 \
        --out heat_python/data/aw1_volume_dataset.npz
"""

from __future__ import annotations
from copy import copy
from pathlib import Path

import numpy as np

from .case import load_case
from .domain import setup_domain
from .materials import load_materials
from .pyrolysis import load_solid
from .solver import State, step_physics, compute_initial_dt

K_SUPER = 10          # solver steps per GNN super-step (matches record_every)
W = 25                # window width (center at 12, >K_SUPER from both edges)


def make(case_dir: Path, n: int, out_path: Path, seed: int = 0,
         verbose: bool = True):
    rng = np.random.default_rng(seed)
    case = load_case(case_dir / "heat.case")
    mats = load_materials(case, case_dir, lrad=False, leff=False)
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    rho_w = solid.rho_w
    virgin = solid.rho_v          # (ns,)
    char = solid.rho_c
    ns = solid.n_species

    # Tile n windows of W cells into one mesh; same dx as aw1 (uniform).
    dx_aw1 = case.l0 / case.nbrn
    vcase = copy(case)
    vcase.nbrn = n * W
    vcase.l0 = vcase.nbrn * dx_aw1
    domain = setup_domain(vcase)
    m = domain.n_cells + 2
    dt = compute_initial_dt(case, setup_domain(case), mats, solid)
    fluxcm = 2.0 / (domain.dx[1:m - 1] + domain.dx[0:m - 2])
    fluxcp = 2.0 / (domain.dx[2:m] + domain.dx[1:m - 1])

    # --- Sample smooth (T, virginity) field windows ---
    c = W // 2                                   # center index within a window
    j = np.arange(W) - c                          # (W,) offset from center
    # Gentle, realistic gradients keep the windows on the smooth-field manifold
    # (the heat-shield analog of the paper's "polar near rest length"); broad /
    # steep sampling gives stable loss but diverging rollouts.
    T_c = rng.uniform(298.15, 1750.0, n)          # center temperature
    slope = rng.normal(0.0, 10.0, n)              # per-cell T gradient [K/cell]
    curv = rng.normal(0.0, 0.4, n)                # T curvature
    T = T_c[:, None] + slope[:, None] * j[None, :] + curv[:, None] * j[None, :] ** 2
    T = np.clip(T, 298.15, 1800.0)                # (n, W)

    # Char state must CORRELATE with temperature (the diagnosed fix): a hot cell
    # cannot be virgin -- it has already pyrolyzed. Cap per-cell virginity by a
    # temperature-dependent ceiling spanning the pyrolysis range (~700-1500 K),
    # then scale by a per-window level u. This keeps (T, char) on the manifold
    # and removes the explosive hot-but-virgin samples that diverged in v1/v2.
    w_cap = np.clip((1500.0 - T) / 800.0 + 0.2, 0.0, 1.0)   # (n, W) ceiling
    u = rng.uniform(0.0, 1.0, n)                            # per-window char level
    w = u[:, None] * w_cap                                  # smooth, T-correlated
    # rho_i = char + w*(virgin-char) per species (smooth char field)
    rho_i_win = (char[None, :, None]
                 + w[:, None, :] * (virgin - char)[None, :, None])  # (n,ns,W)

    # --- Lay windows into the big mesh ---
    T_full = np.empty(m); rho_i_full = np.empty((ns, m))
    T_full[1:m - 1] = T.reshape(-1)
    rho_i_full[:, 1:m - 1] = np.transpose(rho_i_win, (1, 0, 2)).reshape(ns, -1)
    T_full[0] = T_full[1]; T_full[m - 1] = T_full[m - 2]        # arbitrary ghosts
    rho_i_full[:, 0] = rho_i_full[:, 1]; rho_i_full[:, m - 1] = rho_i_full[:, m - 2]
    rho_full = (rho_w[:, None] * rho_i_full).sum(0)

    state = State(time=0.0, T=T_full.copy(), rho=rho_full.copy(),
                  rho_i=rho_i_full.copy())
    T0, rhoi0 = state.T.copy(), state.rho_i.copy()             # inputs at t=0

    # --- Advance the real physics K_SUPER steps ---
    for _ in range(K_SUPER):
        state = step_physics(state, dt, case, domain, mats, solid,
                             fluxcm, fluxcp, gas=None, lgas=False,
                             gas_energy=False, pamb=None)
    if verbose:
        print(f"  stepped {n} windows x {W} cells, {K_SUPER} solver steps, "
              f"dt={dt:.3e}s (super-step dt={K_SUPER*dt:.3e}s)")

    # --- Read center cell (+ its 2 neighbors) per window ---
    # window k occupies interior cells [1+kW .. kW+W]; center at 1+kW+c.
    base = 1 + np.arange(n) * W
    ctr = base + c
    F = ns + 3                                    # [T, rho, rho_i..., porosity]

    def feats(idx, T_arr, rhoi_arr):
        rho = (rho_w[:, None] * rhoi_arr[:, idx]).sum(0)       # (n,)
        beta = np.clip((solid.rhov_bulk - rho)
                       / max(solid.rhov_bulk - solid.rhoc_bulk, 1e-30), 0, 1)
        por = np.clip(solid.phi + (case.phi_c - solid.phi) * beta,
                      min(solid.phi, case.phi_c), max(solid.phi, case.phi_c))
        cols = [T_arr[idx], rho] + [rhoi_arr[k, idx] for k in range(ns)] + [por]
        return np.stack(cols, axis=1)             # (n, F)

    self0 = feats(ctr, T0, rhoi0)
    left0 = feats(ctr - 1, T0, rhoi0)
    right0 = feats(ctr + 1, T0, rhoi0)
    self_next = feats(ctr, state.T, state.rho_i)

    # Full-mesh format with m=3 (left|center|right), n=1 interior -> train_gnn.
    node_in = np.stack([left0, self0, right0], axis=1).astype(np.float32)   # (n,3,F)
    target = self_next[:, None, :].astype(np.float32)                       # (n,1,F)
    target_delta = (target - node_in[:, 1:2, :]).astype(np.float32)
    # 3-cell mini-mesh: 2 interfaces, both directions -> 4 directed edges.
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_attr = np.full((4, 1), dx_aw1, dtype=np.float32)
    traj_id = (np.arange(n) % 24).astype(np.int32)              # pseudo-groups

    flat = node_in.reshape(-1, F)
    node_mean = flat.mean(0); node_std = flat.std(0); node_std[node_std < 1e-8] = 1.0
    names = ["T", "rho"] + [f"rho_i{k}" for k in range(ns)] + ["porosity"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, node_in=node_in, target=target, target_delta=target_delta,
        traj_id=traj_id, edge_index=edge_index, edge_attr=edge_attr,
        node_mean=node_mean.astype(np.float32),
        node_std=node_std.astype(np.float32),
        gap=1, n_cells=1, n_species=ns,
        feature_names=np.array(names), dt_gap=float(K_SUPER * dt),
        n_trajectories=24)
    if verbose:
        print(f"dataset: {n} volume samples (3-cell mini-meshes) -> {out_path}")
        print(f"  T sampled [{T.min():.0f},{T.max():.0f}]K  "
              f"target dT std {target_delta[:,0,0].std():.2f}K "
              f"max {np.abs(target_delta[:,0,0]).max():.0f}K")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1")
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--out", default="heat_python/data/aw1_volume_dataset.npz")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    make(cd, args.n, Path(args.out))
