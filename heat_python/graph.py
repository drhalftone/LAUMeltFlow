"""Graph + training-pair construction for the heat-shield GNN surrogate.

Stage 2: turn a recorded trajectory (heat_python/data_gen.py) into GNN-ready
arrays. Mirrors meltflow_gnn/graph.py conventions but for the heat-shield's
per-cell multi-physics state.

Graph (fixed for a given mesh):
    Nodes  = all m = n_cells + 2 cells, INCLUDING the two ghost cells. The
             ghosts carry the boundary forcing (wall temperature / aero flux),
             so the GNN sees the drivers as node state at the domain ends.
    Edges  = the 1D path: each adjacent pair (i, i+1) connected both ways.
    Edge feature = interface spacing 0.5*(dx[i]+dx[i+1])  (cell-center distance).

Node features (F = n_species + 3):  [T, rho, rho_i[0..ns-1], porosity]
    rho and porosity are derived from rho_i, so they are redundant DOF kept as
    convenience inputs; the independent state is [T, rho_i].

A built dataset (.npz), with P = number of time-pairs:
    node_in    (P, m, F)     input node features at time t (incl. ghosts)
    target     (P, n, F)     interior next-state node features at t+gap
    target_delta (P, n, F)   target - node_in[:, interior]  (often easier)
    edge_index (2, E)        shared graph connectivity
    edge_attr  (E, 1)        shared edge features
    node_mean/node_std (F,)  normalization stats (std clamped >= 1e-8)
    + meta: gap, n_cells, n_species, feature_names, case_name

Usage:
    python -m heat_python.graph --traj heat_python/data/aw1_traj.npz \
        --gap 1 --out heat_python/data/aw1_pairs_gap1.npz
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .data_gen import load_trajectory


def _porosity(rho: np.ndarray, rhov_b: float, rhoc_b: float,
              phi: float = 0.8, phi_c: float = 0.85) -> np.ndarray:
    """Vectorized CALC_POROSITY (kept local so graph.py has no solver deps).
    phi/phi_c default to the TACOT values; override if a case differs."""
    beta = np.clip((rhov_b - rho) / max(rhov_b - rhoc_b, 1.0e-30), 0.0, 1.0)
    por = phi + (phi_c - phi) * beta
    lo, hi = min(phi, phi_c), max(phi, phi_c)
    return np.clip(np.clip(por, lo, hi), 1.0e-6, 0.99)


def node_features(traj: dict, phi: float = 0.8, phi_c: float = 0.85
                  ) -> tuple[np.ndarray, list[str]]:
    """Build (S, m, F) node features for every snapshot, including ghost cells.

    Base (always): [T, rho, rho_i..., porosity].
    Gas runs (when the trajectory recorded them): + [pg, mdotf], the gas
    pressure and face mass flux, appended at the end so the base column layout
    is unchanged."""
    T = traj["T"]                      # (S, m)
    rho = traj["rho"]                  # (S, m)
    rho_i = traj["rho_i"]              # (S, ns, m)
    rhov_b, rhoc_b = float(traj["rhov_bulk"]), float(traj["rhoc_bulk"])
    S, ns, m = rho_i.shape
    por = _porosity(rho.ravel(), rhov_b, rhoc_b, phi, phi_c).reshape(S, m)

    feats = [T[:, :, None], rho[:, :, None]]
    feats += [rho_i[:, k, :, None] for k in range(ns)]
    feats.append(por[:, :, None])
    names = ["T", "rho"] + [f"rho_i{k}" for k in range(ns)] + ["porosity"]

    if "pg" in traj and "mdotf" in traj and traj["pg"].size:
        feats.append(traj["pg"][:, :, None])
        feats.append(traj["mdotf"][:, :, None])
        names += ["pg", "mdotf"]

    X = np.concatenate(feats, axis=2)          # (S, m, F)
    return X.astype(np.float64), names


def mesh_graph(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed 1D path graph over all m cells (incl. ghosts).
    Returns edge_index (2, E) and edge_attr (E, 1) = interface spacing."""
    m = int(traj["n_cells"]) + 2
    dx = traj["dx"]                            # (m,)
    src, dst, attr = [], [], []
    for i in range(m - 1):
        spacing = 0.5 * (dx[i] + dx[i + 1])
        src += [i, i + 1]                      # both directions
        dst += [i + 1, i]
        attr += [[spacing], [spacing]]
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attr, dtype=np.float64)
    return edge_index, edge_attr


def build_adaptive_dataset(traj: dict, gaps=(1, 10), phi: float = 0.8,
                           phi_c: float = 0.85, stride: int = 1) -> dict:
    """Mixed-cadence dataset: pairs built at SEVERAL super-step sizes (gaps), each
    tagged with its physical time step dt as an extra node feature 'dt'. Training
    on a range of dt teaches the model the dt-dependence of the update, so it can
    be rolled out with an ADAPTIVE schedule -- fine dt near fast transients, coarse
    dt on the slow plateau. (Same idea as carrying dx for mesh generalization.)

    'dt' is an input-only feature: it is not predicted (excluded from out_cols)
    and must be re-imposed each step at rollout from the chosen schedule."""
    X, names = node_features(traj, phi, phi_c)          # (S, m, F)
    S, m, F = X.shape
    n = int(traj["n_cells"])
    snap_dt = float(np.median(np.diff(traj["time"])))

    ins, tgts, deltas = [], [], []
    for gap in gaps:
        if S - gap < 1:
            continue
        ni = X[:S - gap][::stride]                      # (P, m, F)  (subsampled)
        tg = X[gap:][::stride][:, 1:n + 1, :]           # (P, n, F)
        td = tg - ni[:, 1:n + 1, :]
        dt_val = gap * snap_dt
        ni = np.concatenate([ni, np.full((ni.shape[0], m, 1), dt_val)], axis=2)
        tg = np.concatenate([tg, np.full((tg.shape[0], n, 1), dt_val)], axis=2)
        td = np.concatenate([td, np.zeros((td.shape[0], n, 1))], axis=2)  # dt not predicted
        ins.append(ni); tgts.append(tg); deltas.append(td)

    node_in = np.concatenate(ins).astype(np.float64)    # (P, m, F+1)
    target = np.concatenate(tgts).astype(np.float64)
    target_delta = np.concatenate(deltas).astype(np.float64)
    names = list(names) + ["dt"]

    flat = node_in.reshape(-1, F + 1)
    node_mean = flat.mean(axis=0)
    node_std = flat.std(axis=0)
    node_std[node_std < 1e-8] = 1.0

    edge_index, edge_attr = mesh_graph(traj)
    return dict(
        node_in=node_in, target=target, target_delta=target_delta,
        edge_index=edge_index, edge_attr=edge_attr,
        node_mean=node_mean, node_std=node_std,
        gap=int(gaps[0]), n_cells=n, n_species=int(traj["n_species"]),
        feature_names=np.array(names), case_name=traj["case_name"],
        dt_gap=snap_dt, snap_dt=snap_dt, gaps=np.array(gaps),
    )


def build_dataset(traj: dict, gap: int = 1, phi: float = 0.8,
                  phi_c: float = 0.85) -> dict:
    """Form (graph_t, interior-state_{t+gap}) training pairs from a trajectory.

    gap = number of recorded snapshots between input and target (the GNN
    super-step, in units of the trajectory's record_every). gap=1 means
    predict the next recorded snapshot."""
    X, names = node_features(traj, phi, phi_c)     # (S, m, F)
    S, m, F = X.shape
    n = int(traj["n_cells"])
    if S - gap < 1:
        raise ValueError(f"trajectory too short ({S} snaps) for gap={gap}")

    node_in = X[:S - gap]                           # (P, m, F)
    nxt = X[gap:]                                   # (P, m, F)
    target = nxt[:, 1:n + 1, :]                     # interior only (P, n, F)
    target_delta = target - node_in[:, 1:n + 1, :]

    flat = node_in.reshape(-1, F)
    node_mean = flat.mean(axis=0)
    node_std = flat.std(axis=0)
    node_std[node_std < 1e-8] = 1.0                 # inert/constant features

    edge_index, edge_attr = mesh_graph(traj)
    return dict(
        node_in=node_in, target=target, target_delta=target_delta,
        edge_index=edge_index, edge_attr=edge_attr,
        node_mean=node_mean, node_std=node_std,
        gap=gap, n_cells=n, n_species=int(traj["n_species"]),
        feature_names=np.array(names), case_name=traj["case_name"],
        dt_gap=float(np.median(np.diff(traj["time"]))) * gap,
    )


def to_neighbor_samples(node_in: np.ndarray, target_delta: np.ndarray,
                        dx: np.ndarray, out_cols, traj_id: np.ndarray = None
                        ) -> dict:
    """Convert a full-mesh dataset into bead-style per-cell training samples.

    For every (pair, interior cell), emit the cell's absolute state, the
    relative state of its left/right neighbor (neighbor - self) plus dx, and
    the target delta (only the predicted columns `out_cols`). Boundary cells
    always have both neighbors here because the ghost cells are included.

    node_in (P, m, F); target_delta (P, n, F); dx (m,). Returns flat arrays of
    N = P*n samples ready for HeatMPGNN.forward.
    """
    P, m, F = node_in.shape
    n = m - 2
    out_cols = np.asarray(out_cols, dtype=int)

    self_f = node_in[:, 1:n + 1, :]                       # (P, n, F)
    left_f = node_in[:, 0:n, :]                           # left neighbors
    right_f = node_in[:, 2:n + 2, :]                      # right neighbors
    dx_int = np.broadcast_to(dx[1:n + 1], (P, n))[..., None]

    left_edge = np.concatenate([left_f - self_f, dx_int], axis=2)   # (P,n,F+1)
    right_edge = np.concatenate([right_f - self_f, dx_int], axis=2)
    target = target_delta[:, :, out_cols]                 # (P, n, |out_cols|)

    flat = lambda a: a.reshape(-1, a.shape[-1]).astype(np.float32)
    out = dict(
        node_feat=flat(self_f), left_edge=flat(left_edge),
        right_edge=flat(right_edge), target=flat(target),
        out_cols=out_cols,
    )
    if traj_id is not None:
        out["sample_traj_id"] = np.repeat(traj_id, n)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    traj = load_trajectory(args.traj)
    ds = build_dataset(traj, gap=args.gap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **ds)
    print(f"dataset: {ds['node_in'].shape[0]} pairs, "
          f"{ds['node_in'].shape[1]} nodes x {ds['node_in'].shape[2]} feats, "
          f"gap={args.gap} (dt_gap~{ds['dt_gap']:.4e}s) -> {out}")
    print(f"  features: {list(ds['feature_names'])}")
    print(f"  node_mean: {np.array2string(ds['node_mean'], precision=2)}")
    print(f"  node_std:  {np.array2string(ds['node_std'], precision=2)}")
