"""Generate a real multi-trajectory GNN dataset for the heat shield.

First dataset: sweep the boundary forcing (hot-wall temperature history) on
the fast gas-off aw1 case (~conduction + pyrolysis). Each sample is a distinct
ramp-and-hold wall-temperature profile T_wall(t); the solver runs it, we record
the full-field trajectory, and build (graph_t, state_{t+gap}) pairs. All
trajectories are concatenated into one dataset with a per-pair trajectory id
(for leakage-free train/val splits) and a single global normalization.

Holding the mesh fixed and varying only the forcing teaches the surrogate to
respond to different heating scenarios; mesh-resolution / material sweeps come
later. Deterministic (seeded) so the dataset is reproducible.

    python -m heat_python.make_dataset --n 24 --record-every 10 --gap 1 \
        --out heat_python/data/aw1_forcing_dataset.npz
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .solver import run
from .io_files import TimeTable
from .graph import build_dataset


def _forcing(t_ramp: float, t_peak: float, t_hold_end: float,
             init_temp: float, t_final: float, cool: bool) -> TimeTable:
    """Ramp from init_temp to t_peak over t_ramp, hold, optionally cool back
    toward init_temp by t_final."""
    ts = [0.0, t_ramp]
    vs = [init_temp, t_peak]
    if cool:
        ts += [t_hold_end, t_final]
        vs += [t_peak, init_temp + 0.25 * (t_peak - init_temp)]
    else:
        ts += [t_final]
        vs += [t_peak]
    return TimeTable(t=np.array(ts), v=np.array(vs))


def make(case_dir: Path, n: int, record_every: int, gap: int,
         out_path: Path, verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    init_temp, t_final = 298.15, 60.0

    chunks = []          # per-trajectory dataset dicts
    traj_ids = []
    for i in range(n):
        t_peak = float(rng.uniform(700.0, 2000.0))
        t_ramp = float(rng.uniform(0.5, 15.0))
        cool = bool(rng.uniform() < 0.4)
        t_hold_end = float(rng.uniform(t_ramp + 5.0, t_final - 2.0))
        tt = _forcing(t_ramp, t_peak, t_hold_end, init_temp, t_final, cool)

        res = run(case_dir, verbose=False, lgas=False,
                  record_every=record_every, write_con=False, time_table=tt)
        traj = res["trajectory"]
        ds = build_dataset(traj, gap=gap)
        chunks.append(ds)
        traj_ids.append(np.full(ds["node_in"].shape[0], i, dtype=np.int32))
        if verbose:
            print(f"  [{i+1}/{n}] peak={t_peak:6.1f}K ramp={t_ramp:4.1f}s "
                  f"cool={cool} -> {ds['node_in'].shape[0]} pairs")

    node_in = np.concatenate([c["node_in"] for c in chunks]).astype(np.float32)
    target = np.concatenate([c["target"] for c in chunks]).astype(np.float32)
    target_delta = np.concatenate(
        [c["target_delta"] for c in chunks]).astype(np.float32)
    traj_id = np.concatenate(traj_ids)

    # Global normalization over all input nodes (std clamped for constants).
    flat = node_in.reshape(-1, node_in.shape[-1])
    node_mean = flat.mean(axis=0)
    node_std = flat.std(axis=0)
    node_std[node_std < 1e-8] = 1.0

    ref = chunks[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        node_in=node_in, target=target, target_delta=target_delta,
        traj_id=traj_id,
        edge_index=ref["edge_index"], edge_attr=ref["edge_attr"],
        node_mean=node_mean.astype(np.float32),
        node_std=node_std.astype(np.float32),
        gap=gap, n_cells=ref["n_cells"], n_species=ref["n_species"],
        feature_names=ref["feature_names"], dt_gap=ref["dt_gap"],
        n_trajectories=n,
    )
    if verbose:
        print(f"\ndataset: {node_in.shape[0]} pairs from {n} trajectories, "
              f"{node_in.shape[1]} nodes x {node_in.shape[2]} feats -> {out_path}")
        print(f"  node_mean {np.array2string(node_mean, precision=2)}")
        print(f"  node_std  {np.array2string(node_std, precision=2)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--out", default="heat_python/data/aw1_forcing_dataset.npz")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    make(cd, args.n, args.record_every, args.gap, Path(args.out))
