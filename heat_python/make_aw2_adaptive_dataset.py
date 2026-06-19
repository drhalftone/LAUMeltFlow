"""Multi-trajectory ADAPTIVE (dt-aware, fine-cadence) gas dataset for aw2.

For the proper HELD-OUT test of the adaptive surrogate: sweep aw2's flux history
(like make_aw2_dataset.py) but record at FINE cadence (record_every=200) and build
mixed-cadence dt-feature pairs (graph.build_adaptive_dataset, gaps=(1,10)). Pairs
are subsampled (stride) to keep the multi-trajectory dataset tractable. Each pair
carries a traj id for leakage-free splits. Train on this, then roll out on a
held-out forcing (e.g. the stock aw2_traj.npz at 1.5 MW / t_off=60, which is NOT
one of the swept forcings).

    python -m heat_python.make_aw2_adaptive_dataset --n 8 --record-every 200 \
        --stride 5 --out heat_python/data/aw2_adaptive_sweep.npz
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .solver import run
from .io_files import TimeTable
from .graph import build_adaptive_dataset


def _flux_forcing(peak, t_on, t_off, t_final):
    ts = [0.0, t_on, t_off, t_off + 0.1, t_final]
    vs = [0.0, peak, peak, 0.0, 0.0]
    return TimeTable(t=np.array(ts), v=np.array(vs))


def make(case_dir, n, record_every, stride, gaps, out_path, verbose=True):
    rng = np.random.default_rng(0)        # same seed as make_aw2_dataset
    t_final = 120.0
    chunks, traj_ids = [], []
    for i in range(n):
        peak = float(rng.uniform(0.8e6, 2.0e6))
        t_on = float(rng.uniform(0.1, 3.0))
        t_off = float(rng.uniform(40.0, 100.0))
        tt = _flux_forcing(peak, t_on, t_off, t_final)
        res = run(case_dir, verbose=False, lgas=True, record_every=record_every,
                  write_con=False, time_table=tt)
        ds = build_adaptive_dataset(res["trajectory"], gaps=gaps, stride=stride)
        chunks.append(ds)
        traj_ids.append(np.full(ds["node_in"].shape[0], i, dtype=np.int32))
        if verbose:
            print(f"  [{i+1}/{n}] peak={peak/1e6:.2f}MW t_off={t_off:4.0f}s "
                  f"-> {ds['node_in'].shape[0]} pairs", flush=True)

    node_in = np.concatenate([c["node_in"] for c in chunks]).astype(np.float32)
    target = np.concatenate([c["target"] for c in chunks]).astype(np.float32)
    target_delta = np.concatenate([c["target_delta"] for c in chunks]).astype(np.float32)
    traj_id = np.concatenate(traj_ids)

    flat = node_in.reshape(-1, node_in.shape[-1])
    node_mean = flat.mean(0); node_std = flat.std(0); node_std[node_std < 1e-8] = 1.0
    ref = chunks[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, node_in=node_in, target=target, target_delta=target_delta,
        traj_id=traj_id, edge_index=ref["edge_index"], edge_attr=ref["edge_attr"],
        node_mean=node_mean.astype(np.float32), node_std=node_std.astype(np.float32),
        gap=1, n_cells=ref["n_cells"], n_species=ref["n_species"],
        feature_names=ref["feature_names"], dt_gap=ref["dt_gap"], n_trajectories=n)
    if verbose:
        print(f"\ndataset: {node_in.shape[0]} pairs from {n} traj, "
              f"{node_in.shape[1]} nodes x {node_in.shape[2]} feats -> {out_path}")
        print(f"  features: {list(ref['feature_names'])}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--record-every", type=int, default=200)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--gaps", type=int, nargs="+", default=[1, 10])
    p.add_argument("--out", default="heat_python/data/aw2_adaptive_sweep.npz")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    make(cd, args.n, args.record_every, args.stride, tuple(args.gaps), Path(args.out))
