"""Multi-trajectory gas dataset for the aw2 (gas + aerothermal) case.

The aw1 forcing sweep (make_dataset.py) varied a wall-temperature ramp on the
gas-off case. Here we vary aw2's heat-flux history (its time_flux table: a pulse
that ramps to a peak flux, holds, then drops to zero) with the gas physics ON,
to give the surrogate the diverse trajectories a stable rollout needs. Each run
records the full gas field (T, rho, rho_i, pg, mdotf) at a coarse cadence so the
rollout step count stays manageable.

    python -m heat_python.make_aw2_dataset --n 8 --record-every 2000 \
        --out heat_python/data/aw2_forcing_dataset.npz
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .solver import run
from .io_files import TimeTable
from .graph import build_dataset


def _flux_forcing(peak: float, t_on: float, t_off: float, t_final: float
                  ) -> TimeTable:
    """Heat-flux pulse: 0 -> peak by t_on, hold to t_off, back to 0 by t_off+0.1,
    zero to t_final. Mirrors the shape of time_flux_tc21.dat."""
    ts = [0.0, t_on, t_off, t_off + 0.1, t_final]
    vs = [0.0, peak, peak, 0.0, 0.0]
    return TimeTable(t=np.array(ts), v=np.array(vs))


def make(case_dir: Path, n: int, record_every: int, gap: int,
         out_path: Path, verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    t_final = 120.0

    chunks, traj_ids = [], []
    for i in range(n):
        peak = float(rng.uniform(0.8e6, 2.0e6))      # 0.8-2.0 MW/m^2
        t_on = float(rng.uniform(0.1, 3.0))          # ramp-up time
        t_off = float(rng.uniform(40.0, 100.0))      # heat-off time
        tt = _flux_forcing(peak, t_on, t_off, t_final)

        res = run(case_dir, verbose=False, lgas=True,
                  record_every=record_every, write_con=False, time_table=tt)
        traj = res["trajectory"]
        ds = build_dataset(traj, gap=gap)
        chunks.append(ds)
        traj_ids.append(np.full(ds["node_in"].shape[0], i, dtype=np.int32))
        if verbose:
            print(f"  [{i+1}/{n}] peak={peak/1e6:.2f}MW t_off={t_off:4.0f}s "
                  f"-> {ds['node_in'].shape[0]} pairs", flush=True)

    node_in = np.concatenate([c["node_in"] for c in chunks]).astype(np.float32)
    target = np.concatenate([c["target"] for c in chunks]).astype(np.float32)
    target_delta = np.concatenate(
        [c["target_delta"] for c in chunks]).astype(np.float32)
    traj_id = np.concatenate(traj_ids)

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
        print(f"  features: {list(ref['feature_names'])}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--record-every", type=int, default=2000)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--out", default="heat_python/data/aw2_forcing_dataset.npz")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    make(cd, args.n, args.record_every, args.gap, Path(args.out))
