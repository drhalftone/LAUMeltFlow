"""Fine-cadence multi-trajectory gas dataset for aw2 (gas + aerothermal).

Same forcing sweep as make_aw2_dataset.py, but records at a FINE cadence
(record-every 200 -> dt ~ 0.018 s, the aw1 regime) instead of the coarse
super-step. The coarse super-step existed only to save computation; dropping
that requirement removes aw2's dominant failure mode (per-step surface jumps of
~700 K that a single delta-prediction cannot track). At this cadence per-step
jumps fall back to aw1 scale (~40 K), so the aw1 recipe (noise + multi-step
BPTT + conservative flux form) can be transplanted directly.

Each aw2 trajectory is a full ~15 min solver run, so the trajectories run in
parallel (one process each). Forcings are pre-drawn from a single seeded RNG so
trajectory i is identical to the sequential make_aw2_dataset.py draw order.

    python -m heat_python.gnn.make_aw2_fine --n 8 --record-every 200 \
        --out heat_python/data/aw2_fine_dataset.npz
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..solver import run
from ..io_files import TimeTable
from .graph import build_dataset


def _flux_forcing(peak: float, t_on: float, t_off: float, t_final: float
                  ) -> TimeTable:
    ts = [0.0, t_on, t_off, t_off + 0.1, t_final]
    vs = [0.0, peak, peak, 0.0, 0.0]
    return TimeTable(t=np.array(ts), v=np.array(vs))


def _draw_forcings(n: int, t_final: float) -> list[tuple[float, float, float]]:
    """Pre-draw all n forcings from one rng(0), matching the sequential order
    of make_aw2_dataset.py so trajectory i is reproducible across runs."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        peak = float(rng.uniform(0.8e6, 2.0e6))
        t_on = float(rng.uniform(0.1, 3.0))
        t_off = float(rng.uniform(40.0, 100.0))
        out.append((peak, t_on, t_off))
    return out


def _worker(i: int, peak: float, t_on: float, t_off: float,
            case_dir: str, record_every: int, gap: int, tmp_dir: str,
            t_final: float) -> str:
    """Run one trajectory, build its dataset chunk, save to tmp_dir/traj_i.npz.
    Returns the path. Chunk 0 also carries the shared graph metadata."""
    tt = _flux_forcing(peak, t_on, t_off, t_final)
    res = run(Path(case_dir), verbose=False, lgas=True,
              record_every=record_every, write_con=False, time_table=tt)
    ds = build_dataset(res["trajectory"], gap=gap)
    out = Path(tmp_dir) / f"traj_{i}.npz"
    payload = dict(
        node_in=ds["node_in"].astype(np.float32),
        target=ds["target"].astype(np.float32),
        target_delta=ds["target_delta"].astype(np.float32),
        peak=peak, t_on=t_on, t_off=t_off,
    )
    if i == 0:
        payload.update(
            edge_index=ds["edge_index"], edge_attr=ds["edge_attr"],
            n_cells=ds["n_cells"], n_species=ds["n_species"],
            feature_names=ds["feature_names"], dt_gap=ds["dt_gap"],
        )
    np.savez_compressed(out, **payload)
    return str(out)


def make(case_dir: Path, n: int, record_every: int, gap: int,
         out_path: Path, workers: int, tmp_dir: Path) -> None:
    t_final = 120.0
    tmp_dir.mkdir(parents=True, exist_ok=True)
    forcings = _draw_forcings(n, t_final)

    print(f"generating {n} fine-cadence aw2 trajectories "
          f"(record-every {record_every}) on {workers} workers...", flush=True)
    paths: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, i, p, on, off, str(case_dir),
                          record_every, gap, str(tmp_dir), t_final): i
                for i, (p, on, off) in enumerate(forcings)}
        for fut in as_completed(futs):
            i = futs[fut]
            paths[i] = fut.result()
            p, on, off = forcings[i]
            print(f"  [{len(paths)}/{n}] traj {i}: peak={p/1e6:.2f}MW "
                  f"t_on={on:.2f}s t_off={off:4.0f}s done", flush=True)

    # Merge in trajectory order.
    node_in, target, target_delta, traj_ids = [], [], [], []
    ref = None
    for i in range(n):
        d = np.load(paths[i], allow_pickle=True)
        node_in.append(d["node_in"])
        target.append(d["target"])
        target_delta.append(d["target_delta"])
        traj_ids.append(np.full(d["node_in"].shape[0], i, dtype=np.int32))
        if i == 0:
            ref = d

    node_in = np.concatenate(node_in).astype(np.float32)
    target = np.concatenate(target).astype(np.float32)
    target_delta = np.concatenate(target_delta).astype(np.float32)
    traj_id = np.concatenate(traj_ids)

    flat = node_in.reshape(-1, node_in.shape[-1])
    node_mean = flat.mean(axis=0)
    node_std = flat.std(axis=0)
    node_std[node_std < 1e-8] = 1.0

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
    print(f"\ndataset: {node_in.shape[0]} pairs from {n} trajectories, "
          f"{node_in.shape[1]} nodes x {node_in.shape[2]} feats -> {out_path}")
    print(f"  features: {list(ref['feature_names'])}")


if __name__ == "__main__":
    import argparse
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--record-every", type=int, default=200)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = min(n, cores-2)")
    p.add_argument("--out", default="heat_python/data/aw2_fine_dataset.npz")
    p.add_argument("--tmp-dir", default="heat_python/data/_fine_tmp")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    workers = args.workers or min(args.n, (os.cpu_count() or 4) - 2)
    make(cd, args.n, args.record_every, args.gap, Path(args.out),
         workers, Path(args.tmp_dir))
