"""Generate a HELD-OUT aw2 (gas) trajectory for rollout evaluation.

The aw2 forcing sweep (make_aw2_dataset.py) trains on 8 random flux histories
(rng seed 0). This makes one trajectory with a deterministic mid-range forcing
that is NOT one of those 8, recorded at the SAME cadence as the training set
(record_every=2000, so the snapshot dt matches what the model trained on -- no
CFL/dt-mismatch confound). Use it with eval_rollout_traj.py as a genuine
cross-forcing generalization test.

    python -m heat_python.make_aw2_holdout_traj \
        --out heat_python/data/aw2_holdout_traj.npz
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .solver import run
from .io_files import TimeTable


def _flux_forcing(peak, t_on, t_off, t_final):
    ts = [0.0, t_on, t_off, t_off + 0.1, t_final]
    vs = [0.0, peak, peak, 0.0, 0.0]
    return TimeTable(t=np.array(ts), v=np.array(vs))


def main(args):
    repo = Path(__file__).resolve().parents[1]
    cd = repo / args.case_dir
    # Mid-range forcing, deterministic -> distinct from the 8 random training draws.
    tt = _flux_forcing(peak=1.4e6, t_on=1.5, t_off=70.0, t_final=120.0)
    res = run(cd, verbose=False, lgas=True, record_every=args.record_every,
              write_con=False, time_table=tt)
    traj = res["trajectory"]
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **traj)
    dt = float(np.diff(traj["time"]).mean())
    print(f"held-out aw2 trajectory: {traj['time'].size} snapshots x "
          f"{int(traj['n_cells'])} cells, snapshot dt~{dt:.4f}s "
          f"(peak=1.4MW t_off=70s) -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--record-every", type=int, default=2000)
    p.add_argument("--out", default="heat_python/data/aw2_holdout_traj.npz")
    main(p.parse_args())
