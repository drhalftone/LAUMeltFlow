"""Training-data generation for the heat-shield GNN surrogate.

Stage 1 (this module): run the validated solver with full-field recording and
save the raw *trajectory* — the complete per-cell state over time — as a
target-agnostic .npz. From one trajectory you can later build either flux
training pairs or state training pairs, at any super-step gap, without
re-running the (slow) solver.

A trajectory .npz holds (S = number of recorded snapshots, m = n_cells+2
including the two ghost cells that carry the boundary forcing):
    time   (S,)              snapshot times [s]
    T      (S, m)            temperature, with ghosts
    rho    (S, m)            bulk density, with ghosts
    rho_i  (S, n_species, m) per-species density, with ghosts
    pg     (S, m)            gas pressure          [gas runs only]
    mdotf  (S, m)            face mass flux        [gas runs only]
    x, dx, da, dv  (m,)      geometry (constant in time)
    + scalars: n_cells, n_species, dt_nom, lgas, record_every,
      rhov_bulk, rhoc_bulk, init_temp, case_name

Usage:
    python -m heat_python.data_gen --case-dir .../aw1 --record-every 5 \
        --out heat_python/data/aw1_traj.npz
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .solver import run


def generate_trajectory(case_dir: str | Path, out_path: str | Path,
                        record_every: int = 5, lgas: bool | None = None,
                        verbose: bool = True) -> dict:
    """Run one case with full-field recording and save the trajectory .npz.

    Returns the trajectory dict (also written to out_path)."""
    result = run(case_dir, verbose=verbose, lgas=lgas,
                 record_every=record_every, write_con=False)
    traj = result["trajectory"]
    if traj is None:
        raise RuntimeError("no trajectory recorded (record_every not honored)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **traj)

    if verbose:
        S = traj["time"].size
        print(f"  trajectory: {S} snapshots x {traj['n_cells']} cells "
              f"(every {record_every} steps) -> {out_path}")
        print(f"  snapshot dt ~ {np.diff(traj['time']).mean():.4e} s "
              f"(= {record_every} solver steps)")
    return traj


def load_trajectory(path: str | Path) -> dict:
    """Load a trajectory .npz back into a plain dict of arrays/scalars."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--record-every", type=int, default=5)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gas", dest="lgas", action="store_true", default=None)
    g.add_argument("--no-gas", dest="lgas", action="store_false")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cd = Path(args.case_dir)
    if not cd.is_absolute():
        cd = repo / cd
    generate_trajectory(cd, args.out, record_every=args.record_every,
                        lgas=args.lgas)
