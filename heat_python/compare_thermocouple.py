"""Compare a Python con.out against a Fortran reference thermocouple.txt.

The reference file has columns:  t, surface, tc1..tc7, bottom  (the 9 probe
locations in tc_locations.dat, ordered surface->bottom). Our con.out row is
[time, t_surf, tp1..tp9, t_bot, r_surf, rp1..rp9, r_bot], so the 9 reference
probe columns map onto con.out columns 2..10 (tp1..tp9).

The two files sample at different cadences, so we linearly interpolate the
Python probe traces onto the reference time grid before differencing.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

PROBE_NAMES = ["surface", "tc1", "tc2", "tc3", "tc4", "tc5", "tc6", "tc7",
               "bottom"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--python", required=True)
    args = p.parse_args()

    ref = np.loadtxt(args.reference, skiprows=1)
    py = np.loadtxt(args.python)

    t_ref = ref[:, 0]
    t_py = py[:, 0]
    # Restrict to the overlapping time window to avoid extrapolation.
    lo, hi = max(t_ref[0], t_py[0]), min(t_ref[-1], t_py[-1])
    mask = (t_ref >= lo) & (t_ref <= hi)
    t_cmp = t_ref[mask]

    print(f"reference rows={ref.shape[0]}, python rows={py.shape[0]}")
    print(f"comparing {t_cmp.size} reference samples in t=[{lo:.3g}, {hi:.3g}]")
    print(f"\n{'probe':<9}{'max_abs':>13}{'mean_abs':>13}{'max_rel':>11}"
          f"{'@t':>9}")
    print("-" * 64)
    worst = 0.0
    for j, name in enumerate(PROBE_NAMES):
        ref_col = ref[mask, 1 + j]
        py_col = np.interp(t_cmp, t_py, py[:, 2 + j])
        abs_err = np.abs(py_col - ref_col)
        rel = abs_err / np.maximum(np.abs(ref_col), 1.0)
        i_max = int(np.argmax(abs_err))
        worst = max(worst, abs_err.max())
        print(f"{name:<9}{abs_err.max():>13.4e}{abs_err.mean():>13.4e}"
              f"{rel.max():>11.2e}{t_cmp[i_max]:>9.2f}")
    print("-" * 64)
    print(f"worst max_abs across all probes: {worst:.4e} K")


if __name__ == "__main__":
    main()
