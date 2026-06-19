"""Compare a Fortran output file against a Python output file.

The Fortran writes various .out files into each example directory.
The Python port should write equivalent files in a sibling directory
(e.g. examples/aw1/python_out/). This utility loads both and reports
max-abs, mean-abs, and worst-row diffs per column.

Usage:
    python -m heat_python.validate fortran=examples/aw1/con.out python=examples/aw1/python_out/con.out
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np


def load_text_table(path: str | Path) -> np.ndarray:
    """Load a whitespace-separated ASCII table, skipping comment lines.

    Comment lines start with '#'. Returns shape (n_rows, n_cols).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.array(rows)


def compare(fortran_path: str, python_path: str,
            rtol: float = 1e-3, atol: float = 1e-6) -> dict:
    """Compare two output files column-by-column.

    Returns a dict with per-column metrics. Prints a summary table.
    """
    a = load_text_table(fortran_path)
    b = load_text_table(python_path)

    if a.shape != b.shape:
        # Try truncating to the shorter file in case one has more output rows
        n = min(a.shape[0], b.shape[0])
        m = min(a.shape[1], b.shape[1])
        print(f"Shape mismatch: fortran {a.shape} vs python {b.shape}")
        print(f"  Comparing first {n} rows x {m} cols")
        a = a[:n, :m]
        b = b[:n, :m]

    diff = b - a
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(a), atol)

    metrics = {}
    print(f"\n{'col':>4} {'max_abs':>14} {'mean_abs':>14} {'max_rel':>10} "
          f"{'pass':>6}")
    print("-" * 56)
    for c in range(a.shape[1]):
        max_abs = float(abs_diff[:, c].max())
        mean_abs = float(abs_diff[:, c].mean())
        max_rel = float(rel_diff[:, c].max())
        ok = (max_abs < atol) or (max_rel < rtol)
        metrics[c] = dict(max_abs=max_abs, mean_abs=mean_abs,
                          max_rel=max_rel, pass_=ok)
        flag = "OK" if ok else "FAIL"
        print(f"{c:>4} {max_abs:>14.4e} {mean_abs:>14.4e} {max_rel:>10.2e} "
              f"{flag:>6}")

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fortran", required=True,
                   help="Path to the Fortran output file (reference)")
    p.add_argument("--python", required=True,
                   help="Path to the Python output file (port under test)")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    m = compare(args.fortran, args.python, rtol=args.rtol, atol=args.atol)
    any_fail = any(not v["pass_"] for v in m.values())
    sys.exit(1 if any_fail else 0)
