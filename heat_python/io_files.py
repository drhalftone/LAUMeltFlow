"""File readers for material/BC data files. Replaces READ_TEMP,
READ_TC_LOCATIONS, READ_GAS, READ_AERO, READ_BPRIME_HW, SKIP_DATA_HEADERS
in 1Dheat.f.

All input files are plain ASCII. Comment lines start with '#'. Numeric
rows are whitespace-separated.

The cp/k tables for materials live in materials.py.
The solid.dat parser for pyrolysis lives in pyrolysis.py.
This module covers the time-series BC files and thermocouple locations.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TimeTable:
    """Time-series boundary condition (time_*.dat). Each row is `t value`.

    Used for both time-varying wall temperature and time-varying heat flux,
    depending on which BC type the case selects.
    """
    t: np.ndarray             # (N,) times [s]
    v: np.ndarray             # (N,) value at time t (T_wall [K] or flux [W/m^2])

    def at(self, time: float) -> float:
        """Linear interpolation, clamped at the endpoints."""
        return float(np.interp(time, self.t, self.v))


def load_time_table(path: str | Path) -> TimeTable:
    """Load a 2-column time-series file (t, value)."""
    data = np.loadtxt(str(path), comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2 columns in {path}, got shape {data.shape}")
    return TimeTable(t=data[:, 0].astype(np.float64),
                     v=data[:, 1].astype(np.float64))


def load_tc_locations(path: str | Path, l0: float, n_tc: int) -> np.ndarray:
    """Load thermocouple x-positions (from tc_locations.dat).

    The file format is a list of *depths from the right wall* (1 per line,
    possibly with comments). The Fortran's READ_TC_LOCATIONS converts each
    depth to an x-coordinate via  XTC = L0 - depth, so TC1 = "1mm" means
    "1mm from the right wall" = x = L0 - 1mm. We do the same.

    Values may be in Fortran double notation ('1.0D-3') -- we convert those
    to E notation before parsing.

    Returns an array of x-positions (length n_tc), padded or truncated.
    """
    text = Path(path).read_text()
    # Convert Fortran 1.0D-3 -> 1.0e-3 (digit-D-sign-digit pattern).
    import re
    text = re.sub(r"(\d)[dD]([+-]?\d)", r"\1e\2", text)
    lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    depths = np.array([float(ln) for ln in lines if ln])
    x_pos = l0 - depths   # convert depth-from-right to x-from-left
    out = x_pos[:n_tc] if x_pos.size >= n_tc else np.pad(x_pos, (0, n_tc - x_pos.size))
    return out
