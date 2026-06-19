"""Aerothermal boundary data. Replaces READ_AERO, READ_BPRIME_HW, and the
aero helper functions (CALC_AERO_CH/HR/P/QR/LAM/TEXT, CALC_BLOWCORR,
CALC_WALL_HW) in 1Dheat.f. Used by the type-2 aero boundary in bcs.py.

The aero table (BLinputFile.txt) is the time history of the boundary-layer
edge state:
    time, rho_e*u_e*C_H, h_r, p_w, q_rad, lambda, T_ext
The B' surface-chemistry tables give the wall enthalpy h_w as a function of
(wall temperature, dimensionless blowing rate B'_g). The directory holds one
file per B'_g breakpoint listed in helpbPrime.3dth.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from .case import Case
from .domain import Domain
from .gas import GasTable, calc_hgas_abs


@dataclass
class AeroTable:
    """Time histories of the aerothermal edge state (BLinputFile.txt)."""
    t: np.ndarray        # (N,) time [s]
    ch: np.ndarray       # (N,) rho_e u_e C_H  [kg/m^2/s]
    hr: np.ndarray       # (N,) recovery enthalpy [J/kg]
    p: np.ndarray        # (N,) wall pressure [Pa]
    qr: np.ndarray       # (N,) radiative flux [W/m^2]
    lam: np.ndarray      # (N,) blowing-correction lambda [-]
    text: np.ndarray     # (N,) external (re-radiation) temperature [K]


@dataclass
class BPrimeTable:
    """B' surface-chemistry wall-enthalpy table h_w(T_wall, B'_g)."""
    bg: np.ndarray       # (n_bg,) blowing-rate breakpoints
    t: np.ndarray        # (n_t,) wall temperatures [K]
    hw: np.ndarray       # (n_t, n_bg) wall enthalpy [J/kg]


def load_aero_table(path: str | Path) -> AeroTable | None:
    """Parse BLinputFile.txt (comma- or whitespace-delimited, one header
    line). Returns None if absent."""
    p = Path(path)
    if str(path).strip() == "" or not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.replace(",", " ").strip()
        if not line:
            continue
        parts = line.split()
        try:
            vals = [float(x) for x in parts[:7]]
        except ValueError:
            continue  # header / non-numeric line
        if len(vals) >= 7:
            rows.append(vals)
    if not rows:
        return None
    arr = np.array(rows)
    return AeroTable(t=arr[:, 0], ch=arr[:, 1], hr=arr[:, 2], p=arr[:, 3],
                     qr=arr[:, 4], lam=arr[:, 5], text=arr[:, 6])


def load_bprime(bprime_dir: str | Path) -> BPrimeTable | None:
    """Load the B' wall-enthalpy tables from <dir>/helpbPrime.3dth (the
    list of B'_g breakpoints) plus one <dir>/p=101325/Bg=<x.xxx>.txt per
    breakpoint. Each per-Bg file is (T, dummy, h_w). Returns None if absent.
    """
    d = Path(bprime_dir)
    if str(bprime_dir).strip() == "":
        return None
    header = d / "helpbPrime.3dth"
    if not header.exists():
        return None
    bg = np.array([float(x) for x in header.read_text().split()])
    if bg.size == 0:
        return None

    t_ref = None
    hw_cols = []
    for b in bg:
        # Fortran writes the name with F6.3 then strips spaces -> e.g. 0.001
        fname = d / "p=101325" / f"Bg={b:.3f}.txt"
        data = np.loadtxt(str(fname), comments=("#", "!"))
        if data.ndim != 2:
            data = data.reshape(1, -1)
        if t_ref is None:
            t_ref = data[:, 0]
        hw_cols.append(data[:, 2])
    return BPrimeTable(bg=bg, t=t_ref, hw=np.column_stack(hw_cols))


# ----------------------- Scalar lookups (called once per step) ----------- #


def _interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    return float(np.interp(x, xp, fp))


def calc_aero_ch(time: float, aero: AeroTable | None, case: Case) -> float:
    if aero is None:
        return case.qan
    return _interp(time, aero.t, aero.ch)


def calc_aero_hr(time: float, aero: AeroTable | None) -> float:
    return 0.0 if aero is None else _interp(time, aero.t, aero.hr)


def calc_aero_p(time: float, aero: AeroTable | None, case: Case) -> float:
    if aero is None:
        return case.pamb
    return _interp(time, aero.t, aero.p)


def calc_aero_qr(time: float, aero: AeroTable | None) -> float:
    return 0.0 if aero is None else _interp(time, aero.t, aero.qr)


def calc_aero_lam(time: float, aero: AeroTable | None) -> float:
    return 0.0 if aero is None else _interp(time, aero.t, aero.lam)


def calc_aero_text(time: float, aero: AeroTable | None, case: Case) -> float:
    if aero is None:
        return case.init_temp
    return _interp(time, aero.t, aero.text)


def calc_blowcorr(time: float, mdotface: float, aero: AeroTable | None,
                  domain: Domain, case: Case) -> float:
    """Blowing (transpiration) correction to the heat-transfer coefficient."""
    n = domain.n_cells
    ch0 = max(calc_aero_ch(time, aero, case), 1.0e-30)
    lam = calc_aero_lam(time, aero)
    mdot = max(0.0, mdotface / max(domain.da[n + 1], 1.0e-30))
    phi = 2.0 * lam * mdot / ch0
    if abs(phi) < 1.0e-7:
        return 1.0 - 0.5 * phi + phi * phi / 12.0
    elif phi <= 20.0:
        return phi / (math.exp(phi) - 1.0)
    return 1.0e-8


def calc_wall_hw(time: float, twall: float, mdotface: float,
                 aero: AeroTable | None, bprime: BPrimeTable | None,
                 domain: Domain, case: Case, gas: GasTable | None) -> float:
    """Wall enthalpy h_w from the B' tables, bilinear in (T_wall, B'_g).
    Falls back to the absolute gas enthalpy when no B' table is loaded."""
    if bprime is None or bprime.bg.size == 0 or bprime.t.size == 0:
        return float(calc_hgas_abs(twall, gas, case))

    n = domain.n_cells
    ch0 = max(calc_aero_ch(time, aero, case), 1.0e-30)
    if ch0 <= 1.0e-12:
        return float(calc_hgas_abs(twall, gas, case))
    corr = max(calc_blowcorr(time, mdotface, aero, domain, case), 1.0e-8)
    bg = max(0.0, mdotface / max(domain.da[n + 1], 1.0e-30) / (ch0 * corr))

    tbg = bprime.bg
    if bg <= tbg[0]:
        return _interp(twall, bprime.t, bprime.hw[:, 0])
    if bg >= tbg[-1]:
        return _interp(twall, bprime.t, bprime.hw[:, -1])
    for i in range(tbg.size - 1):
        if bg <= tbg[i + 1]:
            v0 = _interp(twall, bprime.t, bprime.hw[:, i])
            v1 = _interp(twall, bprime.t, bprime.hw[:, i + 1])
            w = (bg - tbg[i]) / max(tbg[i + 1] - tbg[i], 1.0e-30)
            return v0 + (v1 - v0) * w
    return _interp(twall, bprime.t, bprime.hw[:, -1])
