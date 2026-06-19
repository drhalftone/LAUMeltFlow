"""Material properties. Replaces READ_KAPPA, READ_CP, CALC_K, CALC_K_MIX,
CALC_CP, CALC_CP_MIX, INTERP_TABLE, INTERP_ENERGY_TABLE, CALC_HS_FORM,
CALC_HS_MIX, CALC_EPS_RHO in 1Dheat.f (lines ~895-1300, 1461-1470).

Holds virgin/char tables for k(T) and cp(T) and computes mixed properties
as a linear blend by the char-progress variable tau:

    tau = 1   -> fully virgin   -> use *_virgin tables
    tau = 0   -> fully char     -> use *_char tables
    k_mix(T,tau)  = tau*k_virgin(T)  + (1-tau)*k_char(T)
    cp_mix(T,tau) = tau*cp_virgin(T) + (1-tau)*cp_char(T)

The cp tables also store a cumulative enthalpy h_s(T) = integral of cp dT
(if the file has 3 or 4 columns), or it's computed via trapezoidal
integration on load (if only 2 columns).

If LRAD is on and LEFF is off, we use the *solid* k tables (sol_kappa).
Otherwise we use the *effective* k tables (eff_kappa), which fold a
Rosseland-style radiation contribution into k.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .case import Case


@dataclass
class CpTable:
    """Heat-capacity table: T, cp(T), and cumulative h_s(T) = int(cp dT)."""
    T: np.ndarray       # (N,) temperatures [K]
    cp: np.ndarray      # (N,) heat capacities [J/kg/K]
    hs: np.ndarray      # (N,) cumulative enthalpy [J/kg]


@dataclass
class KTable:
    """Thermal-conductivity table: T, k(T)."""
    T: np.ndarray       # (N,) temperatures [K]
    k: np.ndarray       # (N,) conductivities [W/m/K]


@dataclass
class Materials:
    """All material property tables loaded for a run."""
    cp_virgin: CpTable
    cp_char: CpTable
    k_virgin: KTable
    k_char: KTable


def _load_table(path: str | Path) -> np.ndarray:
    """Load a whitespace-delimited table, treating '#' lines as comments."""
    return np.loadtxt(str(path), comments="#")


def load_cp_table(path: str | Path) -> CpTable:
    """Load a cp file. Accepts 2, 3, or 4 columns:
        col 1: T
        col 2: cp(T)
        col 3 or 4 (if present): h_sensible(T)  (cumulative integral of cp dT)
    If only 2 columns, h_sensible is computed via trapezoidal rule.
    """
    data = _load_table(path)
    if data.ndim != 2:
        # Single-row table -> reshape to (1, ncols)
        data = data.reshape(1, -1)

    T = data[:, 0].astype(np.float64)
    cp = data[:, 1].astype(np.float64)
    ncols = data.shape[1]
    if ncols >= 4:
        hs = data[:, 3].astype(np.float64)
    elif ncols == 3:
        hs = data[:, 2].astype(np.float64)
    else:
        # Compute cumulative integral via trapezoidal rule
        # h[0] = 0, h[j] = h[j-1] + 0.5*(cp[j] + cp[j-1])*(T[j] - T[j-1])
        hs = np.zeros_like(cp)
        hs[1:] = np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(T))
    return CpTable(T=T, cp=cp, hs=hs)


def load_k_table(path: str | Path) -> KTable:
    data = _load_table(path)
    if data.ndim != 2:
        data = data.reshape(1, -1)
    return KTable(T=data[:, 0].astype(np.float64),
                  k=data[:, 1].astype(np.float64))


def load_materials(case: Case, case_dir: str | Path,
                   lrad: bool = False, leff: bool = False) -> Materials:
    """Load all material property tables for this case.

    case_dir is the directory containing heat.case and the data files.
    The lrad/leff flags choose between solid and effective k tables, mirroring
    READ_KAPPA in 1Dheat.f.
    """
    d = Path(case_dir)

    # K tables: solid if (LRAD and not LEFF), effective otherwise.
    use_solid_k = lrad and not leff
    if use_solid_k:
        k_v_path = d / case.sol_k_file
        k_c_path = d / case.sol_k_char_file
    else:
        k_v_path = d / case.eff_k_file
        k_c_path = d / case.eff_k_char_file

    return Materials(
        cp_virgin=load_cp_table(d / case.cp_file),
        cp_char=load_cp_table(d / case.cp_char_file),
        k_virgin=load_k_table(k_v_path),
        k_char=load_k_table(k_c_path),
    )


# ----------------------- Interpolation helpers ----------------------- #


def interp_table(T: float, T_tab: np.ndarray, V_tab: np.ndarray) -> float:
    """Linear interpolation, clamped at boundaries. (INTERP_TABLE in Fortran.)

    For T below T_tab[0]: returns V_tab[0]. For T above T_tab[-1]: returns
    V_tab[-1]. Both array inputs assumed monotonically increasing.
    """
    return float(np.interp(T, T_tab, V_tab))


def interp_energy_table(T: float, T_tab: np.ndarray, cp_tab: np.ndarray,
                        E_tab: np.ndarray) -> float:
    """Sensible enthalpy at T, computed from cp(T) and the cumulative integral
    E_tab (where E_tab[i] = integral_T_tab[0]^T_tab[i] cp(T') dT').

    Within a segment [T_tab[i], T_tab[i+1]], uses the average cp over the
    sub-segment [T_tab[i], T]:

        E(T) = E_tab[i] + (T - T_tab[i]) * 0.5 * ((2-F)*cp_tab[i] + F*cp_tab[i+1])

    where F = (T - T_tab[i]) / (T_tab[i+1] - T_tab[i]).

    Clamps at boundaries: returns E_tab[0] for T < T_tab[0], E_tab[-1] for T > T_tab[-1].
    """
    if T <= T_tab[0]:
        return float(E_tab[0])
    if T >= T_tab[-1]:
        return float(E_tab[-1])
    # Find segment containing T
    i = int(np.searchsorted(T_tab, T)) - 1
    # searchsorted gives insertion index; -1 to get the left bracket.
    # Edge case: if T == T_tab[i+1] exactly, fine.
    dT_seg = T_tab[i + 1] - T_tab[i]
    F = (T - T_tab[i]) / max(dT_seg, 1.0e-30)
    return float(E_tab[i] + (T - T_tab[i]) * 0.5 *
                 ((2.0 - F) * cp_tab[i] + F * cp_tab[i + 1]))


# ----------------------- Mixing-rule lookups ----------------------- #


def calc_k_mix(T: float, tau: float, mats: Materials) -> float:
    """Mixed thermal conductivity: tau*k_virgin(T) + (1-tau)*k_char(T)."""
    k_v = interp_table(T, mats.k_virgin.T, mats.k_virgin.k)
    k_c = interp_table(T, mats.k_char.T, mats.k_char.k)
    return tau * k_v + (1.0 - tau) * k_c


def calc_cp_mix(T: float, tau: float, mats: Materials) -> float:
    """Mixed heat capacity: tau*cp_virgin(T) + (1-tau)*cp_char(T)."""
    cp_v = interp_table(T, mats.cp_virgin.T, mats.cp_virgin.cp)
    cp_c = interp_table(T, mats.cp_char.T, mats.cp_char.cp)
    return tau * cp_v + (1.0 - tau) * cp_c


def calc_hs_form(tau: float, case: Case, mats: Materials) -> float:
    """Formation enthalpy at the reference temperature. Used to align the
    energy origin so h_s(T_ref) = 0 inside CALC_HS_MIX.

    T_ref = case.init_temp by default, or case.hgas_tref if positive.
    """
    T_ref = case.init_temp
    if case.hgas_tref > 0.0:
        T_ref = case.hgas_tref

    e_v = interp_energy_table(T_ref, mats.cp_virgin.T,
                              mats.cp_virgin.cp, mats.cp_virgin.hs)
    e_c = interp_energy_table(T_ref, mats.cp_char.T,
                              mats.cp_char.cp, mats.cp_char.hs)
    return tau * e_v + (1.0 - tau) * e_c


def calc_hs_mix(T: float, tau: float, case: Case, mats: Materials) -> float:
    """Sensible enthalpy at T, relative to the reference temperature.

    h_s_mix(T, tau) = tau*h_s_virgin(T) + (1-tau)*h_s_char(T) - h_s_form(tau)
    """
    e_v = interp_energy_table(T, mats.cp_virgin.T,
                              mats.cp_virgin.cp, mats.cp_virgin.hs)
    e_c = interp_energy_table(T, mats.cp_char.T,
                              mats.cp_char.cp, mats.cp_char.hs)
    return tau * e_v + (1.0 - tau) * e_c - calc_hs_form(tau, case, mats)


def calc_eps_rho(tau: float, case: Case) -> float:
    """Emissivity blended by tau: tau*eps_v + (1-tau)*eps_c.

    Note: the Fortran's CALC_EPS_RHO takes density rho and internally
    computes tau. Here we take tau directly to keep modules independent
    (tau lives in pyrolysis.py).
    """
    return tau * case.eps_v + (1.0 - tau) * case.eps_c


if __name__ == "__main__":
    # Sanity check: load aw1 materials, spot-check a few interpolated values
    from pathlib import Path
    from .case import load_case

    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / "heat_2026-04-11_1837" / "examples" / "aw1"
    case = load_case(case_dir / "heat.case")
    mats = load_materials(case, case_dir, lrad=False, leff=False)

    print(f"aw1 materials loaded:")
    print(f"  cp_virgin: {len(mats.cp_virgin.T)} entries, "
          f"T range [{mats.cp_virgin.T[0]:.1f}, {mats.cp_virgin.T[-1]:.1f}] K")
    print(f"  cp_char:   {len(mats.cp_char.T)} entries, "
          f"T range [{mats.cp_char.T[0]:.1f}, {mats.cp_char.T[-1]:.1f}] K")
    print(f"  k_virgin:  {len(mats.k_virgin.T)} entries, "
          f"T range [{mats.k_virgin.T[0]:.1f}, {mats.k_virgin.T[-1]:.1f}] K")
    print(f"  k_char:    {len(mats.k_char.T)} entries, "
          f"T range [{mats.k_char.T[0]:.1f}, {mats.k_char.T[-1]:.1f}] K")

    print(f"\nInterpolated values at T=600 K:")
    print(f"  cp_virgin(600) = {interp_table(600.0, mats.cp_virgin.T, mats.cp_virgin.cp):.2f} J/kg/K")
    print(f"  cp_char(600)   = {interp_table(600.0, mats.cp_char.T, mats.cp_char.cp):.2f} J/kg/K")
    print(f"  k_virgin(600)  = {interp_table(600.0, mats.k_virgin.T, mats.k_virgin.k):.4f} W/m/K")
    print(f"  k_char(600)    = {interp_table(600.0, mats.k_char.T, mats.k_char.k):.4f} W/m/K")

    print(f"\nMixing rules at T=600, tau=0.5 (half-charred):")
    print(f"  calc_cp_mix  = {calc_cp_mix(600.0, 0.5, mats):.2f}")
    print(f"  calc_k_mix   = {calc_k_mix(600.0, 0.5, mats):.4f}")
    print(f"  calc_hs_mix  = {calc_hs_mix(600.0, 0.5, case, mats):.2f} J/kg "
          f"(should equal 0 at T=init_temp={case.init_temp})")
    print(f"  calc_eps_rho = {calc_eps_rho(0.5, case):.3f}")

    print(f"\nVerification: hs_mix at the reference temperature ({case.init_temp} K) should be ~0:")
    print(f"  calc_hs_mix({case.init_temp}, 0.5) = "
          f"{calc_hs_mix(case.init_temp, 0.5, case, mats):.2e} J/kg")
