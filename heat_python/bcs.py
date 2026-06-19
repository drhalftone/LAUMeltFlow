"""Boundary conditions. Replaces the BC blocks in PROGRAM HEAT
(1Dheat.f lines ~374-449, 636-662).

Sets ghost-cell temperatures T[0] and T[N+1] each timestep based on the
case's BC types:

    bnd_type = 1 - fixed wall temperature (Dirichlet)
                   ghost = 2*T_wall - T_adjacent
    bnd_type = 2 - fixed heat flux (Neumann)
                   ghost set so that  k * dT/dx = q_w  at the face

For the simplest path (phase 3, aw1-style cases), bnd_type 2 supports:
  - constant flux (case.qa0 / case.qan)
  - time-varying flux (via a TimeTable)
  - emissivity-corrected: q_w = q_applied * (1 - eps) - BBR (radiative)

The full aerothermal BC (CASE_USE_AERO_BCN: B' chemistry, blowing
correction, wall enthalpy) is deferred to phase 4.

Sign convention: DIR0 = -1, DIRN = +1 in the Fortran, so the ghost
update is:
    T[0]   = T[1]   + q_w0 / k * dx[1]   * DIR0  =  T[1]  - q_w0/k * dx
    T[N+1] = T[N]   + q_wn / k * dx[N]   * DIRN  =  T[N]  + q_wn/k * dx
"""

from __future__ import annotations
import math

import numpy as np

from .case import Case
from .domain import Domain
from .materials import Materials, calc_k_mix
from .pyrolysis import SolidParams, calc_tau, calc_eps_rho
from .io_files import TimeTable
from .gas import GasTable, calc_hgas_abs
from .aero import (
    AeroTable, BPrimeTable, calc_aero_ch, calc_aero_hr, calc_aero_qr,
    calc_aero_text, calc_blowcorr, calc_wall_hw,
)


SIGMA = 5.670374419e-8  # Stefan-Boltzmann [W/m^2/K^4]


def apply_left_bc(T: np.ndarray, rho: np.ndarray, time: float,
                  case: Case, domain: Domain, mats: Materials,
                  solid: SolidParams, time_tbl: TimeTable | None,
                  init_temp: float) -> None:
    """Set the left ghost cell T[0] in place, given the case's bnd0 type.

    Mirrors the left-boundary block in 1Dheat.f around lines 396-409.
    """
    n = domain.n_cells
    rhov_b = solid.rhov_bulk
    rhoc_b = solid.rhoc_bulk

    if case.bnd0 == 1:
        # Fixed wall temperature (Dirichlet)
        t_wall = case.tw0
        if case.use_time_bcn and time_tbl is not None:
            t_wall = time_tbl.at(time)
        T[0] = 2.0 * t_wall - T[1]

    elif case.bnd0 == 2:
        # Fixed heat flux (Neumann)
        qa0 = case.qa0
        if case.use_time_bcn and time_tbl is not None:
            qa0 = time_tbl.at(time)

        rho_bound = 0.5 * (rho[0] + rho[1])
        tau_bound = calc_tau(rho_bound, rhov_b, rhoc_b, case, solid)
        eps = tau_bound * case.eps_v + (1.0 - tau_bound) * case.eps_c
        ttt = 0.5 * (T[0] + T[1])
        # Fortran's BBR is commented out for the left side (set to 0).
        # The radiative emission term is included only on the right side
        # (or via the LRAD radiation module).
        bbr = 0.0
        qw0 = qa0 * (1.0 - eps) - bbr
        k_bound = calc_k_mix(ttt, tau_bound, mats)
        # DIR0 = -1, so:  T[0] = T[1] + qw0/k * dx[1] * (-1)
        T[0] = T[1] - qw0 / max(k_bound, 1.0e-30) * domain.dx[1]

    else:
        raise ValueError(f"Unknown left BC type {case.bnd0}")


def apply_right_bc(T: np.ndarray, rho: np.ndarray, time: float,
                   case: Case, domain: Domain, mats: Materials,
                   solid: SolidParams, time_tbl: TimeTable | None,
                   init_temp: float, lrad: bool = False,
                   aero: AeroTable | None = None,
                   bprime: BPrimeTable | None = None,
                   gas: GasTable | None = None,
                   mdotf: np.ndarray | None = None,
                   use_aero: bool = False) -> None:
    """Set the right ghost cell T[N+1] in place, given the case's bndn type.

    Mirrors the right-boundary block in 1Dheat.f around lines 411-443,
    including the aerothermal branch (CASE_USE_AERO_BCN): convective heating
    with blowing correction, B' wall enthalpy, pyrolysis-gas injection, and
    surface re-radiation.
    """
    n = domain.n_cells

    if case.bndn == 1:
        # Fixed wall temperature (Dirichlet)
        if case.use_time_bcn and time_tbl is not None:
            t_wall = time_tbl.at(time)
        else:
            t_wall = case.twn
        T[n + 1] = 2.0 * t_wall - T[n]

    elif case.bndn == 2:
        rho_bound = 0.5 * (rho[n + 1] + rho[n])
        tau_bound = calc_tau(rho_bound, solid.rhov_bulk, solid.rhoc_bulk,
                             case, solid)
        eps = tau_bound * case.eps_v + (1.0 - tau_bound) * case.eps_c

        if use_aero:
            # Aerothermal heat flux (1Dheat.f lines 422-442).
            mdot_face = float(mdotf[n + 1]) if mdotf is not None else 0.0
            twall = T[n]
            ch = calc_aero_ch(time, aero, case)
            corr = calc_blowcorr(time, mdot_face, aero, domain, case)
            hw = calc_wall_hw(time, twall, mdot_face, aero, bprime,
                              domain, case, gas)
            text = calc_aero_text(time, aero, case)
            qaero = ch * corr * (calc_aero_hr(time, aero) - hw)
            qpyro = (mdot_face / max(domain.da[n + 1], 1.0e-30)
                     * (float(calc_hgas_abs(twall, gas, case)) - hw))
            qwn = qaero + calc_aero_qr(time, aero) + qpyro
            qwn -= eps * SIGMA * (twall ** 4 - text ** 4)
            ttt = twall
        else:
            # Fixed heat flux (Neumann) with radiative emission to ambient.
            qan = case.qan
            if case.use_time_bcn and time_tbl is not None:
                qan = time_tbl.at(time)
            if lrad:
                qan = 0.0
            ttt = 0.5 * (T[n + 1] + T[n])
            qwn = qan * (1.0 - eps) - eps * SIGMA * (ttt ** 4 - init_temp ** 4)

        k_bound = calc_k_mix(ttt, tau_bound, mats)
        # DIRN = +1, so:  T[N+1] = T[N] + qwn/k * dx[N]
        T[n + 1] = T[n] + qwn / max(k_bound, 1.0e-30) * domain.dx[n]

    else:
        raise ValueError(f"Unknown right BC type {case.bndn}")


def apply_bcs(T: np.ndarray, rho: np.ndarray, time: float,
              case: Case, domain: Domain, mats: Materials,
              solid: SolidParams,
              time_tbl_left: TimeTable | None,
              time_tbl_right: TimeTable | None,
              init_temp: float, lrad: bool = False,
              aero: AeroTable | None = None,
              bprime: BPrimeTable | None = None,
              gas: GasTable | None = None,
              mdotf: np.ndarray | None = None,
              use_aero: bool = False) -> None:
    """Apply both boundary conditions in place on T.

    The same time table is typically used for both ends; pass None to the
    side that doesn't use it. The aero/bprime/gas/mdotf args feed the
    right-boundary aerothermal branch (CASE_USE_AERO_BCN).
    """
    apply_left_bc(T, rho, time, case, domain, mats, solid,
                  time_tbl_left, init_temp)
    apply_right_bc(T, rho, time, case, domain, mats, solid,
                   time_tbl_right, init_temp, lrad=lrad,
                   aero=aero, bprime=bprime, gas=gas, mdotf=mdotf,
                   use_aero=use_aero)


if __name__ == "__main__":
    # Sanity check: apply BCs on a uniform-T mesh for aw1
    from pathlib import Path
    from .case import load_case
    from .domain import setup_domain
    from .materials import load_materials
    from .pyrolysis import load_solid
    from .io_files import load_time_table

    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / "heat_2026-04-11_1837" / "examples" / "aw1"
    case = load_case(case_dir / "heat.case")
    domain = setup_domain(case)
    mats = load_materials(case, case_dir, lrad=False, leff=False)
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    time_tbl = load_time_table(case_dir / case.time_file)

    n = domain.n_cells
    T = np.full(n + 2, case.init_temp)
    rho = np.full(n + 2, solid.rhov_bulk)

    print(f"aw1 BCs (bnd0={case.bnd0}, bndn={case.bndn})")
    print(f"  time table range: {time_tbl.t[0]} to {time_tbl.t[-1]} s,"
          f" values {time_tbl.v[0]} to {time_tbl.v[-1]}")
    print(f"  before BC:  T[0]={T[0]:.2f}  T[N+1]={T[n+1]:.2f}")

    for t in [0.0, 0.5, 1.0, 30.0, 60.0]:
        T = np.full(n + 2, case.init_temp)
        apply_bcs(T, rho, t, case, domain, mats, solid,
                  None, time_tbl, case.init_temp)
        print(f"  t={t:6.2f}s  ->  T[0]={T[0]:.2f}  T[N+1]={T[n+1]:.2f}"
              f"  (interp wall T={time_tbl.at(t):.2f})")
