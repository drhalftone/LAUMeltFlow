"""Main time-step loop. Replaces the body of PROGRAM HEAT
(1Dheat.f lines ~234-715, minus the LGAS and LRAD branches which are
deferred to phase 4).

Phase 3 scope: heat conduction + pyrolysis only. No gas flow, no
radiation. Validates against aw1's con.out reference output.

Per timestep:
    1. Pick dt (CFL-based, clipped to land on BC event times)
    2. Set ghost cells (apply BCs)
    3. Pyrolysis species update -> RHON, RHOI_NEW
    4. Compute face conductivities k_face[i]
    5. Cell loop, vectorized:
        - face fluxes      FLUXM, FLUXP
        - old energy       EOLD  = rho * hs_mix(T, tau) * dv
        - chemical heat    ECHEM = (rho_new*hs_form(tau_new) - rho*hs_form(tau)) * dv
        - new energy       ENEW  = EOLD + dt*(net_flux) + ECHEM
    6. 3-iteration Newton to back out T_new from
           rho_new * hs_mix(T_new, tau_new) * dv = ENEW
    7. Advance: T <- T_new, rho <- rho_new
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time as _time

import numpy as np

from .case import Case, load_case
from .domain import Domain, setup_domain
from .materials import (
    Materials, load_materials,
    calc_k_mix, calc_cp_mix, calc_hs_mix, calc_hs_form,
)
from .pyrolysis import (
    SolidParams, load_solid,
    calc_tau, calc_porosity, step_species,
)
from .bcs import apply_bcs
from .io_files import TimeTable, load_time_table, load_tc_locations
from .gas import (
    GasTable, load_gas_table, solve_gas, _porosity_array,
    calc_rhogp, calc_cvg, calc_ugas, calc_hgas,
)
from .aero import AeroTable, BPrimeTable, load_aero_table, load_bprime, calc_aero_p


# ----------------------- State container ----------------------- #


@dataclass
class State:
    """All time-varying arrays for one snapshot of the simulation."""
    time: float
    T: np.ndarray        # (n+2,) temperature, includes ghost cells
    rho: np.ndarray      # (n+2,) bulk density
    rho_i: np.ndarray    # (n_species, n+2) per-species densities
    pg: np.ndarray | None = None       # (n+2,) gas pressure       [gas runs]
    mdotf: np.ndarray | None = None    # (n+2,) face mass flux     [gas runs]
    rhogsto: np.ndarray | None = None  # (n,) stored gas density   [gas runs]


# ----------------------- Helpers ----------------------- #


# These mirror the scalar calc_* in materials.py / pyrolysis.py but operate
# on whole arrays. np.interp matches the Fortran clamped linear lookup; the
# energy-table form matches INTERP_ENERGY_TABLE segment by segment.


def _interp_energy_array(T: np.ndarray, T_tab: np.ndarray, cp_tab: np.ndarray,
                         E_tab: np.ndarray) -> np.ndarray:
    """Vectorized INTERP_ENERGY_TABLE (see materials.interp_energy_table)."""
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    below = T <= T_tab[0]
    above = T >= T_tab[-1]
    mid = ~(below | above)
    out[below] = E_tab[0]
    out[above] = E_tab[-1]
    if np.any(mid):
        Tm = T[mid]
        idx = np.clip(np.searchsorted(T_tab, Tm) - 1, 0, T_tab.size - 2)
        dT = T_tab[idx + 1] - T_tab[idx]
        F = (Tm - T_tab[idx]) / np.maximum(dT, 1.0e-30)
        out[mid] = (E_tab[idx] + (Tm - T_tab[idx]) * 0.5
                    * ((2.0 - F) * cp_tab[idx] + F * cp_tab[idx + 1]))
    return out


def _tau_array(rho: np.ndarray, rhov_b: float, rhoc_b: float,
               case: Case, solid: SolidParams) -> np.ndarray:
    """Vectorized CALC_TAU across a 1D density array."""
    inert = np.abs(solid.rho_v - solid.rho_c) < 1.0e-12
    sum_inert = float(np.sum(solid.rho_v[inert]))
    sum_react = float(np.sum(solid.rho_v[~inert]))
    total = sum_inert + sum_react
    rho_inert = rhov_b * sum_inert / total if total > 1.0e-30 else 0.0

    rho_vr = max(rhov_b - rho_inert, 1.0e-30)
    rho_cr = max(rhoc_b - rho_inert, 0.0)
    rho_r = np.maximum(rho - rho_inert, 0.0)

    if case.tau_linear:
        den = max(rho_vr - rho_cr, 1.0e-30)
        tau = (rho_r - rho_cr) / den
    else:
        den = 1.0 - rho_cr / max(rho_vr, 1.0e-30)
        if den <= 1.0e-30:
            tau = np.zeros_like(rho_r)
        else:
            tau = (1.0 - rho_cr / np.maximum(rho_r, 1.0e-30)) / den
    return np.clip(tau, 0.0, 1.0)


def _k_mix_array(T: np.ndarray, tau: np.ndarray,
                 mats: Materials) -> np.ndarray:
    kv = np.interp(T, mats.k_virgin.T, mats.k_virgin.k)
    kc = np.interp(T, mats.k_char.T, mats.k_char.k)
    return tau * kv + (1.0 - tau) * kc


def _cp_mix_array(T: np.ndarray, tau: np.ndarray,
                  mats: Materials) -> np.ndarray:
    cpv = np.interp(T, mats.cp_virgin.T, mats.cp_virgin.cp)
    cpc = np.interp(T, mats.cp_char.T, mats.cp_char.cp)
    return tau * cpv + (1.0 - tau) * cpc


def _hs_form_array(tau: np.ndarray, case: Case,
                   mats: Materials) -> np.ndarray:
    """tau*e_v(Tref) + (1-tau)*e_c(Tref); the two e_* are scalar constants."""
    e_v = calc_hs_form(1.0, case, mats)   # tau=1 -> pure virgin reference
    e_c = calc_hs_form(0.0, case, mats)   # tau=0 -> pure char reference
    return tau * e_v + (1.0 - tau) * e_c


def _hs_mix_array(T: np.ndarray, tau: np.ndarray, case: Case,
                  mats: Materials) -> np.ndarray:
    e_v = _interp_energy_array(T, mats.cp_virgin.T, mats.cp_virgin.cp,
                               mats.cp_virgin.hs)
    e_c = _interp_energy_array(T, mats.cp_char.T, mats.cp_char.cp,
                               mats.cp_char.hs)
    return tau * e_v + (1.0 - tau) * e_c - _hs_form_array(tau, case, mats)


def compute_initial_dt(case: Case, domain: Domain, mats: Materials,
                       solid: SolidParams) -> float:
    """CFL-based initial timestep, matching the Fortran formula
    (1Dheat.f lines 258-263).

        dt = 0.5 * max_dx^2 * init_rho * cp_mix(600, tau_init) /
                 k_mix(600, tau_init) / CFL
    """
    init_rho = solid.rhov_bulk  # full virgin
    tau_init = calc_tau(init_rho, solid.rhov_bulk, solid.rhoc_bulk,
                        case, solid)
    cp = calc_cp_mix(600.0, tau_init, mats)
    k = calc_k_mix(600.0, tau_init, mats)
    return 0.5 * domain.max_dx ** 2 * init_rho * cp / k / case.cfl


def setup_thermocouples(case: Case, domain: Domain,
                        case_dir: Path) -> np.ndarray:
    """Load thermocouple x-positions (XTC). Sampling is done by sample_probes,
    which mirrors the Fortran SAMPLE_PROBES face-reconstruction interpolation,
    so we only need the positions here."""
    if not case.tc_file or case.tc_file == " ":
        return np.array([])
    return load_tc_locations(case_dir / case.tc_file, case.l0, case.nbrtc)


def _interp_face_from_cell(xq: float, x: np.ndarray, v: np.ndarray,
                           n: int, gradm: np.ndarray) -> float:
    """Port of INTERP_FACE_FROM_CELL: reconstruct face values from the two
    bracketing cells (via GRADM) and linearly interpolate between the faces
    surrounding xq. x are face positions x[1..n+1]; v is cell-centered with
    ghosts v[0..n+1]."""
    xtol = 1.0e-12 * max(1.0, x[n + 1] - x[1])
    if xq <= x[1] + xtol:
        return float(v[1])
    for i in range(1, n):
        if xq <= x[i + 1]:
            v0 = v[i - 1] + (v[i] - v[i - 1]) * gradm[i]
            v1 = v[i] + (v[i + 1] - v[i]) * gradm[i + 1]
            w = (xq - x[i]) / max(x[i + 1] - x[i], 1.0e-30)
            return float(v0 + (v1 - v0) * w)
    return float(v[n])


def sample_probes(arr: np.ndarray, xtc: np.ndarray,
                  domain: Domain) -> np.ndarray:
    """Sample a cell-centered field (with ghosts) at the thermocouple
    positions, mirroring the Fortran SAMPLE_PROBES: probes at/before the
    left boundary or at/beyond the last cell face take the boundary FACE
    value; interior probes use the face-reconstruction interpolation."""
    if xtc.size == 0:
        return np.array([])
    n = domain.n_cells
    x = domain.x
    out = np.empty(xtc.size)
    for j, xq in enumerate(xtc):
        if xq <= x[1] + 1.0e-12:
            out[j] = 0.5 * (arr[0] + arr[1])
        elif xq >= x[n] - 1.0e-12:
            out[j] = 0.5 * (arr[n] + arr[n + 1])
        else:
            out[j] = _interp_face_from_cell(xq, x, arr, n, domain.gradm)
    return out


# ----------------------- One timestep ----------------------- #


def step_physics(state: State, dt: float, case: Case, domain: Domain,
                 mats: Materials, solid: SolidParams,
                 fluxcm: np.ndarray, fluxcp: np.ndarray,
                 gas: GasTable | None = None, lgas: bool = False,
                 gas_energy: bool = False, pamb: float | None = None) -> State:
    """Physics step ONLY. Caller must have already applied BCs at state.time.

    Mirrors the energy + species update of the DO IT loop in 1Dheat.f. When
    lgas is set it also runs the Darcy gas solve and (if gas_energy) folds
    the gas internal energy and advection into the energy balance. Returns
    the next state with time advanced by dt.
    """
    n = domain.n_cells
    rhov_b = solid.rhov_bulk
    rhoc_b = solid.rhoc_bulk
    ci = slice(1, n + 1)

    # --- 1. Pyrolysis species update over the interior cells. ---
    T_interior = state.T[1:n + 1]
    rho_i_interior = state.rho_i[:, 1:n + 1]
    rho_i_new_interior, rho_bulk_new_interior, gas_src = step_species(
        T_interior, rho_i_interior, dt, solid)

    rho_i_new = state.rho_i.copy()
    rho_i_new[:, 1:n + 1] = rho_i_new_interior
    rho_new = state.rho.copy()
    rho_new[1:n + 1] = rho_bulk_new_interior
    # Mirror to ghosts (like the Fortran does: RHO(0) = RHO(1) etc.)
    rho_new[0] = rho_new[1]
    rho_new[n + 1] = rho_new[n]
    rho_i_new[:, 0] = rho_i_new[:, 1]
    rho_i_new[:, n + 1] = rho_i_new[:, n]

    # --- 2. Gas solve + advection/energy coupling (LGAS only). ---
    pg_new = state.pg
    mdotf = state.mdotf
    advect = 0.0
    poro_old = poro_new = poro_mean = None
    if lgas:
        pg_new, mdotf = solve_gas(state.T, state.rho, rho_new, gas_src,
                                  state.rhogsto, state.pg, pamb, dt,
                                  case, domain, solid, gas)
        # HFACE(I) = MDOTF(I)*hgas(T_face),  I = 1..n+1 (with BC zeroing)
        ff = np.arange(1, n + 2)
        tface_h = (state.T[ff - 1]
                   + (state.T[ff] - state.T[ff - 1]) * domain.gradm[ff])
        hface = np.zeros(n + 2)
        hface[ff] = mdotf[ff] * calc_hgas(tface_h, gas, case)
        if case.gbnd0 == 1 and mdotf[1] > 0.0:
            hface[1] = 0.0
        if case.gbndn == 1 and mdotf[n + 1] < 0.0:
            hface[n + 1] = 0.0
        advect = hface[2:n + 2] - hface[1:n + 1]            # (n,)
        if gas_energy:
            poro_old = _porosity_array(state.rho[ci], rhov_b, rhoc_b,
                                       case, solid)
            poro_new = _porosity_array(rho_new[ci], rhov_b, rhoc_b,
                                       case, solid)
            poro_mean = 0.5 * (poro_old + poro_new)

    # --- 3. Face conductivities k_face[i] for i = 1..n+1. ---
    # Important: tau at the face uses RHO_OLD (state.rho), NOT rho_new --
    # the Fortran's k_face computation (1Dheat.f lines 557-566) uses RHO,
    # not RHON. Using rho_new here was a bug that biased k in hot cells
    # where pyrolysis is fast.
    k_face = np.zeros(n + 2)
    rho_face = 0.5 * (state.rho[0:n + 1] + state.rho[1:n + 2])
    tau_face_arr = _tau_array(rho_face, rhov_b, rhoc_b, case, solid)
    T_face = np.zeros(n + 1)
    T_face[:] = state.T[1:n + 2] + (state.T[0:n + 1] - state.T[1:n + 2]) * \
        domain.gradp[0:n + 1]
    k_face[1:n + 2] = _k_mix_array(T_face, tau_face_arr, mats)

    # --- 4. Per-cell quantities for the energy update. ---
    # tau_old, tau_new, tau_mean (used to evaluate cp at mean tau).
    tau_old = _tau_array(state.rho[1:n + 1], rhov_b, rhoc_b, case, solid)
    tau_new = _tau_array(rho_new[1:n + 1], rhov_b, rhoc_b, case, solid)
    rho_mean = 0.5 * (state.rho[1:n + 1] + rho_new[1:n + 1])
    tau_mean = _tau_array(rho_mean, rhov_b, rhoc_b, case, solid)

    cp_mean = _cp_mix_array(T_interior, tau_mean, mats)
    if gas_energy:
        # Gas internal energy adds to the effective heat capacity:
        # ALP = ALPDV / (RHON*Cp + PORM*rho_gas(T,PG)*Cvg(T)).
        denom = (rho_new[1:n + 1] * cp_mean
                 + poro_mean * calc_rhogp(T_interior, pg_new[ci], gas, case)
                 * calc_cvg(T_interior, gas, case))
    else:
        denom = rho_new[1:n + 1] * cp_mean
    alp = (dt / domain.dv[1:n + 1]) / np.maximum(denom, 1.0e-30)

    # Fluxes at faces I and I+1 (per-cell view).
    # FLUXM(I) = k_face[i]   * fluxcm[i] * (T[i]   - T[i-1])     ... i = 1..n
    # FLUXP(I) = k_face[i+1] * fluxcp[i] * (T[i+1] - T[i])
    flux_m = k_face[1:n + 1] * fluxcm * (state.T[1:n + 1] - state.T[0:n])
    flux_p = k_face[2:n + 2] * fluxcp * (state.T[2:n + 2] - state.T[1:n + 1])

    # Old and new energies (heat + chemical), excluding gas-energy and Q_rad.
    hs_old = _hs_mix_array(T_interior, tau_old, case, mats)
    hs_form_old = _hs_form_array(tau_old, case, mats)
    hs_form_new = _hs_form_array(tau_new, case, mats)

    dv = domain.dv[1:n + 1]
    da_left = domain.da[1:n + 1]
    da_right = domain.da[2:n + 2]

    e_old = state.rho[1:n + 1] * hs_old * dv
    if gas_energy:
        # Stored gas internal energy at the old state: PORO*rho_gas*Ugas*dv.
        e_old = e_old + (poro_old
                         * calc_rhogp(T_interior, pg_new[ci], gas, case)
                         * calc_ugas(T_interior, gas, case) * dv)
    echem = (rho_new[1:n + 1] * hs_form_new
             - state.rho[1:n + 1] * hs_form_old) * dv
    net_flux = -flux_m * da_left + flux_p * da_right
    e_new = e_old + dt * (net_flux - advect) + echem

    # --- 5. 3-iteration Newton on T_new to satisfy ENEW = rho_new * hs_mix * dv. ---
    # T bounds from the cp tables (Fortran clamps with this).
    tmin_h = max(case.init_temp,
                 min(mats.cp_virgin.T[0], mats.cp_char.T[0]))
    tmax_h = max(mats.cp_virgin.T[-1], mats.cp_char.T[-1])

    # Initial guess: T + dt*alp*(flux - advect) (the explicit Euler step)
    T_n = np.clip(T_interior + alp * (net_flux - advect), tmin_h, tmax_h)
    for _ in range(3):
        hs_n = _hs_mix_array(T_n, tau_new, case, mats)
        cp_n = _cp_mix_array(T_n, tau_new, mats)
        h_new = rho_new[1:n + 1] * hs_n * dv
        cp_den = rho_new[1:n + 1] * cp_n
        if gas_energy:
            rhogn = calc_rhogp(T_n, pg_new[ci], gas, case)
            h_new = h_new + poro_new * rhogn * calc_ugas(T_n, gas, case) * dv
            cp_den = cp_den + poro_new * rhogn * calc_cvg(T_n, gas, case)
        T_n = T_n + (e_new - h_new) / np.maximum(cp_den * dv, 1.0e-30)
        T_n = np.clip(T_n, tmin_h, tmax_h)

    # --- 6. Assemble new full T array (still with old ghosts; BCs will reset
    # them next step). The interior gets updated; we also clamp at >= init_temp
    # to match the Fortran post-step floor at line 688. ---
    T_full_new = state.T.copy()
    T_full_new[1:n + 1] = np.maximum(case.init_temp, T_n)

    # --- 7. Update stored gas density for the next step's storage term. ---
    rhogsto_new = state.rhogsto
    if lgas:
        rhogsto_new = (_porosity_array(rho_new[ci], rhov_b, rhoc_b, case, solid)
                       * calc_rhogp(T_full_new[ci], pg_new[ci], gas, case))

    return State(
        time=state.time + dt,
        T=T_full_new,
        rho=rho_new,
        rho_i=rho_i_new,
        pg=pg_new,
        mdotf=mdotf,
        rhogsto=rhogsto_new,
    )


# ----------------------- Driver ----------------------- #


def run(case_dir: str | Path, output_dir: str | Path | None = None,
        verbose: bool = True, lgas: bool | None = None,
        record_every: int | None = None,
        write_con: bool = True,
        time_table: TimeTable | None = None,
        nbrn: int | None = None,
        record_dt: float | None = None) -> dict:
    """Drive a full simulation from a case directory.

    Reads heat.case + all data files, runs the time loop, writes
    a Python output file matching the Fortran's con.out format
    (time, T at each TC, rho at each TC).

    lgas enables the gas (Darcy + advection) physics, equivalent to the
    Fortran '-g' flag. When None it is auto-enabled for cases that need it
    (gas-energy or aero boundary).

    record_every: if set, capture the full per-cell field (T, rho, rho_i,
    and pg/mdotf for gas runs) every `record_every` solver steps, with BCs
    applied (ghost cells carry the boundary forcing). Snapshots are taken at
    the *start* of the step, so consecutive snapshots are `record_every`
    solver steps apart -- a valid (state_t, state_{t+gap}) training pair.
    Returned under key 'trajectory'. write_con=False skips the con.out file
    (useful for bulk data generation).

    Returns a dict with run stats (n_steps, wall_time, etc.).
    """
    case_dir = Path(case_dir)
    if output_dir is None:
        output_dir = case_dir / "python_out"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Load all case data ---
    case = load_case(case_dir / "heat.case")
    # nbrn override lets a data-gen caller change the mesh resolution without
    # editing heat.case -- the uniform grid just gets more/fewer cells over the
    # same L0 (for mesh-resolution generalization experiments).
    if nbrn is not None:
        case.nbrn = nbrn
    domain = setup_domain(case)
    mats = load_materials(case, case_dir, lrad=False, leff=False)
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    # time_table override lets a data-gen caller inject a custom boundary
    # forcing (wall-temperature / flux history) without editing files.
    if time_table is not None:
        time_tbl = time_table
    else:
        time_tbl = (load_time_table(case_dir / case.time_file)
                    if case.use_time_bcn else None)
    xtc = setup_thermocouples(case, domain, case_dir)

    # --- Gas / aerothermal data (phase 4) ---
    # lgas mirrors the Fortran '-g' CLI flag, NOT case.gas_energy (which is
    # only a sub-option active *within* gas mode). The case file does not
    # encode the -g choice, so default OFF (Fortran's default) and let the
    # caller opt in. Auto-enable only when the case clearly cannot run
    # without gas, i.e. it uses the aero boundary (which needs mdotf).
    if lgas is None:
        lgas = case.use_aero_bcn
    gas = load_gas_table(case_dir / case.gas_file) if lgas else None
    aero = (load_aero_table(case_dir / case.aero_file)
            if (lgas and case.use_aero_bcn) else None)
    bprime = (load_bprime(case_dir / case.bprime_dir)
              if (lgas and case.use_aero_bcn) else None)
    gas_energy = bool(lgas and case.gas_energy)

    n = domain.n_cells

    # Precompute geometric flux coefficients (Fortran FLUXCM, FLUXCP).
    fluxcm = 2.0 / (domain.dx[1:n + 1] + domain.dx[0:n])         # (n,)
    fluxcp = 2.0 / (domain.dx[2:n + 2] + domain.dx[1:n + 1])     # (n,)

    # --- Initial conditions ---
    T0 = np.full(n + 2, case.init_temp)
    rho_i0 = np.tile(solid.rho_v[:, None], (1, n + 2))
    rho0 = np.full(n + 2, solid.rhov_bulk)
    pg0 = mdotf0 = rhogsto0 = None
    if lgas:
        pamb0 = (calc_aero_p(case.time_init, aero, case)
                 if case.use_aero_bcn else case.pamb)
        pg0 = np.full(n + 2, pamb0)
        mdotf0 = np.zeros(n + 2)
        # RHOGSTO = porosity(virgin)*rho_gas(T_init, pamb) = phi*rho_gas(...)
        phi_virgin = _porosity_array(np.full(n, solid.rhov_bulk),
                                     solid.rhov_bulk, solid.rhoc_bulk,
                                     case, solid)
        rhogsto0 = phi_virgin * calc_rhogp(case.init_temp, pamb0, gas, case)
    state = State(time=case.time_init, T=T0, rho=rho0, rho_i=rho_i0,
                  pg=pg0, mdotf=mdotf0, rhogsto=rhogsto0)

    # --- Time discretization ---
    dt_nom = compute_initial_dt(case, domain, mats, solid)
    n_steps_est = int((case.time_final - case.time_init) / dt_nom) + 2

    if verbose:
        print(f"=== {case.name} ===")
        print(f"  n_cells = {n}, dx = {domain.dx[1]:.4e} m")
        print(f"  dt_nom = {dt_nom:.4e} s, est steps = {n_steps_est}")
        print(f"  time = {case.time_init} -> {case.time_final} s")

    # Output frequency: roughly NBRT/min(NBRT/2, 1000) steps per write.
    ndebug = min(n_steps_est // 2, 1000) if n_steps_est > 2 else 1
    out_every = max(1, n_steps_est // ndebug)
    out_path = output_dir / "con.out"
    out_f = open(out_path, "w") if write_con else None

    # Full-field trajectory recording (for GNN training data).
    # record_dt records at a fixed PHYSICAL interval (resolution-independent,
    # for mesh-generalization experiments); record_every records every N solver
    # steps (snapshot dt then scales with dt_nom ~ dx^2 -- resolution-dependent).
    rec = (record_every is not None and record_every > 0) or (
        record_dt is not None and record_dt > 0)
    rec_by_time = record_dt is not None and record_dt > 0
    next_rec_time = case.time_init
    traj_time, traj_T, traj_rho, traj_rhoi = [], [], [], []
    traj_pg, traj_mdotf = [], []

    # --- Time loop ---
    # Order matches Fortran's PROGRAM HEAT (lines 348-715):
    #   for IT in 1..NBRT:
    #       pick dt, apply BCs at TIME, sample probes (write if interval hits),
    #       run physics, TIME += dt
    t0 = _time.time()
    n_steps = 0
    while state.time < case.time_final:
        n_steps += 1
        dt = dt_nom
        if state.time + dt > case.time_final:
            dt = case.time_final - state.time
        # Clip to land exactly on next BC event (matches Fortran lines 351-369),
        # considering both the flux time-table and the aero time-table.
        next_event = case.time_final
        if case.use_time_bcn and time_tbl is not None:
            future = time_tbl.t[time_tbl.t > state.time + 1.0e-12]
            if future.size > 0:
                next_event = min(next_event, float(future[0]))
        if case.use_aero_bcn and aero is not None:
            future = aero.t[aero.t > state.time + 1.0e-12]
            if future.size > 0:
                next_event = min(next_event, float(future[0]))
        dt = min(dt, max(1.0e-12, next_event - state.time))

        # Ambient pressure this step (CALC_AERO_P), shared by BC and gas solve.
        pamb = (calc_aero_p(state.time, aero, case)
                if case.use_aero_bcn else case.pamb)

        # Apply BCs at current time (modifies ghost cells in state.T in place).
        apply_bcs(state.T, state.rho, state.time, case, domain, mats, solid,
                  time_tbl, time_tbl, case.init_temp, lrad=False,
                  aero=aero, bprime=bprime, gas=gas, mdotf=state.mdotf,
                  use_aero=case.use_aero_bcn)

        # Write probes BEFORE the physics step (matches Fortran's order).
        if write_con and n_steps % out_every == 0:
            # WRITE_THERM_HISTORY format:
            #   time, T_surface, T_tc..., T_bottom, rho_surface, rho_tc..., rho_bottom
            # where T_surface = 0.5*(T[0]+T[1]) and T_bottom = 0.5*(T[N+1]+T[N]).
            t_probes = sample_probes(state.T, xtc, domain)
            r_probes = sample_probes(state.rho, xtc, domain)
            t_surf = 0.5 * (state.T[0] + state.T[1])
            t_bot = 0.5 * (state.T[n + 1] + state.T[n])
            r_surf = 0.5 * (state.rho[0] + state.rho[1])
            r_bot = 0.5 * (state.rho[n + 1] + state.rho[n])
            row = ([state.time, t_surf] + t_probes.tolist() + [t_bot]
                   + [r_surf] + r_probes.tolist() + [r_bot])
            out_f.write(" ".join(f"{v:.10e}" for v in row) + "\n")

        # Full-field snapshot (BCs applied -> ghosts carry the forcing).
        if rec_by_time:
            take = state.time >= next_rec_time - 1.0e-12
            if take:
                next_rec_time += record_dt
        else:
            take = (n_steps - 1) % record_every == 0
        if rec and take:
            traj_time.append(state.time)
            traj_T.append(state.T.copy())
            traj_rho.append(state.rho.copy())
            traj_rhoi.append(state.rho_i.copy())
            if lgas:
                traj_pg.append(state.pg.copy())
                traj_mdotf.append(state.mdotf.copy())

        # Physics step (BC already applied above, so skip inside).
        state = step_physics(state, dt, case, domain, mats, solid,
                             fluxcm, fluxcp, gas=gas, lgas=lgas,
                             gas_energy=gas_energy, pamb=pamb)

    if out_f is not None:
        out_f.close()
    elapsed = _time.time() - t0

    # Assemble the recorded trajectory (full per-cell field over time).
    trajectory = None
    if rec and traj_time:
        trajectory = dict(
            time=np.asarray(traj_time),                 # (S,)
            T=np.stack(traj_T),                         # (S, n+2)  incl. ghosts
            rho=np.stack(traj_rho),                     # (S, n+2)
            rho_i=np.stack(traj_rhoi),                  # (S, n_species, n+2)
            x=domain.x.copy(),                          # (n+2,) face positions
            dx=domain.dx.copy(), da=domain.da.copy(), dv=domain.dv.copy(),
            n_cells=n, n_species=solid.n_species,
            dt_nom=dt_nom, lgas=lgas, record_every=record_every,
            rhov_bulk=solid.rhov_bulk, rhoc_bulk=solid.rhoc_bulk,
            init_temp=case.init_temp, case_name=case.name,
        )
        if lgas:
            trajectory["pg"] = np.stack(traj_pg)        # (S, n+2)
            trajectory["mdotf"] = np.stack(traj_mdotf)  # (S, n+2)

    # Final summary
    if verbose:
        print(f"  completed: {n_steps} steps, {elapsed:.2f} s wall")
        print(f"  final time: {state.time:.4e} s")
        print(f"  final T (interior min/mean/max): "
              f"{state.T[1:n+1].min():.2f} / "
              f"{state.T[1:n+1].mean():.2f} / "
              f"{state.T[1:n+1].max():.2f} K")
        print(f"  final rho (interior min/mean/max): "
              f"{state.rho[1:n+1].min():.2f} / "
              f"{state.rho[1:n+1].mean():.2f} / "
              f"{state.rho[1:n+1].max():.2f} kg/m^3")
        if write_con:
            print(f"  output: {out_path}")

    return dict(case_name=case.name, n_steps=n_steps,
                wall_time_s=elapsed, output_path=str(out_path),
                final_state=state, trajectory=trajectory)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", type=str, default="heat_2026-04-11_1837/examples/aw1")
    p.add_argument("--quiet", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gas", dest="lgas", action="store_true", default=None,
                   help="force gas physics on (Fortran -g)")
    g.add_argument("--no-gas", dest="lgas", action="store_false",
                   help="force gas physics off")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case_dir = (repo / args.case_dir) if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    run(case_dir, verbose=not args.quiet, lgas=args.lgas)
