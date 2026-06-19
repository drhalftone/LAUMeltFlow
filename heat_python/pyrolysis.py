"""Pyrolysis chemistry. Replaces DECOMP.INC, READ_RHO (solid.dat parser),
CALC_TAU, CALC_BETA, CALC_POROSITY, CALC_RHOVB, CALC_RHOCB, DECOMP, and the
inline Arrhenius species update in PROGRAM HEAT (1Dheat.f lines ~482-513).

For the TACOT 3.0 model (3 reactive species + porosity), each species k
has parameters:
    rho_v_i  - intrinsic virgin density (at full virgin)        [kg/m^3]
    rho_c_i  - intrinsic char density (at full char)            [kg/m^3]
    A_i      - Arrhenius pre-exponential                        [1/s]
    psi_i    - reaction order
    E_i      - activation energy / R (often called E_over_R)    [K]
    gamma_i  - mass-fraction weight in the bulk (sums to 1 by species)
    T_min_i  - minimum activation temperature                    [K]

The cell-level update over one timestep dt, for each species:

    W = (rho_i - rho_c_i) / rho_v_i      (normalized 0=char, 1=virgin)

    if psi == 1:   W_new = W * exp(-A*dt*exp(-E/T))
    else:          W_new = (W^(1-psi) - dt*(1-psi)*A*exp(-E/T))^(1/(1-psi))

    rho_i_new = rho_c_i + rho_v_i * W_new

The bulk density rho = sum_k (1-phi)*gamma_k * rho_i_k, and the gas
source from pyrolysis is sum_k (1-phi)*gamma_k * (rho_i_old - rho_i_new)/dt.

The "char-progress" variable tau (1=virgin, 0=char) is used by the
materials module to blend virgin/char properties.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .case import Case


@dataclass
class SolidParams:
    """Per-species Arrhenius parameters + porosity for a TPS material.

    Loaded once from solid.dat at the start of a run.
    """
    n_species: int                       # NBRS
    phi: float                           # porosity of virgin material
    rho_v: np.ndarray                    # (n_species,) intrinsic virgin density
    rho_c: np.ndarray                    # (n_species,) intrinsic char density
    A: np.ndarray                        # (n_species,) Arrhenius prefactor
    psi: np.ndarray                      # (n_species,) reaction order
    E: np.ndarray                        # (n_species,) E_over_R
    gamma: np.ndarray                    # (n_species,) mass-fraction weights
    T_min: np.ndarray                    # (n_species,) min activation T
    rho_w: np.ndarray = field(init=False)  # (n_species,) = (1-phi)*gamma

    def __post_init__(self):
        self.rho_w = (1.0 - self.phi) * self.gamma

    @property
    def rhov_bulk(self) -> float:
        """sum_k (1-phi)*gamma_k * rho_v_k  --- CALC_RHOVB"""
        return float(np.sum(self.rho_w * self.rho_v))

    @property
    def rhoc_bulk(self) -> float:
        """sum_k (1-phi)*gamma_k * rho_c_k  --- CALC_RHOCB"""
        return float(np.sum(self.rho_w * self.rho_c))


def _skip_comments(f) -> str:
    """Read past '#' comment lines and return the next data line."""
    while True:
        line = f.readline()
        if not line:
            return ""
        if not line.strip().startswith("#"):
            return line


def load_solid(path: str | Path, solid_kats: bool = False) -> SolidParams:
    """Parse solid.dat (TACOT-3.0-style). Format:
        # comment
        <phi>
        # comment
        <n_species>
        # comment
        rho_v_1  rho_c_1  A_1  psi_1  E_1  gamma_1  T_min_1
        rho_v_2  rho_c_2  A_2  psi_2  E_2  gamma_2  T_min_2
        ...

    If solid_kats is True, rho_v and rho_c are scaled by 1/(1-phi) to
    convert from bulk to intrinsic. Mirrors READ_RHO in 1Dheat.f.
    """
    with open(path) as f:
        phi = float(_skip_comments(f).split()[0])
        n = int(_skip_comments(f).split()[0])
        # Skip the species-table header
        data_line = _skip_comments(f)
        rows = [data_line]
        for _ in range(n - 1):
            rows.append(f.readline())

    arr = np.array([list(map(float, r.split())) for r in rows])
    rho_v = arr[:, 0]
    rho_c = arr[:, 1]

    if solid_kats:
        rho_v = rho_v / max(1.0 - phi, 1.0e-30)
        rho_c = rho_c / max(1.0 - phi, 1.0e-30)

    return SolidParams(
        n_species=n,
        phi=phi,
        rho_v=rho_v,
        rho_c=rho_c,
        A=arr[:, 2],
        psi=arr[:, 3],
        E=arr[:, 4],
        gamma=arr[:, 5],
        T_min=arr[:, 6],
    )


# ----------------------- Cell-level quantities ----------------------- #


def calc_tau(rho: float, rhov_bulk: float, rhoc_bulk: float,
             case: Case, solid: SolidParams) -> float:
    """Char-progress variable tau in [0, 1]. tau=1 means fully virgin,
    tau=0 means fully char. Mirrors CALC_TAU in 1Dheat.f.

    The Fortran does a slightly subtle accounting for "inert" species
    (those with rho_v == rho_c, which never decompose). They get
    subtracted from rho before computing the ratio.
    """
    # Inert vs reactive split among the species' intrinsic virgin densities
    inert_mask = np.abs(solid.rho_v - solid.rho_c) < 1.0e-12
    sum_inert = float(np.sum(solid.rho_v[inert_mask]))
    sum_react = float(np.sum(solid.rho_v[~inert_mask]))
    total = sum_inert + sum_react
    if total > 1.0e-30:
        rho_inert = rhov_bulk * sum_inert / total
    else:
        rho_inert = 0.0

    rho_vr = max(rhov_bulk - rho_inert, 1.0e-30)
    rho_cr = max(rhoc_bulk - rho_inert, 0.0)
    rho_r = max(rho - rho_inert, 0.0)

    if case.tau_linear:
        den = max(rho_vr - rho_cr, 1.0e-30)
        tau = (rho_r - rho_cr) / den
    else:
        den = 1.0 - rho_cr / max(rho_vr, 1.0e-30)
        if den <= 1.0e-30:
            tau = 0.0
        else:
            tau = (1.0 - rho_cr / max(rho_r, 1.0e-30)) / den
    return float(min(1.0, max(0.0, tau)))


def calc_beta(rho: float, rhov_bulk: float, rhoc_bulk: float) -> float:
    """Linear char fraction in [0, 1]: 0 = virgin, 1 = char. Used to blend
    porosity. (CALC_BETA in 1Dheat.f)
    """
    beta = (rhov_bulk - rho) / max(rhov_bulk - rhoc_bulk, 1.0e-30)
    return float(min(1.0, max(0.0, beta)))


def calc_porosity(rho: float, rhov_bulk: float, rhoc_bulk: float,
                  case: Case, solid: SolidParams) -> float:
    """Porosity blended between virgin (PHI) and char (CASE_PHI_C) by beta.
    Clamped to [PHI_min, PHI_max], then to [1e-6, 0.99]. (CALC_POROSITY)
    """
    beta = calc_beta(rho, rhov_bulk, rhoc_bulk)
    porosity = solid.phi + (case.phi_c - solid.phi) * beta
    phi_min = min(solid.phi, case.phi_c)
    phi_max = max(solid.phi, case.phi_c)
    porosity = max(phi_min, min(phi_max, porosity))
    return float(max(1.0e-6, min(0.99, porosity)))


def calc_eps_rho(rho: float, rhov_bulk: float, rhoc_bulk: float,
                 case: Case, solid: SolidParams) -> float:
    """Emissivity blended by tau. (CALC_EPS_RHO in 1Dheat.f)"""
    tau = calc_tau(rho, rhov_bulk, rhoc_bulk, case, solid)
    return tau * case.eps_v + (1.0 - tau) * case.eps_c


# ----------------------- Species update step ----------------------- #


def decomp_rate(T: float, rho_i: float, k: int, solid: SolidParams) -> float:
    """Instantaneous decomposition rate dRHO_i/dt for species k at temp T.
    Returns the rate (negative for decomposition). DECOMP in 1Dheat.f.
    """
    W = (rho_i - solid.rho_c[k]) / max(solid.rho_v[k], 1.0e-30)
    W = min(1.0, max(0.0, W))
    if W <= 0.0:
        return 0.0
    return float(-solid.A[k] * max(solid.rho_v[k], 0.0)
                 * W ** solid.psi[k] * np.exp(-solid.E[k] / T))


def step_species_cell(T: float, rho_i: np.ndarray, dt: float,
                      solid: SolidParams) -> tuple[np.ndarray, float, np.ndarray]:
    """Closed-form Arrhenius update for all species in one cell over dt.

    Returns:
        rho_i_new: (n_species,) new per-species densities
        rho_bulk_new: float, sum_k rho_w_k * rho_i_new_k
        gas_src_per_k: (n_species,) per-species gas mass source [kg/m^3/s]
                       (the LGAS branch in the main loop)
    """
    n = solid.n_species
    W = (rho_i - solid.rho_c) / np.maximum(solid.rho_v, 1.0e-30)
    W = np.clip(W, 0.0, 1.0)
    W_new = W.copy()
    active = W > 0.0

    # First-order branch (psi == 1)
    psi1 = active & (np.abs(solid.psi - 1.0) < 1.0e-12)
    if np.any(psi1):
        W_new[psi1] = W[psi1] * np.exp(
            -solid.A[psi1] * dt * np.exp(-solid.E[psi1] / T)
        )

    # General-order branch (psi != 1)
    other = active & ~psi1
    if np.any(other):
        term = (W[other] ** (1.0 - solid.psi[other])
                - dt * (1.0 - solid.psi[other]) * solid.A[other]
                * np.exp(-solid.E[other] / T))
        term = np.maximum(0.0, term)
        W_new[other] = term ** (1.0 / (1.0 - solid.psi[other]))

    W_new = np.clip(W_new, 0.0, 1.0)
    rho_i_new = solid.rho_c + solid.rho_v * W_new
    rho_i_new = np.maximum(solid.rho_c, rho_i_new)
    rho_bulk_new = float(np.sum(solid.rho_w * rho_i_new))
    gas_src_per_k = solid.rho_w * np.maximum(0.0, rho_i - rho_i_new) / max(dt, 1.0e-30)
    return rho_i_new, rho_bulk_new, gas_src_per_k


def step_species(T: np.ndarray, rho_i: np.ndarray, dt: float,
                 solid: SolidParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized version of step_species_cell across all cells.

    Args:
        T:     (n_cells,) cell temperatures
        rho_i: (n_species, n_cells) per-species densities
        dt:    timestep
        solid: pyrolysis parameters

    Returns:
        rho_i_new:   (n_species, n_cells)
        rho_bulk_new: (n_cells,)  bulk density per cell
        gas_src:     (n_cells,)   total gas source per cell  (sum over species)
    """
    n_species, n_cells = rho_i.shape
    # Broadcast: A[k], psi[k], E[k] are (n_species,); T[i] is (n_cells,)
    # Need to make rho_i, A, psi, E all have shape (n_species, n_cells)
    A = solid.A[:, None]
    psi = solid.psi[:, None]
    E = solid.E[:, None]
    rho_v = solid.rho_v[:, None]
    rho_c = solid.rho_c[:, None]

    W = (rho_i - rho_c) / np.maximum(rho_v, 1.0e-30)
    W = np.clip(W, 0.0, 1.0)
    W_new = W.copy()
    active = W > 0.0

    psi1_mask = (np.abs(psi - 1.0) < 1.0e-12) & np.ones_like(W, dtype=bool)
    psi1 = active & psi1_mask
    if np.any(psi1):
        # exp(-A*dt*exp(-E/T))
        exp_arg = -np.broadcast_to(A, W.shape) * dt * np.exp(
            -np.broadcast_to(E, W.shape) / T[None, :]
        )
        W_new = np.where(psi1, W * np.exp(exp_arg), W_new)

    other = active & ~psi1_mask
    if np.any(other):
        bA = np.broadcast_to(A, W.shape)
        bpsi = np.broadcast_to(psi, W.shape)
        bE = np.broadcast_to(E, W.shape)
        term = W ** (1.0 - bpsi) - dt * (1.0 - bpsi) * bA * np.exp(-bE / T[None, :])
        term = np.maximum(0.0, term)
        W_new = np.where(other, term ** (1.0 / (1.0 - bpsi)), W_new)

    W_new = np.clip(W_new, 0.0, 1.0)
    rho_i_new = rho_c + rho_v * W_new
    rho_i_new = np.maximum(rho_c, rho_i_new)

    rho_w = solid.rho_w[:, None]
    rho_bulk_new = np.sum(rho_w * rho_i_new, axis=0)
    gas_src_per_k = rho_w * np.maximum(0.0, rho_i - rho_i_new) / max(dt, 1.0e-30)
    gas_src = np.sum(gas_src_per_k, axis=0)
    return rho_i_new, rho_bulk_new, gas_src


if __name__ == "__main__":
    from pathlib import Path
    from .case import load_case

    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / "heat_2026-04-11_1837" / "examples" / "aw1"
    case = load_case(case_dir / "heat.case")
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)

    print(f"aw1 solid parameters:")
    print(f"  n_species  = {solid.n_species}")
    print(f"  phi        = {solid.phi}")
    print(f"  rho_v      = {solid.rho_v}")
    print(f"  rho_c      = {solid.rho_c}")
    print(f"  A          = {solid.A}")
    print(f"  psi        = {solid.psi}")
    print(f"  E (E/R)    = {solid.E} K")
    print(f"  gamma      = {solid.gamma}")
    print(f"  T_min      = {solid.T_min}")
    print(f"  rho_w      = {solid.rho_w}  (= (1-phi)*gamma)")
    print(f"  rhov_bulk  = {solid.rhov_bulk:.4f} kg/m^3 (CALC_RHOVB)")
    print(f"  rhoc_bulk  = {solid.rhoc_bulk:.4f} kg/m^3 (CALC_RHOCB)")

    # Sanity tests
    rhov_b = solid.rhov_bulk
    rhoc_b = solid.rhoc_bulk

    print(f"\nVirgin material (rho = rhov_bulk):")
    print(f"  tau    = {calc_tau(rhov_b, rhov_b, rhoc_b, case, solid):.4f} (should be 1.0)")
    print(f"  beta   = {calc_beta(rhov_b, rhov_b, rhoc_b):.4f} (should be 0.0)")
    print(f"  porosity = {calc_porosity(rhov_b, rhov_b, rhoc_b, case, solid):.4f} (should be phi={solid.phi})")

    print(f"\nChar (rho = rhoc_bulk):")
    print(f"  tau    = {calc_tau(rhoc_b, rhov_b, rhoc_b, case, solid):.4f} (should be 0.0)")
    print(f"  beta   = {calc_beta(rhoc_b, rhov_b, rhoc_b):.4f} (should be 1.0)")
    print(f"  porosity = {calc_porosity(rhoc_b, rhov_b, rhoc_b, case, solid):.4f} (should be phi_c={case.phi_c})")

    # Step a single cell at a hot temperature
    print(f"\nSingle-cell species update at T=1500K, dt=0.01s, rho_i = rho_v (fully virgin):")
    rho_i_init = solid.rho_v.copy()
    rho_i_new, rho_bulk_new, gas_per_k = step_species_cell(
        1500.0, rho_i_init, 0.01, solid)
    print(f"  rho_i: {rho_i_init} -> {rho_i_new}")
    print(f"  rho_bulk_new = {rho_bulk_new:.4f}  (initial rhov_bulk={rhov_b:.4f})")
    print(f"  gas src per species: {gas_per_k}")

    # Vectorized version on a small mesh
    n_cells = 5
    T = np.array([500.0, 800.0, 1000.0, 1500.0, 2000.0])
    rho_i_mesh = np.tile(solid.rho_v[:, None], (1, n_cells))
    r_new, b_new, gas = step_species(T, rho_i_mesh, 0.01, solid)
    print(f"\nVectorized over a 5-cell mesh with T = {T}:")
    print(f"  rho_bulk_new = {b_new}")
    print(f"  gas_src      = {gas}")
