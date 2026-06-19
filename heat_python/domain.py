"""1D finite-volume domain. Replaces DOMA.INC + the grid-setup block in
PROGRAM HEAT (1Dheat.f lines ~140-195).

Convention: arrays are padded with one ghost cell on each side and indexed
to match the Fortran (1-indexed semantically). For an N-cell mesh:

    index:    0       1       2     ...      N      N+1
              ghost   real    real            real   ghost

So dx[0] and dx[N+1] hold ghost values (copies of dx[1] and dx[N]),
and the interior is dx[1:N+1].

Face positions x[1..N+1] go from 0 (inner boundary) to l0 (outer
boundary), so there are N cells and N+1 faces.

For non-Cartesian geometries (cylindrical / spherical), face areas and
cell volumes are computed from the radial coordinate:

    m=0  (Cartesian):    da = 1,             dv = dx
    m=1  (Cylindrical):  da = 2*pi*x,        dv = pi*(x_R^2 - x_L^2)
    m=2  (Spherical):    da = 4*pi*x^2,      dv = (4/3)*pi*(x_R^3 - x_L^3)
"""

from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np

from .case import Case


@dataclass
class Domain:
    """1D mesh + geometric weights. Built once at the start of a run."""
    n_cells: int                       # NBRN
    geometry: int                      # m: 0=Cart, 1=Cyl, 2=Sph
    l0: float                          # domain length [m]
    dx: np.ndarray                     # (n_cells+2,) cell widths
    x: np.ndarray                      # (n_cells+2,) face positions; x[1..n+1]
    da: np.ndarray                     # (n_cells+2,) face areas;     da[1..n+1]
    dv: np.ndarray                     # (n_cells+2,) cell volumes;   dv[0..n+1]
    gradm: np.ndarray                  # (n_cells+2,) face-from-cell weight (left side)
    gradp: np.ndarray                  # (n_cells+2,) face-from-cell weight (right side)
    max_dx: float                      # max cell width (for CFL)

    @property
    def cell_centers(self) -> np.ndarray:
        """x-coordinate of each interior cell center (length n_cells)."""
        return 0.5 * (self.x[1:self.n_cells + 1] + self.x[2:self.n_cells + 2])


def setup_domain(case: Case) -> Domain:
    """Build the mesh from the case config. Mirrors 1Dheat.f lines 140-195."""
    n = case.nbrn
    m = case.m
    l0 = case.l0

    if m < 0 or m > 2:
        raise ValueError(f"Geometry m must be 0/1/2, got {m}")

    # Uniform grid (F = 1 in Martin's code). Stretched grid (F != 1) is
    # supported in the Fortran but isn't used by any example case; we
    # implement only the uniform case for now.
    dx_uniform = l0 / n
    dx = np.full(n + 2, dx_uniform)        # ghosts mirror interior
    # If anyone wants stretched: dx[1] = l0*(1-F)/(1-F**n); dx[i] = F*dx[i-1]

    # Face positions: x[1] = 0, x[i] = x[i-1] + dx[i-1]
    # Vectorize: x[1] = 0, x[2..n+1] = cumsum(dx[1..n])
    x = np.zeros(n + 2)
    x[1] = 0.0
    x[2:n + 2] = np.cumsum(dx[1:n + 1])
    # x[0] is unused; leave as 0

    # Face-from-cell interpolation weights
    # gradm[i] = dx[i-1] / (dx[i-1] + dx[i])  for i = 1..n+1
    # gradp[i] = dx[i+1] / (dx[i] + dx[i+1])  for i = 0..n
    gradm = np.zeros(n + 2)
    gradp = np.zeros(n + 2)
    gradm[1:n + 2] = dx[0:n + 1] / (dx[0:n + 1] + dx[1:n + 2])
    gradp[0:n + 1] = dx[1:n + 2] / (dx[0:n + 1] + dx[1:n + 2])

    # Face areas: da[i] = (2*x[i])^m * pi^(0.5*m*(3-m)) for i = 1..n+1
    da = np.zeros(n + 2)
    pi_exp = 0.5 * m * (3.0 - m)
    da[1:n + 2] = (2.0 * x[1:n + 2]) ** m * math.pi ** pi_exp

    # Cell volumes: dv[i] = 2^m * pi^(m*(3-m)/2) / (m+1) * (x[i+1]^(m+1) - x[i]^(m+1))
    dv = np.zeros(n + 2)
    vol_pi_exp = m * (3 - m) / 2.0
    vol_coef = (2.0 ** m) * math.pi ** vol_pi_exp / (m + 1)
    dv[1:n + 1] = vol_coef * (x[2:n + 2] ** (m + 1) - x[1:n + 1] ** (m + 1))
    dv[0] = dv[1]
    dv[n + 1] = dv[n]

    return Domain(
        n_cells=n,
        geometry=m,
        l0=l0,
        dx=dx,
        x=x,
        da=da,
        dv=dv,
        gradm=gradm,
        gradp=gradp,
        max_dx=float(dx[1:n + 1].max()),
    )


if __name__ == "__main__":
    # Sanity check: build the aw1 mesh and confirm vs known values
    from pathlib import Path
    from .case import load_case

    repo = Path(__file__).resolve().parents[1]
    case = load_case(repo / "heat_2026-04-11_1837" / "examples" / "aw1" / "heat.case")
    d = setup_domain(case)

    print(f"aw1 mesh:")
    print(f"  n_cells = {d.n_cells}")
    print(f"  geometry = {d.geometry} (Cartesian)")
    print(f"  l0 = {d.l0} m")
    print(f"  dx (uniform) = {d.dx[1]:.6e} m")
    print(f"  x[1] = {d.x[1]}, x[n+1] = {d.x[d.n_cells + 1]} (should be 0 and l0)")
    print(f"  da[1] = {d.da[1]}, da[n+1] = {d.da[d.n_cells + 1]} (Cartesian = 1)")
    print(f"  dv[1] = {d.dv[1]:.6e} (Cartesian = dx)")
    print(f"  gradm[1] = {d.gradm[1]} (uniform grid = 0.5)")
    print(f"  gradp[1] = {d.gradp[1]} (uniform grid = 0.5)")
    print(f"  max_dx = {d.max_dx:.6e}")

    # Try cylindrical / spherical too (synthetic case)
    for geom_name, m in [("Cylindrical", 1), ("Spherical", 2)]:
        case.m = m
        case.l0 = 1.0
        case.nbrn = 4
        d = setup_domain(case)
        print(f"\n{geom_name} (m={m}, l0=1, n=4):")
        print(f"  x  = {d.x[1:6]}")
        print(f"  da = {d.da[1:6]}")
        print(f"  dv = {d.dv[1:5]}")
