"""
Discontinuous Galerkin solver for perfect gas.

This module implements a DG method for the 1D Euler equations that can be
directly compared with the finite volume (Roe) solver.

Key features:
- p=0 (constant basis) is mathematically equivalent to finite volume
- p=1 (linear basis) gives 2nd order accuracy
- Uses the same Roe numerical flux as the FVM solver for fair comparison

References:
- Cockburn & Shu (1998) - "The Runge-Kutta Discontinuous Galerkin Method"
- Hesthaven & Warburton (2008) - "Nodal Discontinuous Galerkin Methods"
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters

from .flux import roe_flux1D
from .thermodynamics import pres


def legendre_basis(p: int, xi: float) -> np.ndarray:
    """
    Evaluate Legendre polynomial basis functions at point xi in [-1, 1].

    Parameters
    ----------
    p : int
        Polynomial order (0, 1, or 2 supported)
    xi : float
        Local coordinate in [-1, 1]

    Returns
    -------
    np.ndarray
        Basis function values [phi_0, phi_1, ...] at xi
    """
    if p == 0:
        return np.array([1.0])
    elif p == 1:
        return np.array([1.0, xi])
    elif p == 2:
        return np.array([1.0, xi, 0.5 * (3.0 * xi**2 - 1.0)])
    else:
        raise ValueError(f"DG order p={p} not supported (max p=2)")


def legendre_basis_deriv(p: int, xi: float) -> np.ndarray:
    """
    Evaluate derivatives of Legendre polynomial basis functions at point xi.

    Parameters
    ----------
    p : int
        Polynomial order (0, 1, or 2 supported)
    xi : float
        Local coordinate in [-1, 1]

    Returns
    -------
    np.ndarray
        Basis function derivative values [dphi_0/dxi, dphi_1/dxi, ...] at xi
    """
    if p == 0:
        return np.array([0.0])
    elif p == 1:
        return np.array([0.0, 1.0])
    elif p == 2:
        return np.array([0.0, 1.0, 3.0 * xi])
    else:
        raise ValueError(f"DG order p={p} not supported (max p=2)")


def get_mass_matrix(p: int) -> np.ndarray:
    """
    Get the mass matrix for Legendre basis.

    M_ij = integral_{-1}^{1} phi_i * phi_j dxi

    For Legendre polynomials, this is diagonal: M_ii = 2/(2i+1)

    Parameters
    ----------
    p : int
        Polynomial order

    Returns
    -------
    np.ndarray
        Mass matrix (diagonal for Legendre basis)
    """
    n_dof = p + 1
    M = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        M[i, i] = 2.0 / (2.0 * i + 1.0)
    return M


def get_mass_matrix_inv(p: int) -> np.ndarray:
    """
    Get the inverse mass matrix for Legendre basis.

    Parameters
    ----------
    p : int
        Polynomial order

    Returns
    -------
    np.ndarray
        Inverse mass matrix
    """
    n_dof = p + 1
    M_inv = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        M_inv[i, i] = (2.0 * i + 1.0) / 2.0
    return M_inv


def get_stiffness_matrix(p: int) -> np.ndarray:
    """
    Get the stiffness matrix for Legendre basis.

    S_ij = integral_{-1}^{1} phi_i * dphi_j/dxi dxi

    Parameters
    ----------
    p : int
        Polynomial order

    Returns
    -------
    np.ndarray
        Stiffness matrix
    """
    n_dof = p + 1

    # Use Gauss-Legendre quadrature with enough points
    n_quad = p + 2
    xi_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)

    S = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            for q in range(n_quad):
                phi_i = legendre_basis(p, xi_quad[q])[i]
                dphi_j = legendre_basis_deriv(p, xi_quad[q])[j]
                S[i, j] += w_quad[q] * phi_i * dphi_j

    return S


def evaluate_solution(W_coeffs: np.ndarray, p: int, xi: float) -> np.ndarray:
    """
    Evaluate the DG solution at a point within a cell.

    Parameters
    ----------
    W_coeffs : np.ndarray
        DG coefficients [n_var, n_dof] for this cell
    p : int
        Polynomial order
    xi : float
        Local coordinate in [-1, 1]

    Returns
    -------
    np.ndarray
        Conserved variables W at point xi
    """
    phi = legendre_basis(p, xi)
    return W_coeffs @ phi


def compute_physical_flux(n_dim: int, gam: float, W: np.ndarray) -> np.ndarray:
    """
    Compute the physical flux F(W) for the Euler equations.

    Parameters
    ----------
    n_dim : int
        Number of dimensions
    gam : float
        Specific heat ratio
    W : np.ndarray
        Conserved variables [rho, rho*u, E]

    Returns
    -------
    np.ndarray
        Physical flux F(W)
    """
    rho = W[0]
    u = W[1] / rho
    p = pres(n_dim, gam, W)
    E = W[2]

    F = np.array([
        rho * u,
        rho * u**2 + p,
        u * (E + p)
    ])
    return F


def dg_perfect(prm: 'Parameters', fld: int, dt: float, X: np.ndarray,
               phi: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Advance one time step using Discontinuous Galerkin method for perfect gas.

    This solver implements the DG weak form:
        M * dW/dt = S^T * F(W) - [F* * phi]_boundaries

    where:
        M = mass matrix
        S = stiffness matrix
        F* = numerical flux (Roe)

    For p=0, this reduces exactly to the finite volume method.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters (includes dg_order)
    fld : int
        Fluid index (0 or 1)
    dt : float
        Time step
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function
    U : np.ndarray
        Primitive variables [rho, u, p]^T
    W : np.ndarray
        Conserved variables [rho, rho*u, E]^T

    Returns
    -------
    np.ndarray
        Updated conserved variables W
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n
    dx = prm.dx
    c_EoS = prm.c_EoS
    flg_BCs = prm.flg_BCs
    p = prm.dg_order  # Polynomial order

    gam = c_EoS[fld]  # Specific heat ratio
    n_dof = p + 1     # Degrees of freedom per cell

    if n_dim != 1:
        raise NotImplementedError("DG solver currently only supports 1D")

    # Get DG matrices
    M_inv = get_mass_matrix_inv(p)

    # Basis function values at cell boundaries
    # For p=0: phi_left = phi_right = [1.0], so surface term simplifies to (F_right - F_left)
    phi_left = legendre_basis(p, -1.0)   # xi = -1 (left boundary)
    phi_right = legendre_basis(p, 1.0)   # xi = +1 (right boundary)

    # Initialize DG coefficients from cell averages
    W_dg = np.zeros((n_var, n_dof, n))
    W_dg[:, 0, :] = W[:, :]  # Cell averages are the first coefficient

    # Quadrature points for volume integral
    n_quad = max(p + 1, 2)
    xi_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)

    # ===== Step 1: Compute fluxes at cell interfaces (matching FVM indexing) =====
    # F_iph[:, i] = flux at interface between cell i and cell i+1
    # This matches the FVM convention exactly
    F_iph = np.zeros((n_var, n))

    for i in range(n):
        i_l = i
        i_r = i + 1
        if i == n - 1:  # Wrap for periodic condition
            i_r = 0

        # Evaluate DG solution at interface
        W_L = evaluate_solution(W_dg[:, :, i_l], p, 1.0)   # Right edge of left cell
        W_R = evaluate_solution(W_dg[:, :, i_r], p, -1.0)  # Left edge of right cell

        F_iph[:, i] = roe_flux1D(n_dim, gam, W_L, W_R)

    # ===== Step 2: Update each cell (matching FVM boundary handling) =====
    W_dg_new = np.zeros_like(W_dg)

    for i in range(n):
        # Determine flux indices (matching FVM exactly)
        i_l = i - 1  # Index for left flux F_{i-1/2}
        i_r = i      # Index for right flux F_{i+1/2}

        # Boundary conditions (matching FVM)
        if i == 0:  # Left end
            bc = int(flg_BCs[0])
            if bc == 0:  # Dirichlet
                i_l = 0
                i_r = 0
            elif bc == 1:  # Neumann
                i_l = 1
                i_r = 0
            elif bc == 2:  # Periodic
                i_l = n - 1

        if i == n - 1:  # Right end
            bc = int(flg_BCs[1])
            if bc == 0:  # Dirichlet
                i_l = 0
                i_r = 0
            elif bc == 1:  # Neumann
                i_l = n - 2
                i_r = n - 3
            elif bc == 2:  # Periodic
                i_l = n - 1
                i_r = 0

        # Get fluxes
        F_left = F_iph[:, i_l]   # Flux at left interface
        F_right = F_iph[:, i_r]  # Flux at right interface

        # Surface term: F*_right * phi(+1) - F*_left * phi(-1)
        # For p=0: phi_left = phi_right = [1], so this is just (F_right - F_left)
        surface_term = np.zeros((n_var, n_dof))
        for k in range(n_var):
            surface_term[k, :] = F_right[k] * phi_right - F_left[k] * phi_left

        # Volume integral (zero for p=0)
        volume_term = np.zeros((n_var, n_dof))

        if p > 0:
            for q in range(n_quad):
                xi = xi_quad[q]
                w = w_quad[q]

                W_q = evaluate_solution(W_dg[:, :, i], p, xi)
                F_q = compute_physical_flux(n_dim, gam, W_q)
                dphi = legendre_basis_deriv(p, xi)

                for k in range(n_var):
                    volume_term[k, :] += w * F_q[k] * dphi

        # DG update
        # For p=0: M_inv = [0.5], jacobian = 2/dx
        # So update is: W_new = W_old - dt/dx * (F_right - F_left)
        # This matches FVM exactly!
        jacobian = 2.0 / dx

        for k in range(n_var):
            residual = volume_term[k, :] - surface_term[k, :]
            dW = M_inv @ residual * jacobian
            W_dg_new[k, :, i] = W_dg[k, :, i] + dt * dW

    # Extract cell averages
    W_new = W_dg_new[:, 0, :]

    return W_new


def dg_perfect_full(prm: 'Parameters', fld: int, dt: float, X: np.ndarray,
                    phi: np.ndarray, U: np.ndarray, W: np.ndarray,
                    W_dg: np.ndarray = None) -> tuple:
    """
    Full DG solver that returns DG coefficients (for higher-order analysis).

    This version returns the full DG representation, not just cell averages.
    Useful for detailed comparison and visualization of the polynomial solution.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    fld : int
        Fluid index
    dt : float
        Time step
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function
    U : np.ndarray
        Primitive variables
    W : np.ndarray
        Conserved variables (cell averages)
    W_dg : np.ndarray, optional
        Full DG coefficients from previous step

    Returns
    -------
    tuple
        (W_new, W_dg_new) - cell averages and full DG coefficients
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n
    dx = prm.dx
    c_EoS = prm.c_EoS
    flg_BCs = prm.flg_BCs
    p = prm.dg_order

    gam = c_EoS[fld]
    n_dof = p + 1

    if n_dim != 1:
        raise NotImplementedError("DG solver currently only supports 1D")

    # Get DG matrices
    M_inv = get_mass_matrix_inv(p)

    # Basis function values at cell boundaries
    phi_left = legendre_basis(p, -1.0)
    phi_right = legendre_basis(p, 1.0)

    # Initialize or use provided DG coefficients
    if W_dg is None:
        W_dg = np.zeros((n_var, n_dof, n))
        W_dg[:, 0, :] = W[:, :]
    else:
        W_dg = W_dg.copy()

    # Quadrature
    n_quad = max(p + 1, 2)
    xi_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)

    # Compute interface fluxes (matching FVM indexing)
    F_iph = np.zeros((n_var, n))

    for i in range(n):
        i_l = i
        i_r = i + 1
        if i == n - 1:
            i_r = 0

        W_L = evaluate_solution(W_dg[:, :, i_l], p, 1.0)
        W_R = evaluate_solution(W_dg[:, :, i_r], p, -1.0)
        F_iph[:, i] = roe_flux1D(n_dim, gam, W_L, W_R)

    # Update cells (matching FVM boundary handling)
    W_dg_new = np.zeros_like(W_dg)

    for i in range(n):
        i_l = i - 1
        i_r = i

        if i == 0:
            bc = int(flg_BCs[0])
            if bc == 0:
                i_l = 0
                i_r = 0
            elif bc == 1:
                i_l = 1
                i_r = 0
            elif bc == 2:
                i_l = n - 1

        if i == n - 1:
            bc = int(flg_BCs[1])
            if bc == 0:
                i_l = 0
                i_r = 0
            elif bc == 1:
                i_l = n - 2
                i_r = n - 3
            elif bc == 2:
                i_l = n - 1
                i_r = 0

        F_left = F_iph[:, i_l]
        F_right = F_iph[:, i_r]

        surface_term = np.zeros((n_var, n_dof))
        for k in range(n_var):
            surface_term[k, :] = F_right[k] * phi_right - F_left[k] * phi_left

        volume_term = np.zeros((n_var, n_dof))

        if p > 0:
            for q in range(n_quad):
                xi = xi_quad[q]
                w = w_quad[q]
                W_q = evaluate_solution(W_dg[:, :, i], p, xi)
                F_q = compute_physical_flux(n_dim, gam, W_q)
                dphi = legendre_basis_deriv(p, xi)
                for k in range(n_var):
                    volume_term[k, :] += w * F_q[k] * dphi

        jacobian = 2.0 / dx

        for k in range(n_var):
            residual = volume_term[k, :] - surface_term[k, :]
            dW = M_inv @ residual * jacobian
            W_dg_new[k, :, i] = W_dg[k, :, i] + dt * dW

    W_new = W_dg_new[:, 0, :]

    return W_new, W_dg_new
