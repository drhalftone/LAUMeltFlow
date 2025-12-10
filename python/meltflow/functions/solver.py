"""
Solver functions for MeltFlow.

Includes time stepping and solver dispatch.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters

from .roe_perfect import roe_perfect
from .ghost_fluid import extrp
from .state_var import state_var


def timestep(prm: 'Parameters', U: np.ndarray, a: np.ndarray) -> float:
    """
    Compute time step dt from Euler primitive variables U.

    Method: Uses CFL number with maximum grid point velocity magnitude.
    Equation: dt = CFL * dx / (|V|_max + a)

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T
    a : np.ndarray
        Speed of sound

    Returns
    -------
    float
        Time step dt
    """
    n_dim = prm.n_dim
    n = prm.n
    dx = prm.dx
    cfl = prm.cfl

    if n_dim == 1:
        # 1D Case
        dt = 0.5 * cfl * dx / np.max(np.abs(U[1, :]) + a)

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        dx = np.atleast_1d(dx)
        V_norm = np.zeros((n[0], n[1]))
        for i in range(n[0]):
            for j in range(n[1]):
                V_norm[i, j] = np.linalg.norm([U[1, i, j], U[2, i, j]])
        dt = 0.5 * cfl * np.min(dx) / np.max(np.abs(V_norm) + a)

    return dt


def none_solver(prm: 'Parameters', fld: int, dt: float, X: np.ndarray,
                phi: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    No-op solver for fluids that don't need solving (e.g., pure liquid).

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
        Conserved variables

    Returns
    -------
    np.ndarray
        Unchanged conserved variables W
    """
    return W


def incomp_1D(prm: 'Parameters', fld: int, dt: float, X: np.ndarray,
              phi: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Advance one time step of incompressible fluid in conserved variables W.

    Method: Simple linear interpolation of pressure in 1D.

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
        Primitive variables [rho, u, (v), p]^T
    W : np.ndarray
        Conserved variables [rho, rho*u, (rho*v), E]^T

    Returns
    -------
    np.ndarray
        Updated conserved variables W
    """
    n_var = prm.n_var
    n = prm.n

    U = U.copy()
    i_l = 0
    i_r = 0
    p_l = 0
    p_r = 0

    # Find liquid regions
    for i in range(1, n - 1):
        if phi[i + 1] <= 0 and phi[i] > 0:
            i_l = i + 1  # Left-most liquid point
            p_l = U[2, i]  # Pressure at interface

        if phi[i - 1] <= 0 and phi[i] > 0:
            i_r = i - 1  # Right-most liquid point
            p_r = U[2, i]  # Pressure at interface

            if i_l != i_r:  # More than one liquid point
                pstr_l = dt / U[0, i_l] * U[2, i_l]
                pstr_r = dt / U[0, i_r] * U[2, i_r]

                # Interpolate liquid pressure
                for j in range(i_l, i_r + 1):
                    U[2, j] = p_l + (p_r - p_l) / (i_r - i_l) * (j - i_l)
                    U[1, j] = U[1, j] - (pstr_r - pstr_l) / (X[i_r] - X[i_l])

    # Extrapolate density/velocity to outer field
    U[0, :] = extrp(2, prm, X, phi, U[0, :])
    U[1, :] = extrp(2, prm, X, phi, U[1, :])

    # Calculate conserved variables
    W = state_var(prm, "cons", n_var, phi, U)

    return W


# Solver dispatch table
SOLVERS = {
    'roe_perfect': roe_perfect,
    'none': none_solver,
    'incomp_1D': incomp_1D,
}


def run_slvr(prm: 'Parameters', dt: float, X: np.ndarray, phi: np.ndarray,
             UU: np.ndarray, WW: np.ndarray) -> np.ndarray:
    """
    Pass primitive and conserved variables to solver for time step updates.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    dt : float
        Time step
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function
    UU : np.ndarray
        Primitive variables for both fluids [2, n_var, ...]
    WW : np.ndarray
        Conserved variables for both fluids [2, n_var, ...]

    Returns
    -------
    np.ndarray
        Updated conserved variables WW
    """
    n_dim = prm.n_dim
    slvr = prm.slvr

    for k in range(2):
        solver_name = slvr[k]
        solver_func = SOLVERS.get(solver_name, none_solver)

        if n_dim == 1:
            U = np.squeeze(UU[k, :, :])
            W = np.squeeze(WW[k, :, :])
            WW[k, :, :] = solver_func(prm, k, dt, X, phi, U, W)
        elif n_dim == 2:
            U = np.squeeze(UU[k, :, :, :])
            W = np.squeeze(WW[k, :, :, :])
            WW[k, :, :, :] = solver_func(prm, k, dt, X, phi, U, W)

    return WW
