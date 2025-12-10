"""
Roe's approximate Riemann solver for perfect gas.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters

from .flux import roe_flux1D, roe_flux2D


def roe_perfect(prm: 'Parameters', fld: int, dt: float, X: np.ndarray,
                phi: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Advance one time step of compressible fluid in conserved variables W.

    Method: Roe's Approximate 1st-order. Updates all grid points with
    dimensional splitting when n_dim > 1.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    fld : int
        Fluid index (0 or 1)
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
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n
    dx = prm.dx
    c_EoS = prm.c_EoS
    flg_BCs = prm.flg_BCs

    gam = c_EoS[fld]  # Specific heat ratio

    if n_dim == 1:
        # ===== 1D Case =====
        F_iph = np.zeros((n_var, n))  # Allocate flux vector

        # Calculate flux at each interface
        for i in range(n):
            i_l = i
            i_r = i + 1
            if i == n - 1:  # Compute i=n flux for periodic condition
                i_r = 0
            F_iph[:, i] = roe_flux1D(n_dim, gam, W[:, i_l], W[:, i_r])

        # Update conserved variables
        for i in range(n):
            i_l = i - 1
            i_r = i

            # Boundary conditions
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

            # Update step
            for k in range(n_var):
                W[k, i] = W[k, i] - dt / dx * (F_iph[k, i_r] - F_iph[k, i_l])

    elif n_dim == 2:
        # ===== 2D Case =====
        n = np.atleast_1d(n)
        dx = np.atleast_1d(dx)

        # --- x-Sweep ---
        for j in range(n[1]):
            F_slc = np.zeros((n_var, n[0]))  # Allocate flux vector
            W_slc = W[:, :, j].copy()  # Slice in y-direction

            # Calculate x-flux F_(i+1/2,x)
            for i in range(n[0]):
                i_l = i
                i_r = i + 1
                if i == n[0] - 1:
                    i_r = 0  # Periodic condition
                F_slc[:, i] = roe_flux2D(n_dim, 2, gam, W_slc[:, i_l], W_slc[:, i_r])

            # Update with boundary conditions
            for i in range(n[0]):
                i_l = i - 1
                i_r = i

                if i == 0:  # Left edge
                    bc = int(flg_BCs[0])
                    if bc == 0:  # Dirichlet
                        i_l = 0
                        i_r = 0
                    elif bc == 1:  # Neumann
                        i_l = 1
                        i_r = 0
                    elif bc == 2:  # Periodic
                        i_l = n[0] - 1

                if i == n[0] - 1:  # Right edge
                    bc = int(flg_BCs[2])
                    if bc == 0:  # Dirichlet
                        i_l = 0
                        i_r = 0
                    elif bc == 1:  # Neumann
                        i_l = i - 1
                        i_r = i - 2
                    elif bc == 2:  # Periodic
                        i_l = i
                        i_r = 0

                # Update variables
                for k in range(n_var):
                    W_slc[k, i] = W_slc[k, i] - dt / dx[1] * (F_slc[k, i_r] - F_slc[k, i_l])

            W[:, :, j] = W_slc  # Pass updated slice

        # --- y-Sweep ---
        for i in range(n[0]):
            F_slc = np.zeros((n_var, n[1]))  # Allocate flux vector
            W_slc = W[:, i, :].copy()

            # Calculate y-flux F_(i+1/2,y)
            for j in range(n[1]):
                j_l = j
                j_r = j + 1
                if j == n[1] - 1:
                    j_r = 0  # Periodic condition
                F_slc[:, j] = roe_flux2D(n_dim, 1, gam, W_slc[:, j_l], W_slc[:, j_r])

            # Update with boundary conditions
            for j in range(n[1]):
                j_l = j - 1
                j_r = j

                if j == n[1] - 1:  # Top edge
                    bc = int(flg_BCs[1])
                    if bc == 0:  # Dirichlet
                        j_l = 0
                        j_r = 0
                    elif bc == 1:  # Neumann
                        j_l = j - 1
                        j_r = j - 2
                    elif bc == 2:  # Periodic
                        j_l = j
                        j_r = 0

                if j == 0:  # Bottom edge
                    bc = int(flg_BCs[3])
                    if bc == 0:  # Dirichlet
                        j_l = 0
                        j_r = 0
                    elif bc == 1:  # Neumann
                        j_l = 1
                        j_r = 1
                    elif bc == 2:  # Periodic
                        j_l = n[1] - 1

                # Update variables
                for k in range(n_var):
                    W_slc[k, j] = W_slc[k, j] - dt / dx[0] * (F_slc[k, j_r] - F_slc[k, j_l])

            W[:, i, :] = W_slc

    return W
