"""
Ghost Fluid Method (GFM) implementation for multi-phase flows.
"""

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters

from .thermodynamics import entrp_perfect
from .state_var import state_var


def extrp(fld: int, prm: 'Parameters', X: np.ndarray, phi: np.ndarray,
          I: np.ndarray) -> np.ndarray:
    """
    Extrapolate (or interpolate) a data set I from one fluid domain to the other.

    Parameters
    ----------
    fld : int
        Which fluid domain to extrapolate from (1 or 2)
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function
    I : np.ndarray
        Data to extrapolate

    Returns
    -------
    np.ndarray
        Extrapolated data over entire grid
    """
    n_dim = prm.n_dim
    n = prm.n

    if n_dim == 1:
        # 1D Case
        x_in = []
        I_in = []
        for i in range(n):
            if (fld == 1 and phi[i] > 0) or (fld == 2 and phi[i] <= 0):
                x_in.append(X[i])
                I_in.append(I[i])

        if len(x_in) > 0:
            x_in = np.array(x_in)
            I_in = np.array(I_in)
            # Sort by x for interpolation
            sort_idx = np.argsort(x_in)
            x_in = x_in[sort_idx]
            I_in = I_in[sort_idx]
            F = interp1d(x_in, I_in, kind='linear', fill_value='extrapolate')
            I = F(X)

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        x_in = []
        y_in = []
        I_in = []
        for i in range(n[0]):
            for j in range(n[1]):
                if (fld == 1 and phi[i, j] > 0) or (fld == 2 and phi[i, j] <= 0):
                    x_in.append(X[0, i, j])
                    y_in.append(X[1, i, j])
                    I_in.append(I[i, j])

        if len(x_in) > 0:
            x_in = np.array(x_in)
            y_in = np.array(y_in)
            I_in = np.array(I_in)
            points = np.column_stack([x_in, y_in])
            F = LinearNDInterpolator(points, I_in)
            X_grid = X[0, :, :]
            Y_grid = X[1, :, :]
            I = F(X_grid, Y_grid)
            # Handle NaN values from extrapolation outside convex hull
            I = np.nan_to_num(I, nan=0.0)

    return I


def extrp_vel(prm: 'Parameters', X: np.ndarray, phi: np.ndarray,
              WW: np.ndarray) -> np.ndarray:
    """
    Extrapolate velocity from fluid 2 domain (phi <= 0) to fluid 1 domain.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function
    WW : np.ndarray
        Conserved variables for both fluids [2, n_var, ...]

    Returns
    -------
    np.ndarray
        Velocity field V
    """
    n_dim = prm.n_dim
    n = prm.n

    if n_dim == 1:
        # 1D Case
        u = np.zeros(n)
        for i in range(n):
            if phi[i] <= 0:
                u[i] = WW[1, 1, i] / WW[1, 0, i]  # Velocity from fluid 2
            else:
                u[i] = WW[0, 1, i] / WW[0, 0, i]  # Velocity from fluid 1
        V = u

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        u = np.zeros((n[0], n[1]))
        v = np.zeros((n[0], n[1]))
        for i in range(n[0]):
            for j in range(n[1]):
                if phi[i, j] <= 0:
                    u[i, j] = WW[1, 1, i, j] / WW[1, 0, i, j]
                    v[i, j] = WW[1, 2, i, j] / WW[1, 0, i, j]
                else:
                    u[i, j] = WW[0, 1, i, j] / WW[0, 0, i, j]
                    v[i, j] = WW[0, 2, i, j] / WW[0, 0, i, j]
        V = np.zeros((2, n[0], n[1]))
        V[0, :, :] = u
        V[1, :, :] = v

    return V


def ghost_GFM(prm: 'Parameters', X: np.ndarray, phi: np.ndarray,
              U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct two domains as per the ghost-fluid method for gas-liquid systems.

    Applies Rankine-Hugoniot jump conditions.

    Method:
    - Ghost gas: Copies velocities u and pressure p from real liquid,
      extrapolates entropy s from adjacent real gas.
    - Ghost liquid: Copies velocities u and pressure p from real gas,
      extrapolates entropy s from adjacent real gas.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    phi : np.ndarray
        Level set function (positive = fluid 1, negative = fluid 2)
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T

    Returns
    -------
    tuple
        (UU, WW) - Primitive and conserved variables for both fluids
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n
    flg_fld = prm.flg_fld
    c_EoS = prm.c_EoS

    if n_dim == 1:
        # ===== 1D Case =====
        UU = np.zeros((2, n_var, n))  # Allocate new domains

        # ----- Extrapolate -----
        # Fluid 1
        if flg_fld[0] == 0:  # Gas
            s = np.zeros(n)
            for i in range(n):
                if phi[i] > 0:  # Compute entropy in gas regions
                    s[i] = entrp_perfect(0, n_dim, c_EoS[0], U[:, i], s[i])
            s = extrp(1, prm, X, phi, s)
            for i in range(n):
                if phi[i] <= 0:  # Invert entropy -> density
                    UU[0, 0, i] = entrp_perfect(1, n_dim, c_EoS[0], U[:, i], s[i])

        # Fluid 2
        if flg_fld[1] == 0:  # Gas
            s = np.zeros(n)
            for i in range(n):
                if phi[i] <= 0:  # Compute entropy in gas regions
                    s[i] = entrp_perfect(0, n_dim, c_EoS[1], U[:, i], s[i])
            s = extrp(2, prm, X, phi, s)
            for i in range(n):
                if phi[i] > 0:  # Invert entropy -> density
                    UU[1, 0, i] = entrp_perfect(1, n_dim, c_EoS[1], U[:, i], s[i])

        # ----- Copy -----
        for i in range(n):
            if phi[i] > 0:  # Fluid 1 region
                UU[0, :, i] = U[:, i]  # Copy real fluid 1 -> fluid 1
                UU[1, 1:3, i] = U[1:3, i]  # Copy velocity, pressure -> ghost fluid 2
            elif phi[i] <= 0:  # Fluid 2 region
                UU[1, :, i] = U[:, i]  # Copy real fluid 2 -> fluid 2
                UU[0, 1:3, i] = U[1:3, i]  # Copy velocity, pressure -> ghost fluid 1

        # Compute conserved variables
        WW = np.zeros((2, n_var, n))
        WW[0, :, :] = state_var(prm, "cons", n_var, phi, UU[0, :, :])
        WW[1, :, :] = state_var(prm, "cons", n_var, phi, UU[1, :, :])

    elif n_dim == 2:
        # ===== 2D Case =====
        n = np.atleast_1d(n)
        UU = np.zeros((2, n_var, n[0], n[1]))  # Allocate gas and liquid variables

        # ----- Extrapolate -----
        # Fluid 1
        if flg_fld[0] == 0:  # Gas
            s = np.zeros((n[0], n[1]))
            for i in range(n[0]):
                for j in range(n[1]):
                    if phi[i, j] > 0:  # Compute entropy in real gas 1
                        s[i, j] = entrp_perfect(0, n_dim, c_EoS[0], U[:, i, j], s[i, j])
            s = extrp(1, prm, X, phi, s)
            for i in range(n[0]):
                for j in range(n[1]):
                    if phi[i, j] <= 0:  # Invert entropy -> density
                        UU[0, 0, i, j] = entrp_perfect(1, n_dim, c_EoS[0], U[:, i, j], s[i, j])

        # Fluid 2
        if flg_fld[1] == 0:  # Gas
            s = np.zeros((n[0], n[1]))
            for i in range(n[0]):
                for j in range(n[1]):
                    if phi[i, j] <= 0:  # Compute entropy in real gas 2
                        s[i, j] = entrp_perfect(0, n_dim, c_EoS[1], U[:, i, j], s[i, j])
            s = extrp(2, prm, X, phi, s)
            for i in range(n[0]):
                for j in range(n[1]):
                    if phi[i, j] > 0:  # Invert entropy -> density
                        UU[1, 0, i, j] = entrp_perfect(1, n_dim, c_EoS[1], U[:, i, j], s[i, j])

        # ----- Copying -----
        for i in range(n[0]):
            for j in range(n[1]):
                if phi[i, j] > 0:  # Gas regions
                    UU[0, :, i, j] = U[:, i, j]  # Copy real fluid 1 -> fluid 1
                    UU[1, 1:4, i, j] = U[1:4, i, j]  # Copy velocity, pressure -> ghost fluid 2
                elif phi[i, j] <= 0:  # Fluid 2 region
                    UU[1, :, i, j] = U[:, i, j]  # Copy real fluid 2 -> fluid 2
                    UU[0, 1:4, i, j] = U[1:4, i, j]  # Copy velocity, pressure -> ghost fluid 1

        # Compute conserved variables
        WW = np.zeros((2, n_var, n[0], n[1]))
        WW[0, :, :, :] = state_var(prm, "cons", n_var, phi, np.squeeze(UU[0, :, :, :]))
        WW[1, :, :, :] = state_var(prm, "cons", n_var, phi, np.squeeze(UU[1, :, :, :]))

    return UU, WW


def real_GFM(prm: 'Parameters', phi: np.ndarray, WW: np.ndarray) -> np.ndarray:
    """
    Construct one real domain from two fluid domains as per ghost-fluid method.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    phi : np.ndarray
        Level set function (positive = fluid 1, negative = fluid 2)
    WW : np.ndarray
        Conserved variables for both fluids [2, n_var, ...]

    Returns
    -------
    np.ndarray
        Single real domain conserved variables W
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n

    if n_dim == 1:
        # 1D Case
        W = np.zeros((n_var, n))
        for i in range(n):
            if phi[i] > 0:  # Fluid 1 regions
                W[:, i] = WW[0, :, i]
            elif phi[i] <= 0:  # Fluid 2 regions
                W[:, i] = WW[1, :, i]

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        W = np.zeros((n_var, n[0], n[1]))
        for i in range(n[0]):
            for j in range(n[1]):
                if phi[i, j] > 0:
                    W[:, i, j] = WW[0, :, i, j]
                elif phi[i, j] <= 0:
                    W[:, i, j] = WW[1, :, i, j]

    return W
