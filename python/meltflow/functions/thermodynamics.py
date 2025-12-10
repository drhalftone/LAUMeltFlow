"""
Thermodynamic functions for perfect gas equation of state.
"""

import numpy as np


def pres(n_dim: int, gam: float, W: np.ndarray) -> float:
    """
    Calculate pressure from conserved variables W.

    Assumption: Perfect gamma-law gas relation in pressure p to total energy E
    Equation: p = (gam-1)*(E - 1/2*rho*(u^2 + v^2))

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (1 or 2)
    gam : float
        Specific heat ratio (gamma)
    W : np.ndarray
        Conserved variables [rho, rho*u, (rho*v), E]^T

    Returns
    -------
    float
        Pressure p
    """
    if n_dim == 1:
        p = (gam - 1) * (W[2] - 0.5 * W[0] * (W[1] / W[0])**2)
    elif n_dim == 2:
        p = (gam - 1) * (W[3] - 0.5 * W[0] * ((W[1] / W[0])**2 + (W[2] / W[0])**2))
    return p


def cons_perfect(n_dim: int, gam: float, U: np.ndarray) -> np.ndarray:
    """
    Compute conserved variables W for a perfect gas from primitive variables U.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (1 or 2)
    gam : float
        Specific heat ratio (gamma)
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T

    Returns
    -------
    np.ndarray
        Conserved variables [rho, rho*u, (rho*v), E]^T
    """
    if n_dim == 1:
        W = np.zeros(3)
        W[0] = U[0]                                    # w_1 = rho
        W[1] = U[0] * U[1]                             # w_2 = rho*u
        W[2] = U[2] / (gam - 1) + 0.5 * U[0] * U[1]**2  # w_3 = E
    elif n_dim == 2:
        W = np.zeros(4)
        W[0] = U[0]                                    # w_1 = rho
        W[1] = U[0] * U[1]                             # w_2 = rho*u
        W[2] = U[0] * U[2]                             # w_3 = rho*v
        W[3] = U[3] / (gam - 1) + 0.5 * U[0] * (U[1]**2 + U[2]**2)  # w_4 = E
    return W


def prim_perfect(n_dim: int, gam: float, W: np.ndarray) -> np.ndarray:
    """
    Compute primitive variables U for a perfect gas from conserved variables W.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (1 or 2)
    gam : float
        Specific heat ratio (gamma)
    W : np.ndarray
        Conserved variables [rho, rho*u, (rho*v), E]^T

    Returns
    -------
    np.ndarray
        Primitive variables [rho, u, (v), p]^T
    """
    if n_dim == 1:
        U = np.zeros(3)
        U[0] = W[0]                                    # u_1 = rho
        U[1] = W[1] / W[0]                             # u_2 = u
        U[2] = (gam - 1) * (W[2] - 0.5 * W[0] * (W[1] / W[0])**2)  # u_3 = p
    elif n_dim == 2:
        U = np.zeros(4)
        U[0] = W[0]                                    # u_1 = rho
        U[1] = W[1] / W[0]                             # u_2 = u
        U[2] = W[2] / W[0]                             # u_3 = v
        U[3] = (gam - 1) * (W[3] - 0.5 * W[0] * ((W[1] / W[0])**2 + (W[2] / W[0])**2))  # u_4 = p
    return U


def SoS_perfect(n_dim: int, gam: float, U: np.ndarray) -> float:
    """
    Compute speed of sound a for a perfect gas.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (1 or 2)
    gam : float
        Specific heat ratio (gamma)
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T

    Returns
    -------
    float
        Speed of sound a
    """
    if n_dim == 1:
        a = np.sqrt(gam * U[2] / U[0])
    elif n_dim == 2:
        a = np.sqrt(gam * U[3] / U[0])
    return a


def entrp_perfect(var: int, n_dim: int, gam: float, U: np.ndarray, s: float) -> float:
    """
    Compute or invert entropy s for a perfect gas.

    Parameters
    ----------
    var : int
        0: Compute entropy s from primitive variables U
        1: Compute density rho from entropy s and other primitive variables
    n_dim : int
        Number of spatial dimensions (1 or 2)
    gam : float
        Specific heat ratio (gamma)
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T
    s : float
        Entropy (used when var=1)

    Returns
    -------
    float
        Entropy (if var=0) or density (if var=1)
    """
    c = 0

    if n_dim == 1:
        if var == 0:  # Compute entropy
            a = np.log(U[2]) - gam * np.log(U[0]) + c
        elif var == 1:  # Compute density
            a = np.exp((np.log(U[2]) - s + c) / gam)
    elif n_dim == 2:
        if var == 0:  # Compute entropy
            a = np.log(U[3]) - gam * np.log(U[0])
        elif var == 1:  # Compute density
            a = np.exp((np.log(U[3]) - s) / gam)
    return a
