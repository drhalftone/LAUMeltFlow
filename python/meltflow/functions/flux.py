"""
Flux computation functions for Roe's approximate Riemann solver.
"""

import numpy as np
from .thermodynamics import pres


def roe_avg(rho_L: float, rho_R: float, r_L: float, r_R: float) -> float:
    """
    Calculate the Roe-averaged value of a property r.

    Parameters
    ----------
    rho_L, rho_R : float
        Densities on left and right sides
    r_L, r_R : float
        Property values on left and right sides

    Returns
    -------
    float
        Roe-averaged value
    """
    return (np.sqrt(rho_L) * r_L + np.sqrt(rho_R) * r_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))


def roe_spd(gam: float, h_roe: float, u_roe: float, v_roe: float = 0.0) -> float:
    """
    Calculate the Roe-averaged speed of sound.

    Parameters
    ----------
    gam : float
        Specific heat ratio
    h_roe : float
        Roe-averaged enthalpy
    u_roe : float
        Roe-averaged x-velocity
    v_roe : float, optional
        Roe-averaged y-velocity (default 0 for 1D)

    Returns
    -------
    float
        Roe-averaged speed of sound
    """
    return np.sqrt((gam - 1) * (h_roe - 0.5 * (u_roe**2 + v_roe**2)))


def roe_flux1D(n_dim: int, gam: float, W_L: np.ndarray, W_R: np.ndarray) -> np.ndarray:
    """
    Calculate flux F_(i+1/2) for 1D Roe's approximate solver.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (should be 1)
    gam : float
        Specific heat ratio
    W_L : np.ndarray
        Conserved variables to left of Riemann problem [rho, rho*u, E]
    W_R : np.ndarray
        Conserved variables to right of Riemann problem [rho, rho*u, E]

    Returns
    -------
    np.ndarray
        Flux at interface F_(i+1/2)
    """
    n_var = n_dim + 2  # Number of variables

    # Pre-Processing
    rho_L, rho_R = W_L[0], W_R[0]           # Densities
    u_L = W_L[1] / W_L[0]                    # Velocities
    u_R = W_R[1] / W_R[0]
    p_L = pres(n_dim, gam, W_L)             # Pressures
    p_R = pres(n_dim, gam, W_R)
    E_L, E_R = W_L[2], W_R[2]               # Total energies
    h_L = (E_L + p_L) / rho_L               # Enthalpies
    h_R = (E_R + p_R) / rho_R
    drho = rho_R - rho_L                    # Primitive variable differences
    du = u_R - u_L
    dp = p_R - p_L

    # Left and Right-side Flux
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (E_L + p_L)
    ])
    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (E_R + p_R)
    ])

    # Roe-Averages
    rho = np.sqrt(rho_R * rho_L)            # Roe-averaged density
    u = roe_avg(rho_L, rho_R, u_L, u_R)     # Roe-averaged velocity
    h = roe_avg(rho_L, rho_R, h_L, h_R)     # Roe-averaged enthalpy
    a = np.sqrt((gam - 1) * (h - 0.5 * u**2))  # Roe-averaged speed of sound

    # Eigenvalues
    lmda = np.array([u, u + a, u - a])

    # Wave strengths
    dv = np.array([
        drho - dp / a**2,
        du + dp / (rho * a),
        du - dp / (rho * a)
    ])

    # Eigenvectors
    r = np.array([
        [1, u, 0.5 * u**2],
        rho / (2 * a) * np.array([1, u + a, h + a * u]),
        -rho / (2 * a) * np.array([1, u - a, h - a * u])
    ])

    # Flux calculation
    flux_sum = np.zeros(n_var)
    for j in range(n_var):
        flux_sum += dv[j] * np.abs(lmda[j]) * r[j, :]

    F_iph = 0.5 * (F_L + F_R) - 0.5 * flux_sum

    return F_iph


def roe_flux2D(n_dim: int, dim: int, gam: float, W_L: np.ndarray, W_R: np.ndarray) -> np.ndarray:
    """
    Calculate flux F_(i+1/2) for 2D Roe's approximate solver.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (should be 2)
    dim : int
        Sweep direction: 1 for x-sweep, 2 for y-sweep
    gam : float
        Specific heat ratio
    W_L : np.ndarray
        Conserved variables to left of Riemann problem [rho, rho*u, rho*v, E]
    W_R : np.ndarray
        Conserved variables to right of Riemann problem [rho, rho*u, rho*v, E]

    Returns
    -------
    np.ndarray
        Flux at interface F_(i+1/2)
    """
    n_var = n_dim + 2  # Number of variables

    # Pre-Processing
    rho_L, rho_R = W_L[0], W_R[0]           # Densities
    u_L = W_L[1] / W_L[0]                    # x-velocities
    u_R = W_R[1] / W_R[0]
    v_L = W_L[2] / W_L[0]                    # y-velocities
    v_R = W_R[2] / W_R[0]
    p_L = pres(n_dim, gam, W_L)             # Pressures
    p_R = pres(n_dim, gam, W_R)
    E_L, E_R = W_L[3], W_R[3]               # Total energies
    h_L = (E_L + p_L) / rho_L               # Enthalpies
    h_R = (E_R + p_R) / rho_R
    dW = W_R - W_L                          # Difference in conserved variables

    # Roe-Averages
    u = roe_avg(rho_L, rho_R, u_L, u_R)     # Roe-averaged x-velocity
    v = roe_avg(rho_L, rho_R, v_L, v_R)     # Roe-averaged y-velocity
    h = roe_avg(rho_L, rho_R, h_L, h_R)     # Roe-averaged enthalpy
    a = roe_spd(gam, h, u, v)               # Roe-averaged speed of sound
    V_2 = u**2 + v**2                       # Roe-averaged squared velocity

    if dim == 1:
        # x-Sweep (actually y-direction flux in MATLAB code)
        F_L = np.array([
            rho_L * v_L,
            rho_L * u_L * v_L,
            rho_L * v_L**2 + p_L,
            v_L * (E_L + p_L)
        ])
        F_R = np.array([
            rho_R * v_R,
            rho_R * u_R * v_R,
            rho_R * v_R**2 + p_R,
            v_R * (E_R + p_R)
        ])

        # Eigenvalues
        lmda = np.array([v - a, v, v, v + a])

        # Eigenvectors
        K = np.array([
            [1, u, v - a, h - a * v],
            [1, u, v, 0.5 * V_2],
            [0, 1, 0, u],
            [1, u, v + a, h + v * a]
        ])

        # Wave strengths coefficients
        c = np.zeros(7)
        c[0] = h - a * v
        c[1] = h + a * v
        c[2] = v - a
        c[3] = v + a
        c[4] = 0.5 * V_2
        c[5] = c[0] * c[3] - c[1] * c[2] + c[2] * c[4] - c[3] * c[4] - c[0] * v + c[1] * v
        c[6] = dW[3] - dW[1] * u + u**2 * dW[0]

        alph = np.zeros(4)
        alph[0] = -1 / c[5] * ((c[3] * c[4] - c[1] * v) * dW[0]
                              + (c[1] - c[4]) * dW[2] + (v - c[3]) * c[6])
        alph[1] = -1 / c[5] * ((c[1] * c[2] - c[0] * c[3]) * dW[0]
                              + (c[0] - c[1]) * dW[2] + (c[3] - c[2]) * c[6])
        alph[2] = dW[1] - u * dW[0]
        alph[3] = 1 / c[5] * ((c[2] * c[4] - c[0] * v) * dW[0]
                             + (c[0] - c[4]) * dW[2] + (v - c[2]) * c[6])

    elif dim == 2:
        # y-Sweep (actually x-direction flux in MATLAB code)
        F_L = np.array([
            rho_L * u_L,
            rho_L * u_L**2 + p_L,
            rho_L * u_L * v_L,
            u_L * (E_L + p_L)
        ])
        F_R = np.array([
            rho_R * u_R,
            rho_R * u_R**2 + p_R,
            rho_R * u_R * v_R,
            u_R * (E_R + p_R)
        ])

        # Eigenvalues
        lmda = np.array([u - a, u, u, u + a])

        # Eigenvectors
        K = np.array([
            [1, u - a, v, h - a * u],
            [1, u, v, 0.5 * V_2],
            [0, 0, 1, v],
            [1, u + a, v, h + a * u]
        ])

        # Wave strengths coefficients
        c = np.zeros(7)
        c[0] = h - a * u
        c[1] = h + a * u
        c[2] = u - a
        c[3] = u + a
        c[4] = 0.5 * V_2
        c[5] = c[0] * c[3] - c[1] * c[2] + c[2] * c[4] - c[3] * c[4] - c[0] * u + c[1] * u
        c[6] = dW[3] - dW[2] * v + v**2 * dW[0]

        alph = np.zeros(4)
        alph[0] = -1 / c[5] * ((c[3] * c[4] - c[1] * u) * dW[0]
                              + (c[1] - c[4]) * dW[1] + (u - c[3]) * c[6])
        alph[1] = -1 / c[5] * ((c[1] * c[2] - c[0] * c[3]) * dW[0]
                              + (c[0] - c[1]) * dW[1] + (c[3] - c[2]) * c[6])
        alph[2] = dW[2] - v * dW[0]
        alph[3] = 1 / c[5] * ((c[2] * c[4] - c[0] * u) * dW[0]
                             + (c[0] - c[4]) * dW[1] + (u - c[2]) * c[6])

    # Flux calculation
    flux_sum = np.zeros(n_var)
    for j in range(n_var):
        flux_sum += alph[j] * np.abs(lmda[j]) * K[j, :]

    F_iph = 0.5 * (F_L + F_R) - 0.5 * flux_sum

    return F_iph
