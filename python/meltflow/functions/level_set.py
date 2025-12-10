"""
Level set functions for interface tracking.

Includes advection (Godunov upwind scheme) and reinitialization (WENO).
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters


def godunov(n_dim: int, lmda: float, u: np.ndarray, f: np.ndarray) -> float:
    """
    Update a time step for a grid point using Godunov's upwind method.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions
    lmda : float or np.ndarray
        dt/dx ratio (scalar for 1D, array for 2D)
    u : np.ndarray
        Stencil of function values [i-1, i, i+1]
    f : np.ndarray
        Stencil of flux values [i-1, i, i+1]

    Returns
    -------
    float
        Updated value u^(n+1)
    """
    if n_dim == 1:
        # 1D Case
        e = np.zeros(2)
        for l in range(2):
            if u[l + 1] == u[l]:
                e[l] = 0  # Artificial viscosity
            else:
                e[l] = max((f[l + 1] - f[l]) / (u[l + 1] - u[l]),
                          (-f[l + 1] + f[l]) / (u[l + 1] - u[l]))

        # Compute u^(n+1)
        u_n = u[1] - lmda / 2 * (f[2] - f[0]) \
            + lmda / 2 * (e[1] * (u[2] - u[1]) - e[0] * (u[1] - u[0]))

    elif n_dim == 2:
        # 2D Case
        lmda = np.atleast_1d(lmda)
        e = np.zeros((n_dim, 2))
        for dim in range(n_dim):
            for l in range(2):
                if u[dim, l + 1] == u[dim, l]:
                    e[dim, l] = 0  # Artificial viscosity
                else:
                    e[dim, l] = max((f[dim, l + 1] - f[dim, l]) / (u[dim, l + 1] - u[dim, l]),
                                   (-f[dim, l + 1] + f[dim, l]) / (u[dim, l + 1] - u[dim, l]))

        # Compute u^(n+1)
        u_n = u[0, 1] - lmda[0] / 2 * (f[0, 2] - f[0, 0]) \
            + lmda[0] / 2 * (e[0, 1] * (u[0, 2] - u[0, 1]) - e[0, 0] * (u[0, 1] - u[0, 0])) \
            - lmda[1] / 2 * (f[1, 2] - f[1, 0]) \
            + lmda[1] / 2 * (e[1, 1] * (u[1, 2] - u[1, 1]) - e[1, 0] * (u[1, 1] - u[1, 0]))

    return u_n


def advc(prm: 'Parameters', dt: float, V: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Advect a scalar function I over the grid.

    Equation: dI/dt + V*grad(I) = 0

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    dt : float
        Time step
    V : np.ndarray
        Flow velocity [u, (v)]
    I : np.ndarray
        Scalar function (level set)

    Returns
    -------
    np.ndarray
        Advected scalar function
    """
    n_dim = prm.n_dim
    n = prm.n
    dx = prm.dx

    if n_dim == 1:
        # 1D Case
        f = V * I  # Advective flux
        In = I.copy()  # Store I^n
        lmda = dt / dx

        for i in range(n):
            I_i = [i - 1, i, i + 1]
            f_i = [i - 1, i, i + 1]

            if i == 0:
                I_i = [i, i, i]
                f_i = [i, i + 1, i + 2]
            elif i == n - 1:
                I_i = [i, i, i]
                f_i = [i - 2, i - 1, i]

            I[i] = godunov(n_dim, lmda, In[I_i], f[f_i])

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        dx = np.atleast_1d(dx)
        u = V[0, :, :]  # Velocity components
        v = V[1, :, :]
        In = I.copy()  # Store I^n
        lmda = dt / dx

        for i in range(n[0]):
            for j in range(n[1]):
                I_i = [i - 1, i, i + 1]
                f_i = [i - 1, i, i + 1]
                I_j = [j - 1, j, j + 1]
                f_j = [j - 1, j, j + 1]

                if not (i > 0 and i < n[0] - 1 and j > 0 and j < n[1] - 1):
                    if i == 0:
                        I_i = [i, i, i]
                        f_i = [i, i + 1, i + 2]
                    if i == n[0] - 1:
                        I_i = [i, i, i]
                        f_i = [i - 2, i - 1, i]
                    if j == 0:
                        I_j = [j, j, j]
                        f_j = [j, j + 1, j + 2]
                    if j == n[1] - 1:
                        I_j = [j, j, j]
                        f_j = [j - 2, j - 1, j]

                Iv = np.zeros((2, 3))
                ff = np.zeros((2, 3))
                Iv[0, :] = In[I_i, j]
                ff[0, :] = u[f_i, j] * In[f_i, j]
                Iv[1, :] = In[i, I_j]
                ff[1, :] = v[i, f_j] * In[i, f_j]

                I[i, j] = godunov(n_dim, lmda, Iv, ff)

    return I


def _der_weno_minus(f: np.ndarray, dx: float, n: int, num_ghost: int) -> np.ndarray:
    """
    Compute WENO derivative from the minus side.

    Parameters
    ----------
    f : np.ndarray
        Function values
    dx : float
        Grid spacing
    n : int
        Number of real grid points
    num_ghost : int
        Number of ghost points

    Returns
    -------
    np.ndarray
        WENO derivative
    """
    n_tot = n + 2 * num_ghost
    ilo = num_ghost
    ihi = n + num_ghost

    # Undivided differences
    D1 = np.zeros(n_tot - 1)
    for i in range(n_tot - 1):
        D1[i] = (f[i + 1] - f[i]) / dx

    der_minus = np.zeros(n_tot)
    for i in range(n):
        k = i
        v = np.zeros(5)
        dx_array = np.zeros(5)
        for j in range(5):
            v[j] = D1[k + j]
            dx_array[j] = v[j]**2

        epsln = 1e-6 * max(dx_array) + 1e-99

        S = np.zeros(3)
        S[0] = 13/12 * (v[0] - 2*v[1] + v[2])**2 + 1/4 * (v[0] - 4*v[1] + 3*v[2])**2
        S[1] = 13/12 * (v[1] - 2*v[2] + v[3])**2 + 1/4 * (v[1] - v[3])**2
        S[2] = 13/12 * (v[2] - 2*v[3] + v[4])**2 + 1/4 * (3*v[2] - 4*v[3] + v[4])**2

        alpha = np.zeros(3)
        alpha[0] = (1/10) / (S[0] + epsln)**2
        alpha[1] = (6/10) / (S[1] + epsln)**2
        alpha[2] = (3/10) / (S[2] + epsln)**2
        alpha_tot = np.sum(alpha)

        f_x = np.zeros(3)
        f_x[0] = v[0]/3 - 7*v[1]/6 + 11*v[2]/6
        f_x[1] = -v[1]/6 + 5*v[2]/6 + v[3]/3
        f_x[2] = v[2]/3 + 5*v[3]/6 - v[4]/6

        der_minus[i + num_ghost] = (alpha[0]*f_x[0] + alpha[1]*f_x[1] + alpha[2]*f_x[2]) / alpha_tot

    return der_minus[ilo:ihi]


def _der_weno_plus(f: np.ndarray, dx: float, n: int, num_ghost: int) -> np.ndarray:
    """
    Compute WENO derivative from the plus side.

    Parameters
    ----------
    f : np.ndarray
        Function values
    dx : float
        Grid spacing
    n : int
        Number of real grid points
    num_ghost : int
        Number of ghost points

    Returns
    -------
    np.ndarray
        WENO derivative
    """
    n_tot = n + 2 * num_ghost
    ilo = num_ghost
    ihi = n + num_ghost

    # Undivided differences
    D1 = np.zeros(n_tot - 1)
    for i in range(n_tot - 1):
        D1[i] = (f[i + 1] - f[i]) / dx

    der_plus = np.zeros(n_tot)
    for i in range(n):
        k = i + 1
        v = np.zeros(5)
        dx_array = np.zeros(5)
        for j in range(5):
            v[j] = D1[k + 5 - j]
            dx_array[j] = v[j]**2

        epsln = 1e-6 * max(dx_array) + 1e-99

        S = np.zeros(3)
        S[0] = 13/12 * (v[0] - 2*v[1] + v[2])**2 + 1/4 * (v[0] - 4*v[1] + 3*v[2])**2
        S[1] = 13/12 * (v[1] - 2*v[2] + v[3])**2 + 1/4 * (v[1] - v[3])**2
        S[2] = 13/12 * (v[2] - 2*v[3] + v[4])**2 + 1/4 * (3*v[2] - 4*v[3] + v[4])**2

        alpha = np.zeros(3)
        alpha[0] = (1/10) / (S[0] + epsln)**2
        alpha[1] = (6/10) / (S[1] + epsln)**2
        alpha[2] = (3/10) / (S[2] + epsln)**2
        alpha_tot = np.sum(alpha)

        f_x = np.zeros(3)
        f_x[0] = v[0]/3 - 7*v[1]/6 + 11*v[2]/6
        f_x[1] = -v[1]/6 + 5*v[2]/6 + v[3]/3
        f_x[2] = v[2]/3 + 5*v[3]/6 - v[4]/6

        der_plus[i + num_ghost] = (alpha[0]*f_x[0] + alpha[1]*f_x[1] + alpha[2]*f_x[2]) / alpha_tot

    return der_plus[ilo:ihi]


def reinit_fast(prm: 'Parameters', phi0: np.ndarray) -> np.ndarray:
    """
    Reinitialize a signed-distance level set function phi.

    Uses WENO scheme for high-order accuracy.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    phi0 : np.ndarray
        Input level set function

    Returns
    -------
    np.ndarray
        Reinitialized level set function
    """
    n_dim = prm.n_dim
    n = prm.n
    dx = prm.dx

    if n_dim != 1:
        raise NotImplementedError("2D and 3D not currently supported")

    phi = phi0.copy()
    itmax = 1
    xlo = -1
    xhi = 1
    dtloc = 0.5 * np.min(np.atleast_1d(dx))
    zero_tol = 0
    num_ghost = 4
    ilo = num_ghost
    ihi = n - num_ghost
    num_ghost_tot = 2 * num_ghost
    n_real = n - num_ghost_tot

    it = 0
    while it < itmax:
        it += 1

        # Compute gradient at boundaries for extrapolation
        endslope = np.gradient(phi, dx)

        # Create extended array with ghost cells
        phicur = np.zeros(n)

        # Fill ghost cells with extrapolated values
        for i in range(num_ghost):
            phicur[i] = phi[ilo] - (ilo - i) * dx * endslope[ilo]
            phicur[i + ihi] = phi[ihi - 1] + (i + 1) * dx * endslope[ihi - 1]

        phicur[ilo:ihi] = phi[ilo:ihi]

        # Compute WENO derivatives
        phi_minus = np.zeros(n)
        phi_plus = np.zeros(n)
        phi_minus[ilo:ihi] = _der_weno_minus(phi, dx, n_real, num_ghost)
        phi_plus[ilo:ihi] = _der_weno_plus(phi, dx, n_real, num_ghost)

        # Compute reinitialization equation RHS
        RHS = np.zeros(n)
        for i in range(num_ghost, n - num_ghost):
            phi_i = phicur[i]

            grad_phi_plus = phi_plus[i]
            grad_phi_minus = phi_minus[i]
            grad_phi_star = 0.5 * (grad_phi_plus + grad_phi_minus)

            if abs(phi_i) >= zero_tol:
                norm_grad_phi = abs(grad_phi_star)
                sgn_phi = phi_i / np.sqrt(phi_i**2 + norm_grad_phi**2 + dx**2)
                RHS[i] = sgn_phi * (1 - norm_grad_phi)
            else:
                RHS[i] = 0

        # Advance in time
        phi = phicur + dtloc * RHS

    return phi
