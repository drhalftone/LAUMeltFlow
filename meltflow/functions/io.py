"""
Input/Output functions for MeltFlow solver.

Includes data writing, reading, and interpolation.
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters


def interpolate(prm: 'Parameters', X: np.ndarray, U: np.ndarray,
                phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate grid primitive variable data U on (X,Y) to a different grid.

    Method: Constructs rectangular uniform grid of size n_out.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T
    phi : np.ndarray
        Level set function

    Returns
    -------
    tuple
        (X_out, U_out, phi_out) - Interpolated grid, variables, and level set
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    x_min = prm.x_min
    x_max = prm.x_max
    n_out = prm.n_out

    if n_dim == 1:
        # 1D Case
        U_out = np.zeros((n_var, n_out))  # Allocate output variables
        x_out = np.linspace(x_min, x_max, n_out)  # Output grid
        X_out = x_out

        # Interpolate each variable
        for k in range(n_var):
            f = interp1d(X, U[k, :], kind='linear', fill_value='extrapolate')
            U_out[k, :] = f(X_out)

        f_phi = interp1d(X, phi, kind='linear', fill_value='extrapolate')
        phi_out = f_phi(X_out)

    elif n_dim == 2:
        # 2D Case
        n_out = np.atleast_1d(n_out)
        x_min = np.atleast_1d(x_min)
        x_max = np.atleast_1d(x_max)

        Y = X[1, :, :]
        X_grid = X[0, :, :]

        U_out = np.zeros((n_var, int(n_out[0]), int(n_out[1])))

        # Create output grid
        x_out = np.linspace(x_min[0], x_max[0], int(n_out[0]))
        y_out = np.linspace(x_min[1], x_max[1], int(n_out[1]))
        X_p, Y_p = np.meshgrid(x_out, y_out, indexing='ij')

        # Interpolate each variable
        for k in range(n_var):
            # Use RegularGridInterpolator for better 2D interpolation
            f = RegularGridInterpolator(
                (X_grid[:, 0], Y[0, :]),
                U[k, :, :],
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            pts = np.column_stack([X_p.ravel(), Y_p.ravel()])
            U_out[k, :, :] = f(pts).reshape(X_p.shape)

        # Interpolate level set
        f_phi = RegularGridInterpolator(
            (X_grid[:, 0], Y[0, :]),
            phi,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        phi_out = f_phi(np.column_stack([X_p.ravel(), Y_p.ravel()])).reshape(X_p.shape)

        X_out = np.zeros((2, int(n_out[0]), int(n_out[1])))
        X_out[0, :, :] = X_p
        X_out[1, :, :] = Y_p

    return X_out, U_out, phi_out


def wrt_data(prm: 'Parameters', X_out: np.ndarray, U_out: np.ndarray,
             phi_out: np.ndarray, wrt_fl: str) -> None:
    """
    Write 1D/2D CFD primitive variable data to a file.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    X_out : np.ndarray
        Output grid coordinates
    U_out : np.ndarray
        Output primitive variables [rho, u, (v), p]^T
    phi_out : np.ndarray
        Output level set function
    wrt_fl : str
        Output filename
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n = prm.n
    n_out = prm.n_out
    flg_intrp = prm.flg_intrp

    if not flg_intrp:
        n_out = n

    print(f"Writing flow field (file = '{wrt_fl}')...")

    with open(wrt_fl, 'w') as fid:
        if n_dim == 1:
            # 1D Case
            fid.write('x rho u p phi\n')  # Header
            for i in range(n_out):
                fid.write(f'{X_out[i]:.6f} {U_out[0, i]:.6f} '
                         f'{U_out[1, i]:.6f} {U_out[2, i]:.6f} '
                         f'{phi_out[i]:.6f}\n')

        elif n_dim == 2:
            # 2D Case
            n_out = np.atleast_1d(n_out).astype(int)
            fid.write('x y rho u v p phi\n')  # Header
            for i in range(n_out[0]):
                for j in range(n_out[1]):
                    fid.write(f'{X_out[0, i, j]:.6f} {X_out[1, i, j]:.6f} '
                             f'{U_out[0, i, j]:.6f} {U_out[1, i, j]:.6f} '
                             f'{U_out[2, i, j]:.6f} {U_out[3, i, j]:.6f} '
                             f'{phi_out[i, j]:.6f}\n')


def rd_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read CFD data from a file.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    tuple
        (X, U, phi) - Grid coordinates, primitive variables, level set
    """
    import pandas as pd

    df = pd.read_csv(filename, delim_whitespace=True)

    if 'y' in df.columns:
        # 2D data
        x = df['x'].values
        y = df['y'].values

        # Determine grid dimensions
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        nx, ny = len(unique_x), len(unique_y)

        X = np.zeros((2, nx, ny))
        X[0, :, :] = x.reshape(nx, ny)
        X[1, :, :] = y.reshape(nx, ny)

        U = np.zeros((4, nx, ny))
        U[0, :, :] = df['rho'].values.reshape(nx, ny)
        U[1, :, :] = df['u'].values.reshape(nx, ny)
        U[2, :, :] = df['v'].values.reshape(nx, ny)
        U[3, :, :] = df['p'].values.reshape(nx, ny)

        phi = df['phi'].values.reshape(nx, ny)
    else:
        # 1D data
        X = df['x'].values
        n = len(X)

        U = np.zeros((3, n))
        U[0, :] = df['rho'].values
        U[1, :] = df['u'].values
        U[2, :] = df['p'].values

        phi = df['phi'].values

    return X, U, phi
