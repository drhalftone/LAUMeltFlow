"""
Grid setup and mesh initialization for MeltFlow solver.
"""

import numpy as np
from typing import Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters


def grid_setup(config: Dict[str, Any]) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up the computational grid and allocate arrays for initial conditions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing grid parameters

    Returns
    -------
    tuple
        (n_var, n, x, dx, U, phi) - Grid parameters and allocated arrays
    """
    n_dim = config['n_dim']
    n_var = n_dim + 2
    x_min = config['x_min']
    x_max = config['x_max']

    if n_dim == 1:
        if 'n' in config:
            n = config['n']
            x = np.linspace(x_min, x_max, n)
            dx = x[1] - x[0]
        elif 'dx' in config:
            dx = config['dx']
            x = np.arange(x_min, x_max + dx/2, dx)  # +dx/2 to include endpoint
            n = len(x)

        U = np.zeros((n_var, n))
        phi = np.zeros(n)

    elif n_dim == 2:
        # Handle n as scalar or array
        if 'n' in config:
            n = config['n']
            if np.isscalar(n):
                n = np.array([n, n])
            else:
                n = np.atleast_1d(n)
        else:
            n = None

        # Handle dx as scalar or array
        if 'dx' in config:
            dx = config['dx']
            if np.isscalar(dx):
                dx = np.array([dx, dx])
            else:
                dx = np.atleast_1d(dx)
        else:
            dx = None

        x_min = np.atleast_1d(x_min)
        x_max = np.atleast_1d(x_max)

        if n is not None:
            xx = np.linspace(x_min[0], x_max[0], n[0])
            yy = np.linspace(x_min[1], x_max[1], n[1])
            dx = np.array([xx[1] - xx[0], yy[1] - yy[0]])
        else:
            xx = np.arange(x_min[0], x_max[0] + dx[0]/2, dx[0])
            yy = np.arange(x_min[1], x_max[1] + dx[1]/2, dx[1])
            n = np.array([len(xx), len(yy)])

        # Store x coordinates
        max_len = max(len(xx), len(yy))
        x = np.zeros((n_dim, max_len))
        x[0, :len(xx)] = xx
        x[1, :len(yy)] = yy

        U = np.zeros((n_var, n[0], n[1]))
        phi = np.zeros((n[0], n[1]))

    return n_var, n, x, dx, U, phi


def mesh_ICs(config: Dict[str, Any]) -> Tuple['Parameters', np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate grid and apply initial conditions for GFM solver.

    Parameters
    ----------
    config : dict
        Configuration dictionary from input file

    Returns
    -------
    tuple
        (prm, X, U, phi) - Parameters, mesh coordinates, primitive variables, level set
    """
    from .parameters import create_parameters, apply_defaults

    n_dim = config['n_dim']

    # Set up grid
    n_var, n, x, dx, U, phi = grid_setup(config)

    # Update config with computed values
    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx

    # Apply defaults and create parameters
    config = apply_defaults(config, n_dim)
    prm = create_parameters(config)

    # Update parameters with actual n
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Create mesh
    if n_dim == 1:
        X = x
    elif n_dim == 2:
        n = np.atleast_1d(n)
        xx = x[0, :n[0]]
        yy = x[1, :n[1]]
        XX, YY = np.meshgrid(xx, yy, indexing='ij')
        X = np.zeros((2, n[0], n[1]))
        X[0, :, :] = XX
        X[1, :, :] = YY

    # Apply initial conditions (this will be done in the input config)
    # The config should include an 'init_func' that sets U and phi

    return prm, X, U, phi
