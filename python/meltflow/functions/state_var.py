"""
State variable computations for MeltFlow solver.

Computes state variables (conserved, primitive, speed of sound, etc.)
over the entire grid using the appropriate equation of state.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters

from .thermodynamics import (
    cons_perfect, prim_perfect, SoS_perfect
)


def state_var(prm: 'Parameters', var: str, sz: int, phi: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute state variable or array of variables over entire grid.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    var : str
        Variable type to compute: "cons", "prim", or "SoS"
    sz : int
        Size of output (number of variables)
    phi : np.ndarray
        Level set function (positive = fluid 1, negative = fluid 2)
    H : np.ndarray
        Input array (primitive or conserved variables depending on var)

    Returns
    -------
    np.ndarray
        Computed state variables over the grid
    """
    n_dim = prm.n_dim
    n = prm.n
    EoS = prm.EoS
    c_EoS = prm.c_EoS

    # Get the function for the equation of state
    func_map = {
        'cons': {'perfect': cons_perfect},
        'prim': {'perfect': prim_perfect},
        'SoS': {'perfect': SoS_perfect}
    }

    f_1 = func_map[var].get(EoS[0])
    f_2 = func_map[var].get(EoS[0])  # Note: MATLAB code uses EoS[0] for both

    if n_dim == 1:
        # 1D Case
        a = np.zeros((sz, n))
        for i in range(n):
            if phi[i] > 0 and EoS[0] != "none":
                a[:, i] = f_1(n_dim, c_EoS[0], H[:, i])
            elif phi[i] <= 0 and EoS[1] != "none":
                a[:, i] = f_2(n_dim, c_EoS[0], H[:, i])
        return a

    elif n_dim == 2:
        # 2D Case
        n = np.atleast_1d(n)
        a = np.zeros((sz, n[0], n[1]))
        for i in range(n[0]):
            for j in range(n[1]):
                if phi[i, j] > 0 and EoS[0] != "none":
                    a[:, i, j] = f_1(n_dim, c_EoS[0], H[:, i, j])
                elif phi[i, j] <= 0 and EoS[1] != "none":
                    a[:, i, j] = f_2(n_dim, c_EoS[0], H[:, i, j])
        return np.squeeze(a)
