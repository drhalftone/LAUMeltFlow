"""
Input configurations for MeltFlow test cases.

Each configuration function returns a dictionary with simulation parameters
and an initialization function that sets U and phi.
"""

import numpy as np
from typing import Dict, Any, Callable, Tuple


def in_1Dsod1fl() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Single Fluid Pure Compressible Sod's Shock Tube.

    Regions:
        (1) Left, (2) Right
    Dimensions:
        d = Initial diaphragm location

    Schematic:
        x_min                         x_max
         |----- 1 ------====== 2 ======|
         |<--- d --->|
    """
    config = {
        'ICs_hdr': "%---- 1D Test Case - Single Fluid Pure Compressible Sod's Shock Tube ----%",
        'n_dim': 1,
        'dx': 0.01,
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 1],  # Gas, Liquid (but liquid not used)
        'EoS': ["perfect", "none"],
        'c_EoS': [1.4, 1.0],
        'slvr': ["roe_perfect", "none"],
        'cfl': 0.9,
        't_f': 7.5e-4,
        'flg_BCs': 1,
        'n_out': 51,
        'wrt_nm': "flow_1Dsod1fl",
        'opt_plt': 1,
        't_anmt': 1.2,
        'n_anmt': 2,
    }

    # Region properties
    d = 0.5
    U_r = np.array([
        [1.0, 100.0, 1e5],      # Region 1: [rho, u, p]
        [0.125, 0.0, 1e4],      # Region 2: [rho, u, p]
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry."""
        n = len(x)
        for i in range(n):
            if x[i] <= d:
                U[:, i] = U_r[0, :]
            else:
                U[:, i] = U_r[1, :]
        phi[:] = 1  # Make level set pure gas
        return U, phi

    return config, init_func


def in_1Dsod1fl_dg0() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Sod's Shock Tube using DG with p=0 (should match FVM exactly).
    """
    config = {
        'ICs_hdr': "%---- 1D Test Case - Sod's Shock Tube (DG p=0, FVM-equivalent) ----%",
        'n_dim': 1,
        'dx': 0.01,
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 1],
        'EoS': ["perfect", "none"],
        'c_EoS': [1.4, 1.0],
        'slvr': ["dg_perfect", "none"],
        'cfl': 0.9,
        't_f': 7.5e-4,
        'flg_BCs': 1,
        'n_out': 51,
        'wrt_nm': "flow_1Dsod1fl_dg0",
        'opt_plt': 1,
        't_anmt': 1.2,
        'n_anmt': 2,
        'method': 'dg',
        'dg_order': 0,  # p=0 should be identical to FVM
    }

    d = 0.5
    U_r = np.array([
        [1.0, 100.0, 1e5],
        [0.125, 0.0, 1e4],
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        for i in range(n):
            if x[i] <= d:
                U[:, i] = U_r[0, :]
            else:
                U[:, i] = U_r[1, :]
        phi[:] = 1
        return U, phi

    return config, init_func


def in_1Dsod1fl_dg() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Single Fluid Sod's Shock Tube using Discontinuous Galerkin.

    Same as in_1Dsod1fl but uses DG method for comparison testing.
    Set dg_order=0 for FVM-equivalent, dg_order=1 for 2nd order accuracy.
    """
    config = {
        'ICs_hdr': "%---- 1D Test Case - Sod's Shock Tube (DG p=1, 2nd order) ----%",
        'n_dim': 1,
        'dx': 0.01,
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 1],
        'EoS': ["perfect", "none"],
        'c_EoS': [1.4, 1.0],
        'slvr': ["dg_perfect", "none"],  # Use DG solver
        'cfl': 0.9,
        't_f': 7.5e-4,
        'flg_BCs': 1,
        'n_out': 51,
        'wrt_nm': "flow_1Dsod1fl_dg",
        'opt_plt': 1,
        't_anmt': 1.2,
        'n_anmt': 2,
        # DG-specific parameters
        'method': 'dg',
        'dg_order': 1,  # p=1 for linear (2nd order), p=0 for FVM-equivalent
    }

    d = 0.5
    U_r = np.array([
        [1.0, 100.0, 1e5],
        [0.125, 0.0, 1e4],
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry."""
        n = len(x)
        for i in range(n):
            if x[i] <= d:
                U[:, i] = U_r[0, :]
            else:
                U[:, i] = U_r[1, :]
        phi[:] = 1
        return U, phi

    return config, init_func


def in_1Dsod1fl_dg2() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Sod's Shock Tube using DG with p=2 (3rd order).

    WARNING: High-order DG without limiters may oscillate near shocks.
    """
    config = {
        'ICs_hdr': "%---- 1D Test Case - Sod's Shock Tube (DG p=2, 3rd order) ----%",
        'n_dim': 1,
        'dx': 0.01,
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 1],
        'EoS': ["perfect", "none"],
        'c_EoS': [1.4, 1.0],
        'slvr': ["dg_perfect", "none"],
        'cfl': 0.5,  # Lower CFL for stability with higher order
        't_f': 7.5e-4,
        'flg_BCs': 1,
        'n_out': 51,
        'wrt_nm': "flow_1Dsod1fl_dg2",
        'opt_plt': 1,
        't_anmt': 1.2,
        'n_anmt': 2,
        'method': 'dg',
        'dg_order': 2,
    }

    d = 0.5
    U_r = np.array([
        [1.0, 100.0, 1e5],
        [0.125, 0.0, 1e4],
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        for i in range(n):
            if x[i] <= d:
                U[:, i] = U_r[0, :]
            else:
                U[:, i] = U_r[1, :]
        phi[:] = 1
        return U, phi

    return config, init_func


def in_1Dsod2fl() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Two Fluid Pure Compressible Sod's Shock Tube.

    Regions:
        (1) Left (Gas 1), (2) Right (Gas 2)
    Dimensions:
        d = Initial diaphragm location
    """
    TL, TR = 300.0, 300.0
    RL, RR = 188.9, 287.0
    gamL, gamR = 1.289, 1.4
    PL, PR = 101325 * 3, 101325

    config = {
        'ICs_hdr': "%----- 1D Test Case - Two Fluid Pure Compressible Sod's Shock Tube ------%",
        'n_dim': 1,
        'dx': 0.001,
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 0],  # Both gas
        'EoS': ["perfect", "perfect"],
        'c_EoS': [gamL, gamR],
        'slvr': ["roe_perfect", "roe_perfect"],
        'cfl': 0.3,
        't_f': 5e-4,
        'flg_BCs': 0,
        'n_nds': 0,
        'n_out': 76,
        'wrt_nm': "flow_1Dsod2fl",
        'opt_plt': 1,
        'n_anmt': 2,
    }

    d = 0.5
    U_r = np.array([
        [PL / (RL * TL), 0.0, PL],  # Region 1
        [PR / (RR * TR), 0.0, PR],  # Region 2
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry."""
        n = len(x)
        for i in range(n):
            if x[i] < d:
                U[:, i] = U_r[0, :]
            else:
                U[:, i] = U_r[1, :]
        for i in range(n):
            phi[i] = d - x[i]
        return U, phi

    return config, init_func


def in_1Dcdrop() -> Tuple[Dict[str, Any], Callable]:
    """
    1D Test Case - Centered Liquid Droplet in Gas.

    Regions:
        (1) Gas, (2) Liquid

    Matches MATLAB in_1Dcdrop.m configuration.
    """
    config = {
        'ICs_hdr': "%------------------ 1D Option - Centered Liquid Droplet -----------------%",
        'n_dim': 1,
        'dx': 0.005,  # Match MATLAB (finer grid)
        'x_min': 0.0,
        'x_max': 1.0,
        'flg_fld': [0, 1],  # Gas, Liquid
        'EoS': ["perfect", "perfect"],  # Both use perfect gas EoS for pressure calc
        'c_EoS': [1.4, 1.4],  # Same gamma for both
        'slvr': ["roe_perfect", "incomp_1D"],  # Liquid uses incompressible solver
        'cfl': 0.1,  # Lower CFL for stability (matching MATLAB)
        't_f': 7.5e-4,  # Match MATLAB
        'flg_BCs': 0,  # Match MATLAB
        'n_out': 76,
        'wrt_nm': "flow_1Dcdrop",
        'opt_plt': 1,
        't_anmt': 0.1,
        'n_anmt': 5,
    }

    # Droplet parameters (matching MATLAB: d = [0.2, 0.5])
    d_center = 0.5  # Center of droplet
    d_radius = 0.1  # Radius (half of length 0.2)

    # Match MATLAB: U_r(1,:) = [1.226,0,1.0e5], U_r(2,:) = [1000,100,1.0e5]
    U_r = np.array([
        [1.226, 0.0, 1.0e5],    # Gas (air at ~STP)
        [1000.0, 100.0, 1.0e5], # Liquid (water, moving at 100 m/s)
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry.

        Level set convention (matching MATLAB):
        - phi > 0: outside droplet (GAS = fluid 1)
        - phi <= 0: inside droplet (LIQUID = fluid 2)
        """
        n = len(x)
        # Left interface at x = d_center - d_radius = 0.4
        # Right interface at x = d_center + d_radius = 0.6
        left_interface = d_center - d_radius
        right_interface = d_center + d_radius

        for i in range(n):
            if x[i] <= d_center:
                # Left of center: signed distance to left interface
                phi[i] = left_interface - x[i]  # Positive outside, negative inside
            else:
                # Right of center: signed distance to right interface
                phi[i] = x[i] - right_interface  # Positive outside, negative inside

            # Set fluid properties based on level set
            if phi[i] <= 0:
                U[:, i] = U_r[1, :]  # Inside droplet: Liquid
            else:
                U[:, i] = U_r[0, :]  # Outside droplet: Gas

        return U, phi

    return config, init_func


def in_2Dcdrop() -> Tuple[Dict[str, Any], Callable]:
    """
    2D Test Case - Circular Liquid Droplet.

    Regions:
        (1) Gas, (2) Liquid
    Dimensions:
        d[0] = Diameter of liquid droplet
        d[1] = x-coordinate of liquid droplet center
        d[2] = y-coordinate of liquid droplet center
    """
    config = {
        'ICs_hdr': "%------------------ 2D Option - Circular Liquid Droplet -----------------%",
        'n_dim': 2,
        'dx': np.array([0.02, 0.02]),
        'x_min': np.array([0.0, 0.0]),
        'x_max': np.array([1.0, 1.0]),
        'flg_fld': [0, 0],  # Both gas in this case
        'EoS': ["perfect", "perfect"],
        'c_EoS': [1.4, 1.667],
        'slvr': ["roe_perfect", "roe_perfect"],
        'cfl': 0.9,
        't_f': 3.0e-3,
        'flg_BCs': 0,
        'n_nds': 0,
        'n_disp': 10,
        'wrt_nm': 'flow_2Dcdrop',
        'flg_plt': True,
        'opt_plt': 2,
        't_anmt': 0,
    }

    # Droplet parameters
    d = [0.3125, 0.3125, 0.375]  # [diameter, x_center, y_center]
    U_r = np.array([
        [1.226, 100.0, 100.0, 1.0e5],    # Gas
        [0.164, 100.0, 100.0, 1.0e5],    # Droplet
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry."""
        n = np.array([U.shape[1], U.shape[2]])
        xx = x[0, :n[0]]
        yy = x[1, :n[1]]

        for i in range(n[0]):
            for j in range(n[1]):
                dist = np.sqrt((xx[i] - d[1])**2 + (yy[j] - d[2])**2)
                if dist <= d[0] / 2:
                    U[:, i, j] = U_r[1, :]  # Droplet
                else:
                    U[:, i, j] = U_r[0, :]  # Gas
                phi[i, j] = dist - d[0] / 2  # Positive outside, negative inside

        return U, phi

    return config, init_func


def in_2Dsod1fl() -> Tuple[Dict[str, Any], Callable]:
    """
    2D Test Case - Sod Shock Tube (Single Fluid).
    """
    config = {
        'ICs_hdr': "%------------------ 2D Option - Sod Shock Tube -----------------%",
        'n_dim': 2,
        'dx': np.array([0.02, 0.02]),
        'x_min': np.array([0.0, 0.0]),
        'x_max': np.array([1.0, 1.0]),
        'flg_fld': [0, 1],
        'EoS': ["perfect", "none"],
        'c_EoS': [1.4, 1.0],
        'slvr': ["roe_perfect", "none"],
        'cfl': 0.9,
        't_f': 5e-4,
        'flg_BCs': 1,
        'n_out': np.array([51, 51]),
        'wrt_nm': "flow_2Dsod1fl",
        'opt_plt': 2,
    }

    d = 0.5  # Diaphragm location
    U_r = np.array([
        [1.0, 0.0, 0.0, 1e5],      # Left
        [0.125, 0.0, 0.0, 1e4],    # Right
    ])

    def init_func(x: np.ndarray, U: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize U and phi based on geometry."""
        n = np.array([U.shape[1], U.shape[2]])
        xx = x[0, :n[0]]

        for i in range(n[0]):
            for j in range(n[1]):
                if xx[i] <= d:
                    U[:, i, j] = U_r[0, :]
                else:
                    U[:, i, j] = U_r[1, :]
        phi[:, :] = 1  # Single fluid

        return U, phi

    return config, init_func


# Dictionary mapping config names to functions
CONFIGS = {
    'in_1Dsod1fl': in_1Dsod1fl,
    'in_1Dsod1fl_dg0': in_1Dsod1fl_dg0,
    'in_1Dsod1fl_dg': in_1Dsod1fl_dg,
    'in_1Dsod1fl_dg2': in_1Dsod1fl_dg2,
    'in_1Dsod2fl': in_1Dsod2fl,
    'in_1Dcdrop': in_1Dcdrop,
    'in_2Dcdrop': in_2Dcdrop,
    'in_2Dsod1fl': in_2Dsod1fl,
}


def load_config(name: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Load a configuration by name.

    Parameters
    ----------
    name : str
        Configuration name (e.g., 'in_1Dsod1fl')

    Returns
    -------
    tuple
        (config dict, init_func)
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
