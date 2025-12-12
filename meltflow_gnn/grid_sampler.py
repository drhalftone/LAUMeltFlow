"""
Uniform grid sampling for GNN flux training data.

Instead of relying on simulation trajectories (which have imbalanced state distributions),
this module samples the state space uniformly and computes Roe flux for each combination.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from itertools import product
import sys
import os


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

    # Pressure function for 1D
    def pres(W):
        return (gam - 1) * (W[2] - 0.5 * W[0] * (W[1] / W[0])**2)

    # Roe average helper
    def roe_avg(rho_L, rho_R, r_L, r_R):
        return (np.sqrt(rho_L) * r_L + np.sqrt(rho_R) * r_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))

    # Pre-Processing
    rho_L, rho_R = W_L[0], W_R[0]           # Densities
    u_L = W_L[1] / W_L[0]                    # Velocities
    u_R = W_R[1] / W_R[0]
    p_L = pres(W_L)                          # Pressures
    p_R = pres(W_R)
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


def prim_to_cons(rho: float, u: float, p: float, gam: float) -> np.ndarray:
    """
    Convert primitive variables to conserved variables.

    Parameters
    ----------
    rho : float
        Density
    u : float
        Velocity
    p : float
        Pressure
    gam : float
        Specific heat ratio (gamma)

    Returns
    -------
    np.ndarray
        Conserved variables [rho, rho*u, E]
    """
    E = p / (gam - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])


def get_sod_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter ranges for the Sod shock tube problem.

    Based on analysis of the Sod problem simulation:
    - Initial left state: rho=1.0, u=0, p=100000
    - Initial right state: rho=0.125, u=0, p=10000
    - Final states develop velocity range roughly [-100, 300] m/s
    - Density ranges from ~0.1 to 1.0
    - Pressure ranges from ~6000 to 100000 Pa

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary with keys 'rho', 'u', 'p' and (min, max) tuples
    """
    # Based on analysis from the GNN report and Sod problem physics
    ranges = {
        'rho': (0.05, 1.2),      # Density [kg/m^3]
        'u': (-150.0, 350.0),    # Velocity [m/s]
        'p': (5000.0, 120000.0)  # Pressure [Pa]
    }

    print(f"Sod problem parameter ranges:")
    for var, (vmin, vmax) in ranges.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return ranges


def get_parameter_ranges_from_simulation(config_name: str = 'in_1Dsod1fl') -> Dict[str, Tuple[float, float]]:
    """
    Run a simulation and extract min/max ranges for all primitive variables.

    Note: This requires the full meltflow package with matplotlib.
    Use get_sod_parameter_ranges() for a standalone version.

    Parameters
    ----------
    config_name : str
        Configuration name to run

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary with keys 'rho', 'u', 'p' and (min, max) tuples
    """
    try:
        from meltflow_gnn.data_generator import generate_trajectory
    except ImportError:
        print("Warning: Could not import meltflow_gnn.data_generator")
        print("Using predefined Sod parameter ranges instead.")
        return get_sod_parameter_ranges()

    print(f"Running simulation '{config_name}' to determine parameter ranges...")
    trajectory = generate_trajectory(config_name=config_name, save_interval=1)

    # Collect all states
    all_rho = []
    all_u = []
    all_p = []

    for step in trajectory:
        U = step['U']  # Shape: (3, n) for [rho, u, p]
        all_rho.extend(U[0, :].tolist())
        all_u.extend(U[1, :].tolist())
        all_p.extend(U[2, :].tolist())

    ranges = {
        'rho': (min(all_rho), max(all_rho)),
        'u': (min(all_u), max(all_u)),
        'p': (min(all_p), max(all_p))
    }

    print(f"Parameter ranges from simulation:")
    for var, (vmin, vmax) in ranges.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return ranges


def expand_ranges(ranges: Dict[str, Tuple[float, float]],
                  expansion_factor: float = 1.2) -> Dict[str, Tuple[float, float]]:
    """
    Expand parameter ranges to include some extrapolation region.

    Parameters
    ----------
    ranges : Dict
        Original ranges
    expansion_factor : float
        Factor to expand ranges by (1.2 = 20% expansion on each side)

    Returns
    -------
    Dict
        Expanded ranges
    """
    expanded = {}
    for var, (vmin, vmax) in ranges.items():
        center = (vmin + vmax) / 2
        half_width = (vmax - vmin) / 2
        new_half_width = half_width * expansion_factor

        new_min = center - new_half_width
        new_max = center + new_half_width

        # Ensure physical constraints
        if var == 'rho':
            new_min = max(new_min, 1e-6)  # Density must be positive
        elif var == 'p':
            new_min = max(new_min, 1e-3)  # Pressure must be positive

        expanded[var] = (new_min, new_max)

    print(f"Expanded ranges (factor={expansion_factor}):")
    for var, (vmin, vmax) in expanded.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return expanded


def generate_uniform_grid_samples(
    ranges: Dict[str, Tuple[float, float]],
    n_samples_per_dim: int = 10,
    gam: float = 1.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform grid samples over the 6D state space and compute Roe flux.

    Parameters
    ----------
    ranges : Dict
        Parameter ranges for rho, u, p
    n_samples_per_dim : int
        Number of samples per dimension (total = n^6)
    gam : float
        Specific heat ratio

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs: Shape (N, 6) - [rho_L, u_L, p_L, rho_R, u_R, p_R]
        outputs: Shape (N, 3) - [flux_rho, flux_rhou, flux_E]
    """
    # Create 1D grids for each variable
    rho_grid = np.linspace(ranges['rho'][0], ranges['rho'][1], n_samples_per_dim)
    u_grid = np.linspace(ranges['u'][0], ranges['u'][1], n_samples_per_dim)
    p_grid = np.linspace(ranges['p'][0], ranges['p'][1], n_samples_per_dim)

    total_samples = n_samples_per_dim ** 6
    print(f"Generating {total_samples:,} samples ({n_samples_per_dim}^6)...")

    inputs = []
    outputs = []
    valid_count = 0
    invalid_count = 0

    # Nested loops over all 6 dimensions
    for rho_L in rho_grid:
        for u_L in u_grid:
            for p_L in p_grid:
                for rho_R in rho_grid:
                    for u_R in u_grid:
                        for p_R in p_grid:
                            # Convert to conserved variables
                            W_L = prim_to_cons(rho_L, u_L, p_L, gam)
                            W_R = prim_to_cons(rho_R, u_R, p_R, gam)

                            try:
                                # Compute Roe flux
                                flux = roe_flux1D(n_dim=1, gam=gam, W_L=W_L, W_R=W_R)

                                # Check for NaN or Inf
                                if np.any(np.isnan(flux)) or np.any(np.isinf(flux)):
                                    invalid_count += 1
                                    continue

                                inputs.append([rho_L, u_L, p_L, rho_R, u_R, p_R])
                                outputs.append(flux)
                                valid_count += 1

                            except Exception as e:
                                invalid_count += 1
                                continue

    print(f"Generated {valid_count:,} valid samples, {invalid_count:,} invalid samples")

    return np.array(inputs), np.array(outputs)


def generate_uniform_flux_dataset(
    n_samples_per_dim: int = 10,
    expansion_factor: float = 1.0,
    gam: float = 1.4,
    save_path: Optional[str] = None,
    use_simulation_ranges: bool = False,
    config_name: str = 'in_1Dsod1fl'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: get ranges and generate uniform samples.

    Parameters
    ----------
    n_samples_per_dim : int
        Samples per dimension (total = n^6)
    expansion_factor : float
        Range expansion factor (1.0 = no expansion)
    gam : float
        Specific heat ratio
    save_path : str, optional
        Path to save the dataset
    use_simulation_ranges : bool
        If True, run simulation to get ranges; else use predefined Sod ranges
    config_name : str
        Configuration to run for range estimation (if use_simulation_ranges=True)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs, outputs arrays
    """
    # Step 1: Get ranges
    if use_simulation_ranges:
        ranges = get_parameter_ranges_from_simulation(config_name)
    else:
        ranges = get_sod_parameter_ranges()

    # Step 2: Expand ranges if requested
    if expansion_factor != 1.0:
        expanded_ranges = expand_ranges(ranges, expansion_factor)
    else:
        expanded_ranges = ranges

    # Step 3: Generate uniform samples
    inputs, outputs = generate_uniform_grid_samples(
        expanded_ranges,
        n_samples_per_dim=n_samples_per_dim,
        gam=gam
    )

    # Step 4: Save if requested
    if save_path:
        np.savez(save_path, inputs=inputs, outputs=outputs, ranges=expanded_ranges)
        print(f"Saved dataset to {save_path}")

    return inputs, outputs


def create_flux_training_data(
    inputs: np.ndarray,
    outputs: np.ndarray,
    normalize: bool = True
):
    """
    Convert numpy arrays to PyTorch tensors with optional normalization.

    Parameters
    ----------
    inputs : np.ndarray
        Input features [rho_L, u_L, p_L, rho_R, u_R, p_R]
    outputs : np.ndarray
        Output fluxes [flux_rho, flux_rhou, flux_E]
    normalize : bool
        Whether to normalize the data

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Dict]
        X, Y tensors and normalization statistics
    """
    import torch

    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(outputs, dtype=torch.float32)

    stats = {}

    if normalize:
        # Normalize inputs
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X_std[X_std < 1e-8] = 1.0  # Avoid division by zero
        X = (X - X_mean) / X_std

        # Normalize outputs
        Y_mean = Y.mean(dim=0)
        Y_std = Y.std(dim=0)
        Y_std[Y_std < 1e-8] = 1.0
        Y = (Y - Y_mean) / Y_std

        stats = {
            'X_mean': X_mean,
            'X_std': X_std,
            'Y_mean': Y_mean,
            'Y_std': Y_std
        }

    return X, Y, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate uniform grid training data')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Samples per dimension (total = n^6)')
    parser.add_argument('--expansion', type=float, default=1.0,
                        help='Range expansion factor (1.0 = no expansion)')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='Specific heat ratio')
    parser.add_argument('--output', type=str, default='data/uniform_flux_data.npz',
                        help='Output file path')
    parser.add_argument('--use-simulation', action='store_true',
                        help='Run simulation to get ranges (requires matplotlib)')
    parser.add_argument('--config', type=str, default='in_1Dsod1fl',
                        help='Configuration for range estimation (if --use-simulation)')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate dataset
    inputs, outputs = generate_uniform_flux_dataset(
        n_samples_per_dim=args.n_samples,
        expansion_factor=args.expansion,
        gam=args.gamma,
        save_path=args.output,
        use_simulation_ranges=args.use_simulation,
        config_name=args.config
    )

    print(f"\nDataset summary:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Input ranges:")
    for i, name in enumerate(['rho_L', 'u_L', 'p_L', 'rho_R', 'u_R', 'p_R']):
        print(f"    {name}: [{inputs[:, i].min():.6g}, {inputs[:, i].max():.6g}]")
    print(f"  Output ranges:")
    for i, name in enumerate(['flux_rho', 'flux_rhou', 'flux_E']):
        print(f"    {name}: [{outputs[:, i].min():.6g}, {outputs[:, i].max():.6g}]")
