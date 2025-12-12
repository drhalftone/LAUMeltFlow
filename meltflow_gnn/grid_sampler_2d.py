"""
Uniform grid sampling for 2D MLP flux training data.

Extends the 1D approach to 2D by concatenating 4 neighbors (W, E, S, N) plus center cell
to predict all 4 interface fluxes at once.

Architecture:
- Input: [ρ_W, u_W, v_W, p_W, ρ_C, u_C, v_C, p_C, ρ_E, u_E, v_E, p_E,
          ρ_S, u_S, v_S, p_S, ρ_N, u_N, v_N, p_N] = 20 features
- Output: [F_w(4), F_e(4), G_s(4), G_n(4)] = 16 flux values
"""

import numpy as np
from typing import Tuple, Dict, Optional
import sys
import os


def roe_flux2D(dim: int, gam: float, W_L: np.ndarray, W_R: np.ndarray) -> np.ndarray:
    """
    Calculate flux F_(i+1/2) for 2D Roe's approximate solver.

    Parameters
    ----------
    dim : int
        Sweep direction: 1 for y-sweep, 2 for x-sweep
    gam : float
        Specific heat ratio
    W_L : np.ndarray
        Conserved variables to left of Riemann problem [rho, rho*u, rho*v, E]
    W_R : np.ndarray
        Conserved variables to right of Riemann problem [rho, rho*u, rho*v, E]

    Returns
    -------
    np.ndarray
        Flux at interface F_(i+1/2), shape (4,)
    """
    n_var = 4  # [rho, rho*u, rho*v, E]

    # Pressure function for 2D
    def pres(W):
        return (gam - 1) * (W[3] - 0.5 * W[0] * ((W[1]/W[0])**2 + (W[2]/W[0])**2))

    # Roe average helper
    def roe_avg(rho_L, rho_R, r_L, r_R):
        return (np.sqrt(rho_L) * r_L + np.sqrt(rho_R) * r_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))

    # Pre-Processing
    rho_L, rho_R = W_L[0], W_R[0]
    u_L = W_L[1] / W_L[0]
    u_R = W_R[1] / W_R[0]
    v_L = W_L[2] / W_L[0]
    v_R = W_R[2] / W_R[0]
    p_L = pres(W_L)
    p_R = pres(W_R)
    E_L, E_R = W_L[3], W_R[3]
    h_L = (E_L + p_L) / rho_L
    h_R = (E_R + p_R) / rho_R
    dW = W_R - W_L

    # Roe-Averages
    u = roe_avg(rho_L, rho_R, u_L, u_R)
    v = roe_avg(rho_L, rho_R, v_L, v_R)
    h = roe_avg(rho_L, rho_R, h_L, h_R)
    a = np.sqrt((gam - 1) * (h - 0.5 * (u**2 + v**2)))
    V_2 = u**2 + v**2

    if dim == 1:
        # y-direction flux (vertical interfaces)
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
        # x-direction flux (horizontal interfaces)
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


def prim_to_cons_2d(rho: float, u: float, v: float, p: float, gam: float) -> np.ndarray:
    """
    Convert 2D primitive variables to conserved variables.

    Parameters
    ----------
    rho : float
        Density
    u : float
        x-velocity
    v : float
        y-velocity
    p : float
        Pressure
    gam : float
        Specific heat ratio

    Returns
    -------
    np.ndarray
        Conserved variables [rho, rho*u, rho*v, E]
    """
    E = p / (gam - 1) + 0.5 * rho * (u**2 + v**2)
    return np.array([rho, rho * u, rho * v, E])


def get_2d_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter ranges for 2D problems.

    Extended ranges to cover typical 2D shock tube and implosion problems.
    """
    ranges = {
        'rho': (0.05, 1.2),       # Density [kg/m^3]
        'u': (-150.0, 350.0),     # x-velocity [m/s]
        'v': (-150.0, 350.0),     # y-velocity [m/s]
        'p': (5000.0, 120000.0)   # Pressure [Pa]
    }

    print(f"2D parameter ranges:")
    for var, (vmin, vmax) in ranges.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return ranges


def compute_all_fluxes(W_W: np.ndarray, W_C: np.ndarray, W_E: np.ndarray,
                       W_S: np.ndarray, W_N: np.ndarray, gam: float) -> np.ndarray:
    """
    Compute all 4 interface fluxes for a center cell given its 4 neighbors.

    Parameters
    ----------
    W_W, W_C, W_E, W_S, W_N : np.ndarray
        Conserved variables for West, Center, East, South, North cells
        Each has shape (4,): [rho, rho*u, rho*v, E]
    gam : float
        Specific heat ratio

    Returns
    -------
    np.ndarray
        Shape (16,): [F_w(4), F_e(4), G_s(4), G_n(4)]
        - F_w: flux at west face (between W and C)
        - F_e: flux at east face (between C and E)
        - G_s: flux at south face (between S and C)
        - G_n: flux at north face (between C and N)
    """
    # X-direction fluxes (dim=2)
    F_w = roe_flux2D(dim=2, gam=gam, W_L=W_W, W_R=W_C)  # West face
    F_e = roe_flux2D(dim=2, gam=gam, W_L=W_C, W_R=W_E)  # East face

    # Y-direction fluxes (dim=1)
    G_s = roe_flux2D(dim=1, gam=gam, W_L=W_S, W_R=W_C)  # South face
    G_n = roe_flux2D(dim=1, gam=gam, W_L=W_C, W_R=W_N)  # North face

    return np.concatenate([F_w, F_e, G_s, G_n])


def generate_uniform_grid_samples_2d(
    ranges: Dict[str, Tuple[float, float]],
    n_samples_per_dim: int = 5,
    gam: float = 1.4,
    random_sampling: bool = True,
    n_total_samples: int = 1000000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform grid samples over the 2D state space.

    For 2D with 5 cells (W, C, E, S, N), each with 4 primitives (ρ, u, v, p),
    we have 20 input dimensions. A grid approach would be n^20 which is infeasible.

    Instead, we use random uniform sampling within the parameter ranges.

    Parameters
    ----------
    ranges : Dict
        Parameter ranges for rho, u, v, p
    n_samples_per_dim : int
        Not used for random sampling (kept for API compatibility)
    gam : float
        Specific heat ratio
    random_sampling : bool
        If True, use random sampling. If False, attempt grid (will be very slow)
    n_total_samples : int
        Total number of samples for random sampling

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs: Shape (N, 20) - [W(4), C(4), E(4), S(4), N(4)] primitives
        outputs: Shape (N, 16) - [F_w(4), F_e(4), G_s(4), G_n(4)] fluxes
    """
    print(f"Generating {n_total_samples:,} random 2D samples...")

    # Sample random states for all 5 cells
    rho_range = ranges['rho']
    u_range = ranges['u']
    v_range = ranges['v']
    p_range = ranges['p']

    inputs = []
    outputs = []
    valid_count = 0
    invalid_count = 0

    # Progress tracking
    report_interval = n_total_samples // 20

    for i in range(n_total_samples):
        # Sample primitives for each of 5 cells
        cells_prim = []  # Will hold primitives [rho, u, v, p] for W, C, E, S, N
        cells_cons = []  # Will hold conserved [rho, rho*u, rho*v, E]

        for _ in range(5):  # 5 cells
            rho = np.random.uniform(rho_range[0], rho_range[1])
            u = np.random.uniform(u_range[0], u_range[1])
            v = np.random.uniform(v_range[0], v_range[1])
            p = np.random.uniform(p_range[0], p_range[1])

            cells_prim.append([rho, u, v, p])
            cells_cons.append(prim_to_cons_2d(rho, u, v, p, gam))

        try:
            # Compute all 4 interface fluxes
            W_W, W_C, W_E, W_S, W_N = cells_cons
            flux = compute_all_fluxes(W_W, W_C, W_E, W_S, W_N, gam)

            # Check for NaN or Inf
            if np.any(np.isnan(flux)) or np.any(np.isinf(flux)):
                invalid_count += 1
                continue

            # Flatten primitives: [W, C, E, S, N] each with [rho, u, v, p]
            input_vec = np.concatenate(cells_prim)  # Shape (20,)
            inputs.append(input_vec)
            outputs.append(flux)
            valid_count += 1

        except Exception as e:
            invalid_count += 1
            continue

        # Progress report
        if (i + 1) % report_interval == 0:
            pct = 100 * (i + 1) / n_total_samples
            print(f"  Progress: {pct:.0f}% ({valid_count:,} valid, {invalid_count:,} invalid)")

    print(f"Generated {valid_count:,} valid samples, {invalid_count:,} invalid samples")

    return np.array(inputs), np.array(outputs)


def generate_uniform_flux_dataset_2d(
    n_total_samples: int = 1000000,
    gam: float = 1.4,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: get ranges and generate uniform 2D samples.

    Parameters
    ----------
    n_total_samples : int
        Total number of samples
    gam : float
        Specific heat ratio
    save_path : str, optional
        Path to save the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs (N, 20), outputs (N, 16)
    """
    # Step 1: Get ranges
    ranges = get_2d_parameter_ranges()

    # Step 2: Generate random uniform samples
    inputs, outputs = generate_uniform_grid_samples_2d(
        ranges,
        gam=gam,
        random_sampling=True,
        n_total_samples=n_total_samples
    )

    # Step 3: Save if requested
    if save_path:
        np.savez(save_path, inputs=inputs, outputs=outputs, ranges=ranges)
        print(f"Saved dataset to {save_path}")

    return inputs, outputs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate uniform 2D training data')
    parser.add_argument('--n-samples', type=int, default=1000000,
                        help='Total number of samples')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='Specific heat ratio')
    parser.add_argument('--output', type=str, default='data/uniform_flux_data_2d.npz',
                        help='Output file path')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate dataset
    inputs, outputs = generate_uniform_flux_dataset_2d(
        n_total_samples=args.n_samples,
        gam=args.gamma,
        save_path=args.output
    )

    print(f"\nDataset summary:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")

    # Input statistics
    print(f"\n  Input ranges (primitives for W, C, E, S, N cells):")
    cell_names = ['W', 'C', 'E', 'S', 'N']
    var_names = ['rho', 'u', 'v', 'p']
    for c, cell in enumerate(cell_names):
        print(f"    Cell {cell}:")
        for v, var in enumerate(var_names):
            idx = c * 4 + v
            print(f"      {var}: [{inputs[:, idx].min():.6g}, {inputs[:, idx].max():.6g}]")

    # Output statistics
    print(f"\n  Output ranges (fluxes at W, E, S, N faces):")
    face_names = ['F_w', 'F_e', 'G_s', 'G_n']
    flux_names = ['mass', 'mom_x', 'mom_y', 'energy']
    for f, face in enumerate(face_names):
        print(f"    {face}:")
        for fl, fname in enumerate(flux_names):
            idx = f * 4 + fl
            print(f"      {fname}: [{outputs[:, idx].min():.6g}, {outputs[:, idx].max():.6g}]")
