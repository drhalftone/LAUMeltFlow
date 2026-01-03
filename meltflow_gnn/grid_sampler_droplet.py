"""
Uniform grid sampling for air/water droplet GNN flux training data.

Parameter ranges are derived from the in_1Dcdrop simulation:
- Density: 0.9 to 1000 kg/m³ (air to water)
- Velocity: 0 to 100 m/s
- Pressure: 65,000 to 150,000 Pa

This covers the high density ratio (1000:1) case that is challenging to model.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import os


def roe_flux1D_vectorized(gam: np.ndarray, rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                          rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Roe flux for 1D Euler equations.

    All inputs should be 1D arrays of the same length.
    Returns (N, 3) array of fluxes and validity mask.
    """
    N = len(rho_L)

    # Convert to conserved
    E_L = p_L / (gam - 1) + 0.5 * rho_L * u_L**2
    E_R = p_R / (gam - 1) + 0.5 * rho_R * u_R**2

    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    h_L = (E_L + p_L) / rho_L
    h_R = (E_R + p_R) / rho_R

    rho_roe = sqrt_rho_L * sqrt_rho_R
    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) / denom
    a_roe_sq = (gam - 1) * (h_roe - 0.5 * u_roe**2)

    # Mark invalid samples (negative sound speed squared)
    valid = a_roe_sq > 0
    a_roe = np.sqrt(np.maximum(a_roe_sq, 1e-10))

    # Differences
    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    # Wave strengths (N, 3)
    dv = np.column_stack([
        drho - dp / a_roe**2,
        du + dp / (rho_roe * a_roe),
        du - dp / (rho_roe * a_roe)
    ])

    # Eigenvalues (N, 3)
    lmda = np.column_stack([u_roe, u_roe + a_roe, u_roe - a_roe])

    # Left and right fluxes (N, 3)
    F_L = np.column_stack([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])
    F_R = np.column_stack([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    # Flux correction term
    flux_correction = np.zeros((N, 3))

    # Wave 0: entropy wave
    abs_lmda0 = np.abs(lmda[:, 0])
    flux_correction[:, 0] += dv[:, 0] * abs_lmda0 * 1
    flux_correction[:, 1] += dv[:, 0] * abs_lmda0 * u_roe
    flux_correction[:, 2] += dv[:, 0] * abs_lmda0 * 0.5 * u_roe**2

    # Wave 1: right acoustic
    abs_lmda1 = np.abs(lmda[:, 1])
    coef1 = rho_roe / (2 * a_roe)
    flux_correction[:, 0] += dv[:, 1] * abs_lmda1 * coef1 * 1
    flux_correction[:, 1] += dv[:, 1] * abs_lmda1 * coef1 * (u_roe + a_roe)
    flux_correction[:, 2] += dv[:, 1] * abs_lmda1 * coef1 * (h_roe + a_roe * u_roe)

    # Wave 2: left acoustic
    abs_lmda2 = np.abs(lmda[:, 2])
    coef2 = -rho_roe / (2 * a_roe)
    flux_correction[:, 0] += dv[:, 2] * abs_lmda2 * coef2 * 1
    flux_correction[:, 1] += dv[:, 2] * abs_lmda2 * coef2 * (u_roe - a_roe)
    flux_correction[:, 2] += dv[:, 2] * abs_lmda2 * coef2 * (h_roe - a_roe * u_roe)

    flux = 0.5 * (F_L + F_R) - 0.5 * flux_correction

    # Mark invalid
    flux[~valid] = np.nan

    return flux, valid


def get_droplet_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter ranges for the air/water droplet case (in_1Dcdrop).

    Based on simulation analysis:
    - Density: 0.9 to 1000 kg/m³ (air ~1.2, water ~1000)
    - Velocity: 0 to 100 m/s (liquid moving at 100 m/s)
    - Pressure: 65,000 to 150,000 Pa (around atmospheric)

    We add 20% margin for robustness.
    """
    # Observed ranges from simulation
    rho_min, rho_max = 0.9, 1000.0
    u_min, u_max = 0.0, 100.0
    p_min, p_max = 65000.0, 150000.0

    # Add 20% margin
    margin = 0.2
    rho_range = rho_max - rho_min
    u_range = u_max - u_min
    p_range = p_max - p_min

    ranges = {
        'rho': (max(0.1, rho_min - margin * rho_range), rho_max + margin * rho_range),
        'u': (u_min - margin * u_range, u_max + margin * u_range),
        'p': (max(10000, p_min - margin * p_range), p_max + margin * p_range),
    }

    print("Air/water droplet parameter ranges (with 20% margin):")
    for var, (vmin, vmax) in ranges.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return ranges


def generate_droplet_samples_vectorized(
    n_samples: int = 2000000,
    gamma: float = 1.4,
    batch_size: int = 500000,
    log_sampling: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training samples for the air/water droplet case.

    Uses log-uniform sampling for density to better cover the 1000:1 ratio.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate
    gamma : float
        Specific heat ratio (1.4 for both air and water in this case)
    batch_size : int
        Process in batches for memory efficiency
    log_sampling : bool
        Use log-uniform sampling for density (recommended for high density ratios)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs: Shape (N, 6) - [rho_L, u_L, p_L, rho_R, u_R, p_R]
        outputs: Shape (N, 3) - [flux_rho, flux_rhou, flux_E]
    """
    ranges = get_droplet_parameter_ranges()

    print(f"\nGenerating {n_samples:,} samples for droplet case...")
    print(f"  Gamma: {gamma}")
    print(f"  Log sampling for density: {log_sampling}")
    print(f"  Batch size: {batch_size:,}")

    all_inputs = []
    all_outputs = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    valid_total = 0

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_samples)
        current_batch = end - start

        # Generate random states
        if log_sampling:
            # Log-uniform sampling for density (better coverage of 1000:1 ratio)
            log_rho_min = np.log10(ranges['rho'][0])
            log_rho_max = np.log10(ranges['rho'][1])
            rho_L = 10 ** np.random.uniform(log_rho_min, log_rho_max, current_batch)
            rho_R = 10 ** np.random.uniform(log_rho_min, log_rho_max, current_batch)
        else:
            rho_L = np.random.uniform(ranges['rho'][0], ranges['rho'][1], current_batch)
            rho_R = np.random.uniform(ranges['rho'][0], ranges['rho'][1], current_batch)

        u_L = np.random.uniform(ranges['u'][0], ranges['u'][1], current_batch)
        u_R = np.random.uniform(ranges['u'][0], ranges['u'][1], current_batch)
        p_L = np.random.uniform(ranges['p'][0], ranges['p'][1], current_batch)
        p_R = np.random.uniform(ranges['p'][0], ranges['p'][1], current_batch)

        gamma_arr = np.full(current_batch, gamma)

        # Vectorized flux computation
        flux, valid = roe_flux1D_vectorized(gamma_arr, rho_L, u_L, p_L, rho_R, u_R, p_R)

        # Filter valid samples
        valid_idx = valid & ~np.any(np.isnan(flux), axis=1) & ~np.any(np.isinf(flux), axis=1)
        n_valid = np.sum(valid_idx)
        valid_total += n_valid

        if n_valid > 0:
            inputs_batch = np.column_stack([
                rho_L[valid_idx], u_L[valid_idx], p_L[valid_idx],
                rho_R[valid_idx], u_R[valid_idx], p_R[valid_idx]
            ])
            all_inputs.append(inputs_batch)
            all_outputs.append(flux[valid_idx])

        if (b + 1) % 2 == 0 or b == n_batches - 1:
            print(f"  Batch {b+1}/{n_batches}, valid samples: {valid_total:,}")

    # Concatenate all
    inputs = np.vstack(all_inputs)
    outputs = np.vstack(all_outputs)

    print(f"\nTotal valid samples: {len(inputs):,}")

    # Shuffle
    print("Shuffling...")
    perm = np.random.permutation(len(inputs))
    inputs = inputs[perm]
    outputs = outputs[perm]

    return inputs, outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate droplet training data')
    parser.add_argument('--n-samples', type=int, default=2000000,
                        help='Number of samples to generate')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='Specific heat ratio')
    parser.add_argument('--output', type=str, default='data/droplet_flux_data.npz',
                        help='Output file path')
    parser.add_argument('--no-log-sampling', action='store_true',
                        help='Disable log-uniform sampling for density')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Generate data
    inputs, outputs = generate_droplet_samples_vectorized(
        n_samples=args.n_samples,
        gamma=args.gamma,
        log_sampling=not args.no_log_sampling,
    )

    # Save
    ranges = get_droplet_parameter_ranges()
    np.savez(args.output,
             inputs=inputs,
             outputs=outputs,
             ranges=ranges,
             gamma=args.gamma,
             config='in_1Dcdrop')

    print(f"\nSaved dataset to {args.output}")
    print(f"\nDataset summary:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Input ranges:")
    for i, name in enumerate(['rho_L', 'u_L', 'p_L', 'rho_R', 'u_R', 'p_R']):
        print(f"    {name}: [{inputs[:, i].min():.6g}, {inputs[:, i].max():.6g}]")
    print(f"  Output ranges:")
    for i, name in enumerate(['flux_rho', 'flux_rhou', 'flux_E']):
        print(f"    {name}: [{outputs[:, i].min():.6g}, {outputs[:, i].max():.6g}]")


if __name__ == '__main__':
    main()
