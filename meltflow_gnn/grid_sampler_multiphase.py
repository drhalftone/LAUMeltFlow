"""
Uniform grid sampling for multiphase GNN flux training data.

Extends parameter ranges to cover the two-fluid Sod shock tube case (in_1Dsod2fl).
Supports importance sampling for denser coverage of core operating regions.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import os


def roe_flux1D_vectorized(gam: np.ndarray, rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                          rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray) -> np.ndarray:
    """
    Vectorized Roe flux for 1D Euler equations.

    All inputs should be 1D arrays of the same length.
    Returns (N, 3) array of fluxes.
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

    # Mark invalid samples
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

    # Eigenvectors - compute flux correction term
    # r0 = [1, u_roe, 0.5*u_roe^2]
    # r1 = rho_roe/(2*a_roe) * [1, u_roe+a_roe, h_roe+a_roe*u_roe]
    # r2 = -rho_roe/(2*a_roe) * [1, u_roe-a_roe, h_roe-a_roe*u_roe]

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


def roe_flux1D(gam: float, rho_L: float, u_L: float, p_L: float,
               rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """
    Calculate Roe flux for 1D Euler equations.

    Parameters
    ----------
    gam : float
        Specific heat ratio
    rho_L, u_L, p_L : float
        Left state (primitive variables)
    rho_R, u_R, p_R : float
        Right state (primitive variables)

    Returns
    -------
    np.ndarray
        Flux [flux_rho, flux_rhou, flux_E]
    """
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

    if a_roe_sq <= 0:
        return np.array([np.nan, np.nan, np.nan])

    a_roe = np.sqrt(a_roe_sq)

    # Differences
    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    # Wave strengths
    dv = np.array([
        drho - dp / a_roe**2,
        du + dp / (rho_roe * a_roe),
        du - dp / (rho_roe * a_roe)
    ])

    # Eigenvalues
    lmda = np.array([u_roe, u_roe + a_roe, u_roe - a_roe])

    # Eigenvectors
    r = np.array([
        [1, u_roe, 0.5 * u_roe**2],
        rho_roe / (2 * a_roe) * np.array([1, u_roe + a_roe, h_roe + a_roe * u_roe]),
        -rho_roe / (2 * a_roe) * np.array([1, u_roe - a_roe, h_roe - a_roe * u_roe])
    ])

    # Left and right fluxes
    F_L = np.array([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])
    F_R = np.array([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    # Roe flux
    flux_sum = np.zeros(3)
    for j in range(3):
        flux_sum += dv[j] * np.abs(lmda[j]) * r[j]

    return 0.5 * (F_L + F_R) - 0.5 * flux_sum


def get_multiphase_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter ranges that cover both single-fluid and two-fluid Sod cases.

    Two-fluid case (in_1Dsod2fl):
        Left:  rho = 5.36 kg/m³, p = 303,975 Pa
        Right: rho = 1.18 kg/m³, p = 101,325 Pa

    Single-fluid case (in_1Dsod1fl):
        Left:  rho = 1.0, p = 100,000 Pa
        Right: rho = 0.125, p = 10,000 Pa

    We expand ranges with 20% margin.
    """
    ranges = {
        'rho': (0.05, 7.0),        # Density [kg/m³] - expanded for two-fluid
        'u': (-200.0, 500.0),      # Velocity [m/s] - expanded
        'p': (5000.0, 400000.0)    # Pressure [Pa] - expanded for two-fluid
    }

    print(f"Multiphase parameter ranges:")
    for var, (vmin, vmax) in ranges.items():
        print(f"  {var}: [{vmin:.6g}, {vmax:.6g}]")

    return ranges


def get_core_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get core parameter ranges (original single-fluid operating region).
    Used for importance sampling to ensure dense coverage of common cases.
    """
    return {
        'rho': (0.05, 1.5),        # Density [kg/m³] - original range
        'u': (-150.0, 350.0),      # Velocity [m/s] - original range
        'p': (5000.0, 150000.0)    # Pressure [Pa] - original range
    }


def generate_importance_samples_vectorized(
    n_samples: int = 20000000,
    gamma_values: Optional[List[float]] = None,
    core_fraction: float = 0.6,
    batch_size: int = 1000000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples with importance sampling: denser in core region.

    Uses vectorized Roe flux computation for speed.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate
    gamma_values : List[float]
        List of gamma values to sample from
    core_fraction : float
        Fraction of samples from core (original) range (0-1)
    batch_size : int
        Process in batches for memory efficiency

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs: Shape (N, 7) - [rho_L, u_L, p_L, rho_R, u_R, p_R, gamma]
        outputs: Shape (N, 3) - [flux_rho, flux_rhou, flux_E]
    """
    core_ranges = get_core_parameter_ranges()
    full_ranges = get_multiphase_parameter_ranges()

    n_core = int(n_samples * core_fraction)
    n_extended = n_samples - n_core

    print(f"\nImportance sampling strategy:")
    print(f"  Core region ({core_fraction*100:.0f}%): {n_core:,} samples")
    print(f"    rho: [{core_ranges['rho'][0]}, {core_ranges['rho'][1]}]")
    print(f"    u:   [{core_ranges['u'][0]}, {core_ranges['u'][1]}]")
    print(f"    p:   [{core_ranges['p'][0]}, {core_ranges['p'][1]}]")
    print(f"  Extended region ({(1-core_fraction)*100:.0f}%): {n_extended:,} samples")
    print(f"    rho: [{full_ranges['rho'][0]}, {full_ranges['rho'][1]}]")
    print(f"    u:   [{full_ranges['u'][0]}, {full_ranges['u'][1]}]")
    print(f"    p:   [{full_ranges['p'][0]}, {full_ranges['p'][1]}]")
    print(f"  Gamma values: {gamma_values}")

    all_inputs = []
    all_outputs = []

    def generate_batch(ranges, n, batch_name):
        """Generate a batch of samples from given ranges."""
        nonlocal all_inputs, all_outputs

        n_batches = (n + batch_size - 1) // batch_size
        valid_total = 0

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n)
            current_batch = end - start

            # Generate random states
            rho_L = np.random.uniform(ranges['rho'][0], ranges['rho'][1], current_batch)
            u_L = np.random.uniform(ranges['u'][0], ranges['u'][1], current_batch)
            p_L = np.random.uniform(ranges['p'][0], ranges['p'][1], current_batch)
            rho_R = np.random.uniform(ranges['rho'][0], ranges['rho'][1], current_batch)
            u_R = np.random.uniform(ranges['u'][0], ranges['u'][1], current_batch)
            p_R = np.random.uniform(ranges['p'][0], ranges['p'][1], current_batch)
            gamma_arr = np.random.choice(gamma_values, current_batch)

            # Vectorized flux computation
            flux, valid = roe_flux1D_vectorized(gamma_arr, rho_L, u_L, p_L, rho_R, u_R, p_R)

            # Filter valid samples
            valid_idx = valid & ~np.any(np.isnan(flux), axis=1) & ~np.any(np.isinf(flux), axis=1)
            n_valid = np.sum(valid_idx)
            valid_total += n_valid

            if n_valid > 0:
                inputs_batch = np.column_stack([
                    rho_L[valid_idx], u_L[valid_idx], p_L[valid_idx],
                    rho_R[valid_idx], u_R[valid_idx], p_R[valid_idx],
                    gamma_arr[valid_idx]
                ])
                all_inputs.append(inputs_batch)
                all_outputs.append(flux[valid_idx])

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                print(f"    {batch_name}: batch {b+1}/{n_batches}, valid so far: {valid_total:,}")

        return valid_total

    print(f"\nGenerating core samples...")
    n_core_valid = generate_batch(core_ranges, n_core, "Core")

    print(f"\nGenerating extended samples...")
    n_ext_valid = generate_batch(full_ranges, n_extended, "Extended")

    # Concatenate all
    inputs = np.vstack(all_inputs)
    outputs = np.vstack(all_outputs)

    print(f"\nTotal valid samples: {len(inputs):,}")
    print(f"  From core: ~{n_core_valid:,}")
    print(f"  From extended: ~{n_ext_valid:,}")

    # Shuffle to mix core and extended samples
    print("Shuffling...")
    perm = np.random.permutation(len(inputs))
    inputs = inputs[perm]
    outputs = outputs[perm]

    return inputs, outputs


def generate_uniform_samples_random(
    ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 1000000,
    gamma_values: Optional[List[float]] = None,
    gam: float = 1.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random uniform samples (more efficient than grid for high dimensions).

    Parameters
    ----------
    ranges : Dict
        Parameter ranges for rho, u, p
    n_samples : int
        Total number of samples to generate
    gamma_values : List[float], optional
        List of gamma values to sample from
    gam : float
        Fixed gamma (used if gamma_values is None)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        inputs: Shape (N, 6) or (N, 7) if gamma_values provided
        outputs: Shape (N, 3)
    """
    include_gamma = gamma_values is not None

    if include_gamma:
        print(f"Generating {n_samples:,} random samples with gamma as input...")
        print(f"  Gamma values: {gamma_values}")
    else:
        print(f"Generating {n_samples:,} random samples (gamma={gam})...")

    # Generate random states
    rho_L = np.random.uniform(ranges['rho'][0], ranges['rho'][1], n_samples)
    u_L = np.random.uniform(ranges['u'][0], ranges['u'][1], n_samples)
    p_L = np.random.uniform(ranges['p'][0], ranges['p'][1], n_samples)
    rho_R = np.random.uniform(ranges['rho'][0], ranges['rho'][1], n_samples)
    u_R = np.random.uniform(ranges['u'][0], ranges['u'][1], n_samples)
    p_R = np.random.uniform(ranges['p'][0], ranges['p'][1], n_samples)

    if include_gamma:
        # Random gamma from the provided values
        gamma_arr = np.random.choice(gamma_values, n_samples)
    else:
        gamma_arr = np.full(n_samples, gam)

    # Compute Roe flux for each sample
    inputs = []
    outputs = []
    valid_count = 0
    invalid_count = 0

    print("Computing Roe flux for each sample...")
    for i in range(n_samples):
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,} / {n_samples:,}")

        try:
            flux = roe_flux1D(gamma_arr[i], rho_L[i], u_L[i], p_L[i],
                              rho_R[i], u_R[i], p_R[i])

            if np.any(np.isnan(flux)) or np.any(np.isinf(flux)):
                invalid_count += 1
                continue

            if include_gamma:
                inputs.append([rho_L[i], u_L[i], p_L[i], rho_R[i], u_R[i], p_R[i], gamma_arr[i]])
            else:
                inputs.append([rho_L[i], u_L[i], p_L[i], rho_R[i], u_R[i], p_R[i]])
            outputs.append(flux)
            valid_count += 1

        except Exception:
            invalid_count += 1
            continue

    print(f"Generated {valid_count:,} valid samples, {invalid_count:,} invalid")

    return np.array(inputs), np.array(outputs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate multiphase training data')
    parser.add_argument('--n-samples', type=int, default=2000000,
                        help='Number of random samples')
    parser.add_argument('--gamma-values', type=float, nargs='+',
                        default=[1.2, 1.25, 1.289, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.67],
                        help='Gamma values to sample from')
    parser.add_argument('--output', type=str, default='data/multiphase_flux_data.npz',
                        help='Output file path')
    parser.add_argument('--importance-sampling', action='store_true',
                        help='Use importance sampling (denser in core region)')
    parser.add_argument('--core-fraction', type=float, default=0.6,
                        help='Fraction of samples from core region (with --importance-sampling)')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.importance_sampling:
        # Use importance sampling with vectorized computation
        inputs, outputs = generate_importance_samples_vectorized(
            n_samples=args.n_samples,
            gamma_values=args.gamma_values,
            core_fraction=args.core_fraction
        )
        ranges = get_multiphase_parameter_ranges()
    else:
        # Original uniform sampling
        ranges = get_multiphase_parameter_ranges()
        inputs, outputs = generate_uniform_samples_random(
            ranges=ranges,
            n_samples=args.n_samples,
            gamma_values=args.gamma_values
        )

    # Save
    save_data = {
        'inputs': inputs,
        'outputs': outputs,
        'ranges': ranges,
        'gamma_values': np.array(args.gamma_values),
        'has_gamma_input': True,
        'importance_sampling': args.importance_sampling,
        'core_fraction': args.core_fraction if args.importance_sampling else 0.0
    }
    np.savez(args.output, **save_data)
    print(f"\nSaved dataset to {args.output}")

    print(f"\nDataset summary:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Input ranges:")
    for i, name in enumerate(['rho_L', 'u_L', 'p_L', 'rho_R', 'u_R', 'p_R', 'gamma']):
        if i < inputs.shape[1]:
            print(f"    {name}: [{inputs[:, i].min():.6g}, {inputs[:, i].max():.6g}]")


if __name__ == '__main__':
    main()
