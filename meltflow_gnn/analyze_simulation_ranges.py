"""
Analyze parameter ranges from a MeltFlow simulation.

Runs the simulation and logs all primitive variable states (rho, u, p)
to determine the actual operating range for GNN training.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meltflow.functions.parameters import create_parameters, apply_defaults
from meltflow.functions.grid import grid_setup
from meltflow.functions.state_var import state_var
from meltflow.functions.ghost_fluid import ghost_GFM, real_GFM, extrp_vel
from meltflow.functions.level_set import advc, reinit_fast
from meltflow.functions.solver import run_slvr, timestep
from meltflow.input.configs import load_config


def run_simulation_with_logging(config_name: str = 'in_1Dcdrop'):
    """
    Run simulation and log all primitive variable states.
    """
    # Collect all values
    all_rho = []
    all_u = []
    all_p = []

    start_time = time.time()

    # Load configuration
    config, init_func = load_config(config_name)

    # Grid setup
    n_var, n, x, dx, U, phi = grid_setup(config)
    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx

    # Create parameters
    config = apply_defaults(config, config['n_dim'])
    prm = create_parameters(config)
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Create mesh
    n_dim = prm.n_dim
    X = x

    # Apply initial conditions
    U, phi = init_func(x, U, phi)

    # Initialize counters
    t = 0.0
    it = 0
    cntr_r = 0

    print(f"Running {config_name} simulation...")
    print(f"  Grid points: {n}")
    print(f"  Final time: {prm.t_f}")
    print(f"  CFL: {prm.cfl}")
    print(f"  Gamma: {prm.c_EoS}")
    print()

    # Log initial state
    all_rho.extend(U[0, :].flatten())
    all_u.extend(U[1, :].flatten())
    all_p.extend(U[2, :].flatten())

    # Calculate initial conserved variables
    W = state_var(prm, "cons", n_var, phi, U)

    # Main time loop
    while t < prm.t_f:
        # Compute speed of sound
        a = state_var(prm, "SoS", 1, phi, U)

        # Calculate time step
        dt = timestep(prm, U, a)
        if t + dt > prm.t_f:
            dt = prm.t_f - t

        t += dt

        # Ghost Fluid Method
        UU, WW = ghost_GFM(prm, X, phi, U)

        # Run solvers
        WW = run_slvr(prm, dt, X, phi, UU, WW)

        # Extrapolate velocity field
        V = extrp_vel(prm, X, phi, WW)

        # Advect level set
        phi = advc(prm, dt, V, phi)

        # Reinitialization
        cntr_r += 1
        if cntr_r >= prm.n_r and prm.n_r > 0:
            cntr_r = 0
            phi = reinit_fast(prm, phi)

        # Reassemble real domain
        W = real_GFM(prm, phi, WW)

        # Compute primitive variables
        U = state_var(prm, "prim", n_var, phi, W)

        # Log current state - only valid values (exclude NaN/Inf)
        rho_valid = U[0, :].flatten()
        u_valid = U[1, :].flatten()
        p_valid = U[2, :].flatten()

        # Filter out invalid values
        mask = np.isfinite(rho_valid) & np.isfinite(u_valid) & np.isfinite(p_valid)
        mask &= (rho_valid > 0) & (p_valid > 0)  # Physical constraints

        all_rho.extend(rho_valid[mask])
        all_u.extend(u_valid[mask])
        all_p.extend(p_valid[mask])

        it += 1

        if it % 20 == 0:
            print(f"  Iteration {it}, t = {t:.6e}")

    elapsed = time.time() - start_time

    print()
    print(f"Simulation completed in {elapsed:.2f} seconds")
    print(f"Total iterations: {it}")
    print(f"Total state samples: {len(all_rho)}")

    # Convert to arrays
    all_rho = np.array(all_rho)
    all_u = np.array(all_u)
    all_p = np.array(all_p)

    stats = {
        'rho': {
            'min': float(np.min(all_rho)),
            'max': float(np.max(all_rho)),
            'mean': float(np.mean(all_rho)),
            'std': float(np.std(all_rho)),
        },
        'u': {
            'min': float(np.min(all_u)),
            'max': float(np.max(all_u)),
            'mean': float(np.mean(all_u)),
            'std': float(np.std(all_u)),
        },
        'p': {
            'min': float(np.min(all_p)),
            'max': float(np.max(all_p)),
            'mean': float(np.mean(all_p)),
            'std': float(np.std(all_p)),
        },
        'total_samples': len(all_rho),
    }

    return stats, prm


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze simulation parameter ranges')
    parser.add_argument('--config', '-c', type=str, default='in_1Dcdrop',
                       help='Configuration name to run')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for ranges (optional)')

    args = parser.parse_args()

    stats, prm = run_simulation_with_logging(args.config)

    print()
    print("=" * 60)
    print("PARAMETER RANGES OBSERVED DURING SIMULATION")
    print("=" * 60)
    print()
    print(f"Configuration: {args.config}")
    print(f"Gamma (gas): {prm.c_EoS[0]}")
    print(f"Total state samples: {stats['total_samples']}")
    print()
    print("Density (rho) [kg/m³]:")
    print(f"  Min:  {stats['rho']['min']:.6g}")
    print(f"  Max:  {stats['rho']['max']:.6g}")
    print(f"  Mean: {stats['rho']['mean']:.6g}")
    print(f"  Std:  {stats['rho']['std']:.6g}")
    print()
    print("Velocity (u) [m/s]:")
    print(f"  Min:  {stats['u']['min']:.6g}")
    print(f"  Max:  {stats['u']['max']:.6g}")
    print(f"  Mean: {stats['u']['mean']:.6g}")
    print(f"  Std:  {stats['u']['std']:.6g}")
    print()
    print("Pressure (p) [Pa]:")
    print(f"  Min:  {stats['p']['min']:.6g}")
    print(f"  Max:  {stats['p']['max']:.6g}")
    print(f"  Mean: {stats['p']['mean']:.6g}")
    print(f"  Std:  {stats['p']['std']:.6g}")
    print()

    # Suggested training ranges (with some margin)
    margin = 0.2  # 20% margin
    print("=" * 60)
    print("SUGGESTED TRAINING RANGES (with 20% margin)")
    print("=" * 60)
    print()

    rho_range = stats['rho']['max'] - stats['rho']['min']
    u_range = stats['u']['max'] - stats['u']['min']
    p_range = stats['p']['max'] - stats['p']['min']

    suggested = {
        'rho': (
            max(0.01, stats['rho']['min'] - margin * rho_range),
            stats['rho']['max'] + margin * rho_range
        ),
        'u': (
            stats['u']['min'] - margin * u_range,
            stats['u']['max'] + margin * u_range
        ),
        'p': (
            max(1000, stats['p']['min'] - margin * p_range),
            stats['p']['max'] + margin * p_range
        ),
    }

    print(f"rho: [{suggested['rho'][0]:.6g}, {suggested['rho'][1]:.6g}]")
    print(f"u:   [{suggested['u'][0]:.6g}, {suggested['u'][1]:.6g}]")
    print(f"p:   [{suggested['p'][0]:.6g}, {suggested['p'][1]:.6g}]")
    print()

    if args.output:
        np.savez(args.output,
                 stats=stats,
                 suggested_ranges=suggested,
                 gamma=prm.c_EoS[0],
                 config=args.config)
        print(f"Saved to {args.output}")

    return stats, suggested


if __name__ == '__main__':
    main()
