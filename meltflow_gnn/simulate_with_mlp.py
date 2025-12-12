"""
Run Euler simulation using the trained MLP flux model.

Replaces the analytical Roe flux with the learned MLP.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import sys
import os

# Import the MLP model
from train_uniform import FluxMLP


def load_trained_model(model_path: str = 'flux_model_uniform.npz') -> Tuple[FluxMLP, Dict]:
    """Load the trained MLP model and normalization stats."""
    model, stats, config = FluxMLP.load(model_path)
    print(f"Loaded model from {model_path}")
    print(f"  Architecture: {config.get('input_dim', 6)} -> {config.get('hidden_dims', [])} -> {config.get('output_dim', 3)}")
    return model, stats


def mlp_flux(model: FluxMLP, stats: Dict,
             rho_L: float, u_L: float, p_L: float,
             rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """
    Compute flux using the trained MLP.

    Parameters
    ----------
    model : FluxMLP
        Trained model
    stats : Dict
        Normalization statistics
    rho_L, u_L, p_L : float
        Left state (primitive variables)
    rho_R, u_R, p_R : float
        Right state (primitive variables)

    Returns
    -------
    np.ndarray
        Flux [flux_rho, flux_rhou, flux_E]
    """
    # Create input
    X = np.array([[rho_L, u_L, p_L, rho_R, u_R, p_R]])

    # Normalize
    X_norm = (X - stats['X_mean']) / stats['X_std']

    # Predict
    Y_norm = model.forward(X_norm)

    # Denormalize
    Y = Y_norm * stats['Y_std'] + stats['Y_mean']

    return Y[0]


def roe_flux_analytical(gam: float, rho_L: float, u_L: float, p_L: float,
                        rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """
    Analytical Roe flux for comparison.
    """
    # Convert to conserved
    E_L = p_L / (gam - 1) + 0.5 * rho_L * u_L**2
    E_R = p_R / (gam - 1) + 0.5 * rho_R * u_R**2

    W_L = np.array([rho_L, rho_L * u_L, E_L])
    W_R = np.array([rho_R, rho_R * u_R, E_R])

    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    h_L = (E_L + p_L) / rho_L
    h_R = (E_R + p_R) / rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) / denom
    a_roe = np.sqrt((gam - 1) * (h_roe - 0.5 * u_roe**2))
    rho_roe = sqrt_rho_L * sqrt_rho_R

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


def run_simulation(
    model: FluxMLP,
    stats: Dict,
    nx: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_final: float = 0.00075,
    cfl: float = 0.5,
    gamma: float = 1.4,
    use_mlp: bool = True,
    use_analytical: bool = False
) -> Dict:
    """
    Run 1D Euler simulation with Sod initial conditions.

    Parameters
    ----------
    model : FluxMLP
        Trained flux model
    stats : Dict
        Normalization statistics
    nx : int
        Number of grid cells
    x_min, x_max : float
        Domain bounds
    t_final : float
        Final simulation time
    cfl : float
        CFL number
    gamma : float
        Specific heat ratio
    use_mlp : bool
        Use MLP for flux computation
    use_analytical : bool
        Also compute analytical solution for comparison

    Returns
    -------
    Dict
        Simulation results
    """
    # Grid setup
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx/2, x_max - dx/2, nx)  # Cell centers

    # Primitive variables: [rho, u, p]
    rho = np.zeros(nx)
    u = np.zeros(nx)
    p = np.zeros(nx)

    # Sod shock tube initial conditions
    for i in range(nx):
        if x[i] < 0.5:
            rho[i] = 1.0
            u[i] = 0.0
            p[i] = 100000.0
        else:
            rho[i] = 0.125
            u[i] = 0.0
            p[i] = 10000.0

    # Also run analytical if requested
    if use_analytical:
        rho_ana = rho.copy()
        u_ana = u.copy()
        p_ana = p.copy()

    # Time stepping
    t = 0.0
    n_steps = 0
    history = {'t': [], 'rho': [], 'u': [], 'p': []}

    # Save initial state
    history['t'].append(t)
    history['rho'].append(rho.copy())
    history['u'].append(u.copy())
    history['p'].append(p.copy())

    print(f"\nRunning simulation...")
    print(f"  Grid: {nx} cells, dx = {dx:.4f}")
    print(f"  Final time: {t_final}")
    print(f"  Using: {'MLP' if use_mlp else 'Analytical'} flux")

    while t < t_final:
        # Compute timestep (CFL condition)
        a = np.sqrt(gamma * p / rho)  # Sound speed
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed

        if t + dt > t_final:
            dt = t_final - t

        # Compute fluxes at interfaces
        flux = np.zeros((nx + 1, 3))  # nx+1 interfaces

        for i in range(nx + 1):
            # Get left and right states (with boundary conditions)
            if i == 0:
                # Left boundary (transmissive)
                rho_L, u_L, p_L = rho[0], u[0], p[0]
                rho_R, u_R, p_R = rho[0], u[0], p[0]
            elif i == nx:
                # Right boundary (transmissive)
                rho_L, u_L, p_L = rho[-1], u[-1], p[-1]
                rho_R, u_R, p_R = rho[-1], u[-1], p[-1]
            else:
                rho_L, u_L, p_L = rho[i-1], u[i-1], p[i-1]
                rho_R, u_R, p_R = rho[i], u[i], p[i]

            # Compute flux
            if use_mlp:
                flux[i] = mlp_flux(model, stats, rho_L, u_L, p_L, rho_R, u_R, p_R)
            else:
                flux[i] = roe_flux_analytical(gamma, rho_L, u_L, p_L, rho_R, u_R, p_R)

        # Convert to conserved variables
        E = p / (gamma - 1) + 0.5 * rho * u**2
        W = np.array([rho, rho * u, E])  # Shape: (3, nx)

        # Update conserved variables (finite volume)
        for i in range(nx):
            W[:, i] = W[:, i] - dt / dx * (flux[i+1] - flux[i])

        # Check for invalid states
        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            print(f"  WARNING: Invalid state at t={t:.6f}, step={n_steps}")
            print(f"    Min rho: {W[0].min():.6f}")
            break

        # Convert back to primitive
        rho = W[0]
        u = W[1] / rho
        E = W[2]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        # Check pressure
        if np.any(p <= 0):
            print(f"  WARNING: Negative pressure at t={t:.6f}, step={n_steps}")
            break

        t += dt
        n_steps += 1

        if n_steps % 50 == 0:
            print(f"  Step {n_steps}: t = {t:.6f}, dt = {dt:.2e}")

    print(f"  Completed: {n_steps} steps, final t = {t:.6f}")

    # Save final state
    history['t'].append(t)
    history['rho'].append(rho.copy())
    history['u'].append(u.copy())
    history['p'].append(p.copy())

    return {
        'x': x,
        'rho': rho,
        'u': u,
        'p': p,
        't': t,
        'n_steps': n_steps,
        'history': history
    }


def plot_comparison(mlp_result: Dict, analytical_result: Dict = None, save_path: str = None):
    """Plot MLP vs analytical results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = mlp_result['x']

    # Density
    axes[0].plot(x, mlp_result['rho'], 'b-', label='MLP', linewidth=2)
    if analytical_result:
        axes[0].plot(x, analytical_result['rho'], 'r--', label='Analytical Roe', linewidth=2)
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('Density [kg/m³]')
    axes[0].set_title('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Velocity
    axes[1].plot(x, mlp_result['u'], 'b-', label='MLP', linewidth=2)
    if analytical_result:
        axes[1].plot(x, analytical_result['u'], 'r--', label='Analytical Roe', linewidth=2)
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('Velocity [m/s]')
    axes[1].set_title('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    # Pressure
    axes[2].plot(x, mlp_result['p'], 'b-', label='MLP', linewidth=2)
    if analytical_result:
        axes[2].plot(x, analytical_result['p'], 'r--', label='Analytical Roe', linewidth=2)
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('Pressure [Pa]')
    axes[2].set_title('Pressure')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(f'Sod Shock Tube at t = {mlp_result["t"]:.6f} s', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run simulation with MLP flux')
    parser.add_argument('--model', type=str, default='flux_model_uniform.npz',
                        help='Path to trained model')
    parser.add_argument('--nx', type=int, default=100,
                        help='Number of grid cells')
    parser.add_argument('--t-final', type=float, default=0.00075,
                        help='Final simulation time')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number')
    parser.add_argument('--compare', action='store_true',
                        help='Also run analytical Roe for comparison')
    parser.add_argument('--output', type=str, default='mlp_simulation.png',
                        help='Output plot path')

    args = parser.parse_args()

    print("=" * 60)
    print("MLP Flux Simulation")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model, stats = load_trained_model(args.model)

    # Run MLP simulation
    print("\n2. Running MLP simulation...")
    mlp_result = run_simulation(
        model, stats,
        nx=args.nx,
        t_final=args.t_final,
        cfl=args.cfl,
        use_mlp=True
    )

    # Run analytical simulation for comparison
    analytical_result = None
    if args.compare:
        print("\n3. Running analytical Roe simulation...")
        analytical_result = run_simulation(
            model, stats,
            nx=args.nx,
            t_final=args.t_final,
            cfl=args.cfl,
            use_mlp=False
        )

    # Plot results
    print("\n4. Plotting results...")
    plot_comparison(mlp_result, analytical_result, save_path=args.output)

    # Print comparison metrics
    if analytical_result:
        print("\n5. Comparison metrics:")
        rho_err = np.abs(mlp_result['rho'] - analytical_result['rho'])
        u_err = np.abs(mlp_result['u'] - analytical_result['u'])
        p_err = np.abs(mlp_result['p'] - analytical_result['p'])

        print(f"  Density MAE: {rho_err.mean():.6f}")
        print(f"  Velocity MAE: {u_err.mean():.6f}")
        print(f"  Pressure MAE: {p_err.mean():.6f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
