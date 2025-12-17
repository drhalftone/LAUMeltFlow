"""
Test multiphase (two-fluid) simulation with MLP flux predictor.

Compares analytical Roe flux vs gamma-parameterized MLP on the in_1Dsod2fl case.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meltflow.input.configs import load_config
from meltflow.functions.grid import grid_setup
from meltflow.functions.parameters import create_parameters, apply_defaults


class FluxMLP(nn.Module):
    """MLP for flux prediction (must match training architecture)."""

    def __init__(self, input_dim: int = 7, output_dim: int = 3,
                 hidden_dims: List[int] = [256, 256, 256, 256, 256],
                 activation: str = 'gelu'):
        super().__init__()

        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_gamma_model(model_path: str, device: str = 'cpu') -> Tuple[FluxMLP, Dict, Dict]:
    """Load trained gamma-parameterized model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    stats = checkpoint['stats']

    model = FluxMLP(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Input dim: {config['input_dim']} (has_gamma: {config.get('has_gamma', False)})")

    return model, stats, config


def roe_flux_analytical(gamma: float, rho_L: float, u_L: float, p_L: float,
                        rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """Analytical Roe flux for a single interface."""
    # Convert to conserved
    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2

    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    h_L = (E_L + p_L) / rho_L
    h_R = (E_R + p_R) / rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) / denom
    a_roe = np.sqrt((gamma - 1) * (h_roe - 0.5 * u_roe**2))
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


def mlp_flux_batch(model: FluxMLP, stats: Dict,
                   rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                   rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray,
                   gamma: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Compute flux for batch of states using MLP (supports per-interface gamma)."""
    n = len(rho_L)

    # Build input array with gamma (can be different per interface)
    X = np.column_stack([rho_L, u_L, p_L, rho_R, u_R, p_R, gamma])

    # Normalize
    X_mean = stats['X_mean']
    X_std = stats['X_std']
    X_norm = (X - X_mean) / X_std

    # Convert to tensor and predict
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        Y_norm = model(X_tensor)

    # Denormalize
    Y_mean = stats['Y_mean']
    Y_std = stats['Y_std']
    Y = Y_norm.cpu().numpy() * Y_std + Y_mean

    return Y


def run_multiphase_simulation(
    config_name: str = 'in_1Dsod2fl',
    model: FluxMLP = None,
    stats: Dict = None,
    use_mlp: bool = True,
    cfl: float = 0.3,
    device: str = 'cpu'
) -> Dict:
    """
    Run multiphase Euler simulation.

    Uses Ghost Fluid Method for interface treatment.
    """
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

    # Apply initial conditions
    U, phi = init_func(x, U, phi)

    # Get gamma values for each fluid
    gamma_1 = prm.c_EoS[0]  # Left fluid (phi > 0)
    gamma_2 = prm.c_EoS[1]  # Right fluid (phi <= 0)

    print(f"\nMultiphase simulation: {config_name}")
    print(f"  Grid: {n} cells, dx = {dx:.6f}")
    print(f"  Gamma: fluid 1 = {gamma_1}, fluid 2 = {gamma_2}")
    print(f"  Final time: {prm.t_f}")
    print(f"  Using: {'MLP' if use_mlp else 'Analytical Roe'}")

    t = 0.0
    n_steps = 0

    # Store initial state
    rho_init = U[0].copy()
    u_init = U[1].copy()
    p_init = U[2].copy()
    phi_init = phi.copy()

    while t < prm.t_f:
        rho, u, p = U[0], U[1], U[2]

        # Determine gamma at each interface based on phi
        gamma_interface = np.zeros(n + 1)
        for i in range(n + 1):
            if i == 0:
                phi_i = phi[0]
            elif i == n:
                phi_i = phi[-1]
            else:
                phi_i = 0.5 * (phi[i-1] + phi[i])

            gamma_interface[i] = gamma_1 if phi_i > 0 else gamma_2

        # Compute max wave speed for CFL
        gamma_cells = np.where(phi > 0, gamma_1, gamma_2)
        a = np.sqrt(gamma_cells * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed

        if t + dt > prm.t_f:
            dt = prm.t_f - t

        # Compute interface fluxes
        flux = np.zeros((n + 1, 3))

        # Build arrays for all interfaces
        rho_L = np.zeros(n + 1)
        u_L = np.zeros(n + 1)
        p_L = np.zeros(n + 1)
        rho_R = np.zeros(n + 1)
        u_R = np.zeros(n + 1)
        p_R = np.zeros(n + 1)

        # Left boundary (transmissive)
        rho_L[0], u_L[0], p_L[0] = rho[0], u[0], p[0]
        rho_R[0], u_R[0], p_R[0] = rho[0], u[0], p[0]

        # Interior interfaces
        rho_L[1:n] = rho[:-1]
        u_L[1:n] = u[:-1]
        p_L[1:n] = p[:-1]
        rho_R[1:n] = rho[1:]
        u_R[1:n] = u[1:]
        p_R[1:n] = p[1:]

        # Right boundary (transmissive)
        rho_L[n], u_L[n], p_L[n] = rho[-1], u[-1], p[-1]
        rho_R[n], u_R[n], p_R[n] = rho[-1], u[-1], p[-1]

        if use_mlp and model is not None:
            # Use MLP with per-interface gamma
            flux = mlp_flux_batch(model, stats, rho_L, u_L, p_L,
                                  rho_R, u_R, p_R, gamma_interface, device)
        else:
            # Analytical Roe flux
            for i in range(n + 1):
                flux[i] = roe_flux_analytical(gamma_interface[i],
                                              rho_L[i], u_L[i], p_L[i],
                                              rho_R[i], u_R[i], p_R[i])

        # Update conserved variables
        E = p / (gamma_cells - 1) + 0.5 * rho * u**2
        W = np.array([rho, rho * u, E])

        for i in range(n):
            W[:, i] = W[:, i] - dt / dx * (flux[i + 1] - flux[i])

        # Check validity
        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            print(f"  Invalid state at t={t:.6e}, step {n_steps}")
            break

        # Convert back to primitive (use local gamma)
        rho = W[0]
        u = W[1] / rho
        E = W[2]
        p = (gamma_cells - 1) * (E - 0.5 * rho * u**2)

        if np.any(p <= 0):
            print(f"  Negative pressure at t={t:.6e}, step {n_steps}")
            break

        U[0], U[1], U[2] = rho, u, p

        # Simple level set advection (interface moves with local velocity)
        # phi_t + u * phi_x = 0 (upwind)
        for i in range(n):
            if u[i] > 0:
                if i > 0:
                    phi[i] = phi[i] - dt / dx * u[i] * (phi[i] - phi[i-1])
            else:
                if i < n - 1:
                    phi[i] = phi[i] - dt / dx * u[i] * (phi[i+1] - phi[i])

        t += dt
        n_steps += 1

    print(f"  Completed: {n_steps} steps, final t = {t:.6e}")

    return {
        'x': x,
        'rho': U[0],
        'u': U[1],
        'p': U[2],
        'phi': phi,
        'rho_init': rho_init,
        'u_init': u_init,
        'p_init': p_init,
        'phi_init': phi_init,
        't': t,
        'n_steps': n_steps,
        'gamma_1': gamma_1,
        'gamma_2': gamma_2
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test multiphase simulation with MLP')
    parser.add_argument('--model', type=str, default='models/flux_mlp_gamma.pt',
                        help='Path to gamma-parameterized model')
    parser.add_argument('--config', type=str, default='in_1Dsod2fl',
                        help='Configuration name')
    parser.add_argument('--cfl', type=float, default=0.3,
                        help='CFL number')
    parser.add_argument('--output', type=str, default='multiphase_comparison.png',
                        help='Output plot path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("Multiphase Simulation Test: MLP vs Analytical Roe")
    print("=" * 70)

    # Load gamma-parameterized model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.model)
    if os.path.exists(model_path):
        model, stats, config = load_gamma_model(model_path, args.device)
        has_model = True
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Running analytical Roe only")
        model, stats = None, None
        has_model = False

    # Run analytical Roe simulation
    print("\n" + "-" * 70)
    roe_result = run_multiphase_simulation(
        config_name=args.config,
        use_mlp=False,
        cfl=args.cfl
    )

    # Run MLP simulation if model available
    if has_model:
        print("\n" + "-" * 70)
        mlp_result = run_multiphase_simulation(
            config_name=args.config,
            model=model,
            stats=stats,
            use_mlp=True,
            cfl=args.cfl,
            device=args.device
        )

        # Compute errors
        rho_err = np.abs(mlp_result['rho'] - roe_result['rho'])
        u_err = np.abs(mlp_result['u'] - roe_result['u'])
        p_err = np.abs(mlp_result['p'] - roe_result['p'])

        print("\n" + "=" * 70)
        print("ERROR ANALYSIS (MLP vs Roe)")
        print("=" * 70)
        print(f"Density MAE:  {np.mean(rho_err):.6e} kg/m³")
        print(f"Velocity MAE: {np.mean(u_err):.6e} m/s")
        print(f"Pressure MAE: {np.mean(p_err):.6e} Pa")
        print(f"Density Max:  {np.max(rho_err):.6e} kg/m³")
        print(f"Velocity Max: {np.max(u_err):.6e} m/s")
        print(f"Pressure Max: {np.max(p_err):.6e} Pa")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x = roe_result['x']

    # Top row: Final states
    axes[0, 0].plot(x, roe_result['rho'], 'b-', label='Roe', linewidth=2)
    if has_model:
        axes[0, 0].plot(x, mlp_result['rho'], 'r--', label='MLP', linewidth=1.5)
    axes[0, 0].plot(x, roe_result['rho_init'], 'k:', alpha=0.5, label='Initial')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('Density [kg/m³]')
    axes[0, 0].set_title('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, roe_result['u'], 'b-', label='Roe', linewidth=2)
    if has_model:
        axes[0, 1].plot(x, mlp_result['u'], 'r--', label='MLP', linewidth=1.5)
    axes[0, 1].set_xlabel('x [m]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x, roe_result['p'], 'b-', label='Roe', linewidth=2)
    if has_model:
        axes[0, 2].plot(x, mlp_result['p'], 'r--', label='MLP', linewidth=1.5)
    axes[0, 2].set_xlabel('x [m]')
    axes[0, 2].set_ylabel('Pressure [Pa]')
    axes[0, 2].set_title('Pressure')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom row: Level set and errors
    axes[1, 0].plot(x, roe_result['phi'], 'b-', label='Roe', linewidth=2)
    if has_model:
        axes[1, 0].plot(x, mlp_result['phi'], 'r--', label='MLP', linewidth=1.5)
    axes[1, 0].plot(x, roe_result['phi_init'], 'k:', alpha=0.5, label='Initial')
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('φ')
    axes[1, 0].set_title('Level Set (φ > 0: fluid 1, φ ≤ 0: fluid 2)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if has_model:
        axes[1, 1].semilogy(x, rho_err + 1e-16, 'b-', label='Density')
        axes[1, 1].semilogy(x, u_err + 1e-16, 'g-', label='Velocity')
        axes[1, 1].set_xlabel('x [m]')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Errors (Density, Velocity)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].semilogy(x, p_err + 1e-16, 'r-', label='Pressure')
        axes[1, 2].set_xlabel('x [m]')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].set_title('Error (Pressure)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No MLP model loaded', ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 2].text(0.5, 0.5, 'No MLP model loaded', ha='center', va='center',
                        transform=axes[1, 2].transAxes, fontsize=12)

    gamma_1 = roe_result['gamma_1']
    gamma_2 = roe_result['gamma_2']
    plt.suptitle(f'Two-Fluid Sod Shock Tube ({args.config})\n'
                 f'γ₁ = {gamma_1}, γ₂ = {gamma_2}, t = {roe_result["t"]:.2e} s',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    plt.show()


if __name__ == '__main__':
    main()
