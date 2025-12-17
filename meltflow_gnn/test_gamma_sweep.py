"""
Test MLP across different gamma values.

Runs Sod shock tube with various gamma values to verify
the gamma-parameterized MLP generalizes correctly.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: We implement the simulation directly without meltflow dependencies


class FluxMLP(nn.Module):
    """MLP for flux prediction."""

    def __init__(self, input_dim=7, output_dim=3, hidden_dims=[256]*5, activation='gelu'):
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
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_model(path: str, device: str = 'cuda'):
    """Load trained model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    stats = checkpoint['stats']

    model = FluxMLP(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, stats, config


def mlp_flux(U_L: np.ndarray, U_R: np.ndarray, gamma: float,
             model: nn.Module, stats: Dict, device: str = 'cuda') -> np.ndarray:
    """Compute flux using MLP."""
    rho_L, u_L, p_L = U_L
    rho_R, u_R, p_R = U_R

    # Build input with gamma
    X = np.array([[rho_L, u_L, p_L, rho_R, u_R, p_R, gamma]], dtype=np.float32)

    # Normalize
    X_norm = (X - stats['X_mean']) / stats['X_std']

    # Predict
    with torch.no_grad():
        X_t = torch.from_numpy(X_norm).to(device)
        Y_pred = model(X_t).cpu().numpy()

    # Denormalize
    flux = Y_pred[0] * stats['Y_std'] + stats['Y_mean']

    return flux


def run_sod_simulation(gamma: float, model: nn.Module, stats: Dict,
                       device: str, use_mlp: bool = True,
                       n_cells: int = 101, t_final: float = 0.00025) -> Tuple[np.ndarray, np.ndarray]:
    """Run Sod shock tube with specified gamma."""

    # Grid setup
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells
    x = np.linspace(x_min + dx/2, x_max - dx/2, n_cells)

    # Initial conditions (Sod shock tube)
    U = np.zeros((3, n_cells))
    rho_L, u_L, p_L = 1.0, 0.0, 1e5
    rho_R, u_R, p_R = 0.125, 0.0, 1e4

    for i in range(n_cells):
        if x[i] < 0.5:
            U[0, i] = rho_L
            U[1, i] = u_L
            U[2, i] = p_L
        else:
            U[0, i] = rho_R
            U[1, i] = u_R
            U[2, i] = p_R

    # Convert to conservative
    U_cons = np.zeros_like(U)
    for i in range(n_cells):
        rho, u, p = U[0, i], U[1, i], U[2, i]
        U_cons[0, i] = rho
        U_cons[1, i] = rho * u
        U_cons[2, i] = p / (gamma - 1) + 0.5 * rho * u**2

    # Time stepping
    t = 0.0
    cfl = 0.5

    while t < t_final:
        # Compute primitive variables
        rho = U_cons[0, :]
        u = U_cons[1, :] / rho
        E = U_cons[2, :]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        # CFL condition
        c = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + c)
        if t + dt > t_final:
            dt = t_final - t

        # Compute fluxes at interfaces
        F = np.zeros((3, n_cells + 1))

        for i in range(n_cells + 1):
            if i == 0:
                # Left boundary (transmissive)
                U_L = np.array([rho[0], u[0], p[0]])
                U_R = np.array([rho[0], u[0], p[0]])
            elif i == n_cells:
                # Right boundary (transmissive)
                U_L = np.array([rho[-1], u[-1], p[-1]])
                U_R = np.array([rho[-1], u[-1], p[-1]])
            else:
                U_L = np.array([rho[i-1], u[i-1], p[i-1]])
                U_R = np.array([rho[i], u[i], p[i]])

            if use_mlp:
                F[:, i] = mlp_flux(U_L, U_R, gamma, model, stats, device)
            else:
                # Analytical Roe flux
                F[:, i] = analytical_roe_flux(U_L, U_R, gamma)

        # Update
        for i in range(n_cells):
            U_cons[:, i] = U_cons[:, i] - dt / dx * (F[:, i+1] - F[:, i])

        t += dt

    # Convert back to primitive
    rho = U_cons[0, :]
    u = U_cons[1, :] / rho
    E = U_cons[2, :]
    p = (gamma - 1) * (E - 0.5 * rho * u**2)

    U_final = np.array([rho, u, p])

    return x, U_final


def analytical_roe_flux(U_L: np.ndarray, U_R: np.ndarray, gamma: float) -> np.ndarray:
    """Compute Roe flux analytically."""
    rho_L, u_L, p_L = U_L
    rho_R, u_R, p_R = U_R

    # Specific enthalpies
    H_L = gamma * p_L / ((gamma - 1) * rho_L) + 0.5 * u_L**2
    H_R = gamma * p_R / ((gamma - 1) * rho_R) + 0.5 * u_R**2

    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe**2))

    # Eigenvalues
    lambda1 = u_roe - c_roe
    lambda2 = u_roe
    lambda3 = u_roe + c_roe

    # Wave strengths
    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    alpha2 = drho - dp / c_roe**2
    alpha1 = 0.5 * (dp / (c_roe**2) - du * rho_L / c_roe + drho - alpha2)
    alpha3 = drho - alpha1 - alpha2

    # Actually use standard Roe formulation
    alpha1 = 0.5 * (dp - rho_L * c_roe * du) / c_roe**2
    alpha2 = drho - dp / c_roe**2
    alpha3 = 0.5 * (dp + rho_L * c_roe * du) / c_roe**2

    # Hmm, let me use a simpler approach - average flux with correction
    # Left flux
    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (E_L + p_L)
    ])

    # Right flux
    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (E_R + p_R)
    ])

    # Roe average
    rho_roe = sqrt_rho_L * sqrt_rho_R

    # Right eigenvectors
    R = np.array([
        [1, 1, 1],
        [u_roe - c_roe, u_roe, u_roe + c_roe],
        [H_roe - u_roe * c_roe, 0.5 * u_roe**2, H_roe + u_roe * c_roe]
    ])

    # Wave strengths (correct formulation)
    alpha = np.array([
        0.5 * (dp - rho_roe * c_roe * du) / c_roe**2,
        drho - dp / c_roe**2,
        0.5 * (dp + rho_roe * c_roe * du) / c_roe**2
    ])

    # Absolute eigenvalues
    lambda_abs = np.array([abs(lambda1), abs(lambda2), abs(lambda3)])

    # Roe flux
    F_roe = 0.5 * (F_L + F_R)
    for k in range(3):
        F_roe -= 0.5 * alpha[k] * lambda_abs[k] * R[:, k]

    return F_roe


def main():
    parser = argparse.ArgumentParser(description='Test MLP across gamma values')
    parser.add_argument('--model', type=str, default='models/flux_mlp_multiphase.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='gamma_sweep_results.png')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    print("=" * 70)
    print("Gamma Sweep Test: MLP vs Analytical Roe")
    print("=" * 70)

    # Load model
    model, stats, config = load_model(args.model, args.device)
    print(f"Loaded model from {args.model}")

    # Gamma values to test
    gamma_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.67]

    results = []

    for gamma in gamma_values:
        print(f"\nTesting gamma = {gamma}")

        # Run with analytical Roe
        x, U_roe = run_sod_simulation(gamma, model, stats, args.device, use_mlp=False)

        # Run with MLP
        _, U_mlp = run_sod_simulation(gamma, model, stats, args.device, use_mlp=True)

        # Compute errors
        rho_err = np.abs(U_mlp[0] - U_roe[0])
        u_err = np.abs(U_mlp[1] - U_roe[1])
        p_err = np.abs(U_mlp[2] - U_roe[2])

        results.append({
            'gamma': gamma,
            'x': x,
            'U_roe': U_roe,
            'U_mlp': U_mlp,
            'rho_mae': np.mean(rho_err),
            'u_mae': np.mean(u_err),
            'p_mae': np.mean(p_err),
            'rho_max': np.max(rho_err),
            'u_max': np.max(u_err),
            'p_max': np.max(p_err),
        })

        print(f"  Density MAE: {results[-1]['rho_mae']:.6e}")
        print(f"  Velocity MAE: {results[-1]['u_mae']:.6e}")
        print(f"  Pressure MAE: {results[-1]['p_mae']:.6e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Gamma':>8} | {'Density MAE':>12} | {'Velocity MAE':>12} | {'Pressure MAE':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['gamma']:>8.3f} | {r['rho_mae']:>12.6e} | {r['u_mae']:>12.6e} | {r['p_mae']:>12.6e}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, r in enumerate(results):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.plot(r['x'], r['U_roe'][0], 'b-', label='Roe', linewidth=2)
        ax.plot(r['x'], r['U_mlp'][0], 'r--', label='MLP', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(f'γ = {r["gamma"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sod Shock Tube: MLP vs Roe (Density) for Various γ', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    # Also plot error summary
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    gammas = [r['gamma'] for r in results]
    rho_maes = [r['rho_mae'] for r in results]
    u_maes = [r['u_mae'] for r in results]
    p_maes = [r['p_mae'] for r in results]

    x_pos = np.arange(len(gammas))
    width = 0.25

    ax2.bar(x_pos - width, rho_maes, width, label='Density MAE')
    ax2.bar(x_pos, u_maes, width, label='Velocity MAE')
    ax2.bar(x_pos + width, p_maes, width, label='Pressure MAE')

    ax2.set_xlabel('Gamma')
    ax2.set_ylabel('MAE')
    ax2.set_title('MLP Error vs Gamma')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    error_plot = args.output.replace('.png', '_errors.png')
    plt.tight_layout()
    plt.savefig(error_plot, dpi=150)
    print(f"Saved error plot to {error_plot}")


if __name__ == '__main__':
    main()
