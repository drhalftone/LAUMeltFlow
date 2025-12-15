"""
Compare MLP flux model vs analytical Roe solver for interpolated gamma values.

Tests generalization by running simulations at gamma values midway between
the training points.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


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


def load_model(model_path: str, device: str = 'cuda') -> Tuple[FluxMLP, Dict]:
    """Load trained PyTorch model."""
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
    print(f"  Architecture: {config['input_dim']} -> {config['hidden_dims']} -> {config['output_dim']}")

    return model, stats, config


def mlp_flux_batch(model: FluxMLP, stats: Dict,
                   rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                   rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray,
                   gamma: float, device: str = 'cuda') -> np.ndarray:
    """Compute flux for batch of states using MLP."""
    n = len(rho_L)

    # Build input array with gamma
    X = np.column_stack([rho_L, u_L, p_L, rho_R, u_R, p_R, np.full(n, gamma)])

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


def roe_flux_analytical(gamma: float, rho_L: float, u_L: float, p_L: float,
                        rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """Analytical Roe flux."""
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


def run_simulation(model: FluxMLP, stats: Dict, gamma: float,
                   nx: int = 100, t_final: float = 0.00075, cfl: float = 0.5,
                   use_mlp: bool = True, device: str = 'cuda') -> Dict:
    """Run 1D Euler simulation with Sod initial conditions."""
    dx = 1.0 / nx
    x = np.linspace(dx / 2, 1.0 - dx / 2, nx)

    # Initialize Sod shock tube
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(nx)
    p = np.where(x < 0.5, 100000.0, 10000.0)

    t = 0.0
    n_steps = 0

    while t < t_final:
        # CFL timestep
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t

        # Compute interface fluxes
        flux = np.zeros((nx + 1, 3))

        if use_mlp:
            # Batch compute all interface fluxes (nx+1 interfaces)
            # Interface i is between cell i-1 and cell i
            # Boundaries: interface 0 uses cell 0 for both sides, interface nx uses cell nx-1 for both
            rho_L = np.zeros(nx + 1)
            u_L = np.zeros(nx + 1)
            p_L = np.zeros(nx + 1)
            rho_R = np.zeros(nx + 1)
            u_R = np.zeros(nx + 1)
            p_R = np.zeros(nx + 1)

            # Left boundary (transmissive)
            rho_L[0], u_L[0], p_L[0] = rho[0], u[0], p[0]
            rho_R[0], u_R[0], p_R[0] = rho[0], u[0], p[0]

            # Interior interfaces
            rho_L[1:nx] = rho[:-1]
            u_L[1:nx] = u[:-1]
            p_L[1:nx] = p[:-1]
            rho_R[1:nx] = rho[1:]
            u_R[1:nx] = u[1:]
            p_R[1:nx] = p[1:]

            # Right boundary (transmissive)
            rho_L[nx], u_L[nx], p_L[nx] = rho[-1], u[-1], p[-1]
            rho_R[nx], u_R[nx], p_R[nx] = rho[-1], u[-1], p[-1]

            flux = mlp_flux_batch(model, stats, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, device)
        else:
            for i in range(nx + 1):
                if i == 0:
                    rho_L, u_L, p_L = rho[0], u[0], p[0]
                    rho_R, u_R, p_R = rho[0], u[0], p[0]
                elif i == nx:
                    rho_L, u_L, p_L = rho[-1], u[-1], p[-1]
                    rho_R, u_R, p_R = rho[-1], u[-1], p[-1]
                else:
                    rho_L, u_L, p_L = rho[i - 1], u[i - 1], p[i - 1]
                    rho_R, u_R, p_R = rho[i], u[i], p[i]

                flux[i] = roe_flux_analytical(gamma, rho_L, u_L, p_L, rho_R, u_R, p_R)

        # Convert to conserved and update
        E = p / (gamma - 1) + 0.5 * rho * u**2
        W = np.array([rho, rho * u, E])

        for i in range(nx):
            W[:, i] = W[:, i] - dt / dx * (flux[i + 1] - flux[i])

        # Check validity
        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            print(f"  Invalid state at t={t:.6f}")
            break

        # Convert back to primitive
        rho = W[0]
        u = W[1] / rho
        E = W[2]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        if np.any(p <= 0):
            print(f"  Negative pressure at t={t:.6f}")
            break

        t += dt
        n_steps += 1

    return {'x': x, 'rho': rho, 'u': u, 'p': p, 't': t, 'n_steps': n_steps, 'gamma': gamma}


def compute_errors(mlp_result: Dict, roe_result: Dict) -> Dict:
    """Compute error metrics between MLP and Roe results."""
    rho_err = mlp_result['rho'] - roe_result['rho']
    u_err = mlp_result['u'] - roe_result['u']
    p_err = mlp_result['p'] - roe_result['p']

    return {
        'rho_mae': np.mean(np.abs(rho_err)),
        'rho_max': np.max(np.abs(rho_err)),
        'rho_rms': np.sqrt(np.mean(rho_err**2)),
        'u_mae': np.mean(np.abs(u_err)),
        'u_max': np.max(np.abs(u_err)),
        'u_rms': np.sqrt(np.mean(u_err**2)),
        'p_mae': np.mean(np.abs(p_err)),
        'p_max': np.max(np.abs(p_err)),
        'p_rms': np.sqrt(np.mean(p_err**2)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare MLP vs Roe for interpolated gamma')
    parser.add_argument('--model', type=str, default='models/flux_mlp_gamma.pt')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--t-final', type=float, default=0.00075)
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='gamma_comparison.png')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")

    print("=" * 70)
    print("MLP vs Roe Comparison for Interpolated Gamma Values")
    print("=" * 70)

    # Training gamma values
    train_gammas = [1.2, 1.256, 1.311, 1.367, 1.422, 1.478, 1.533, 1.589, 1.644, 1.7]

    # Midpoint (interpolated) gamma values
    test_gammas = [(train_gammas[i] + train_gammas[i + 1]) / 2 for i in range(len(train_gammas) - 1)]

    print(f"\nTraining gamma values: {[f'{g:.3f}' for g in train_gammas]}")
    print(f"Testing gamma values:  {[f'{g:.3f}' for g in test_gammas]}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model, stats, config = load_model(args.model, args.device)

    # Run simulations for each test gamma
    results = []
    all_errors = []

    print(f"\nRunning simulations (nx={args.nx}, t_final={args.t_final})...")
    print("-" * 70)

    for gamma in test_gammas:
        print(f"\nGamma = {gamma:.4f}:")

        # MLP simulation
        mlp_result = run_simulation(model, stats, gamma, args.nx, args.t_final, args.cfl,
                                    use_mlp=True, device=args.device)
        print(f"  MLP: {mlp_result['n_steps']} steps")

        # Roe simulation
        roe_result = run_simulation(model, stats, gamma, args.nx, args.t_final, args.cfl,
                                    use_mlp=False, device=args.device)
        print(f"  Roe: {roe_result['n_steps']} steps")

        # Compute errors
        errors = compute_errors(mlp_result, roe_result)
        all_errors.append(errors)

        print(f"  Errors - rho: {errors['rho_mae']:.2e} (MAE), "
              f"u: {errors['u_mae']:.2e}, p: {errors['p_mae']:.2e}")

        results.append({
            'gamma': gamma,
            'mlp': mlp_result,
            'roe': roe_result,
            'errors': errors
        })

    # Create comparison plot
    print(f"\nCreating comparison plot...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Select 3 representative gammas: low, mid, high
    plot_indices = [0, 4, 8]  # First, middle, last
    var_names = ['rho', 'u', 'p']
    var_labels = [r'Density $\rho$ [kg/m³]', r'Velocity $u$ [m/s]', r'Pressure $p$ [Pa]']

    for col, idx in enumerate(plot_indices):
        r = results[idx]
        x = r['mlp']['x']
        gamma = r['gamma']

        for row, (var, label) in enumerate(zip(var_names, var_labels)):
            ax = axes[row, col]
            ax.plot(x, r['roe'][var], 'b-', label='Roe', linewidth=2)
            ax.plot(x, r['mlp'][var], 'r--', label='MLP', linewidth=1.5)

            ax.set_xlabel('x [m]')
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(f'$\\gamma$ = {gamma:.3f}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()

    plt.suptitle(f'Sod Shock Tube: MLP vs Roe Solver (Interpolated $\\gamma$ values)\n'
                 f't = {args.t_final} s, {args.nx} cells', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {args.output}")

    # Create error summary plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    gammas = [r['gamma'] for r in results]
    rho_mae = [r['errors']['rho_mae'] for r in results]
    u_mae = [r['errors']['u_mae'] for r in results]
    p_mae = [r['errors']['p_mae'] for r in results]

    axes2[0].bar(range(len(gammas)), rho_mae, color='steelblue')
    axes2[0].set_xticks(range(len(gammas)))
    axes2[0].set_xticklabels([f'{g:.2f}' for g in gammas], rotation=45)
    axes2[0].set_xlabel('$\\gamma$')
    axes2[0].set_ylabel('MAE')
    axes2[0].set_title('Density Error')
    axes2[0].grid(True, alpha=0.3, axis='y')

    axes2[1].bar(range(len(gammas)), u_mae, color='forestgreen')
    axes2[1].set_xticks(range(len(gammas)))
    axes2[1].set_xticklabels([f'{g:.2f}' for g in gammas], rotation=45)
    axes2[1].set_xlabel('$\\gamma$')
    axes2[1].set_ylabel('MAE')
    axes2[1].set_title('Velocity Error')
    axes2[1].grid(True, alpha=0.3, axis='y')

    axes2[2].bar(range(len(gammas)), p_mae, color='coral')
    axes2[2].set_xticks(range(len(gammas)))
    axes2[2].set_xticklabels([f'{g:.2f}' for g in gammas], rotation=45)
    axes2[2].set_xlabel('$\\gamma$')
    axes2[2].set_ylabel('MAE')
    axes2[2].set_title('Pressure Error')
    axes2[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('MLP vs Roe Error by Interpolated $\\gamma$ Value', fontsize=14)
    plt.tight_layout()

    error_plot_path = args.output.replace('.png', '_errors.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved error plot to {error_plot_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Absolute Errors (MLP vs Roe)")
    print("=" * 70)
    print(f"{'Gamma':>8} | {'Density MAE':>12} | {'Velocity MAE':>12} | {'Pressure MAE':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['gamma']:>8.4f} | {r['errors']['rho_mae']:>12.4e} | "
              f"{r['errors']['u_mae']:>12.4e} | {r['errors']['p_mae']:>12.4e}")

    print("-" * 70)

    # Average errors
    avg_rho = np.mean([r['errors']['rho_mae'] for r in results])
    avg_u = np.mean([r['errors']['u_mae'] for r in results])
    avg_p = np.mean([r['errors']['p_mae'] for r in results])
    print(f"{'Average':>8} | {avg_rho:>12.4e} | {avg_u:>12.4e} | {avg_p:>12.4e}")

    print("\nDone!")

    plt.show()


if __name__ == '__main__':
    main()
