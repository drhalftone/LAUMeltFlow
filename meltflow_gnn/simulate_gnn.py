"""
Run Sod shock tube simulation using SimpleEulerGNN with trained MLP weights.

Compares GNN flux predictions against analytical Roe solver for multiple gamma values.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from meltflow_gnn.model_simple import SimpleEulerGNN, load_uniform_weights, create_1d_graph


def roe_flux_analytical(gamma: float, rho_L: float, u_L: float, p_L: float,
                        rho_R: float, u_R: float, p_R: float) -> np.ndarray:
    """Analytical Roe flux for comparison."""
    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2

    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    h_L = (E_L + p_L) / rho_L
    h_R = (E_R + p_R) / rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) / denom
    a_roe = np.sqrt((gamma - 1) * (h_roe - 0.5 * u_roe**2))
    rho_roe = sqrt_rho_L * sqrt_rho_R

    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    dv = np.array([
        drho - dp / a_roe**2,
        du + dp / (rho_roe * a_roe),
        du - dp / (rho_roe * a_roe)
    ])

    lmda = np.array([u_roe, u_roe + a_roe, u_roe - a_roe])

    r = np.array([
        [1, u_roe, 0.5 * u_roe**2],
        rho_roe / (2 * a_roe) * np.array([1, u_roe + a_roe, h_roe + a_roe * u_roe]),
        -rho_roe / (2 * a_roe) * np.array([1, u_roe - a_roe, h_roe - a_roe * u_roe])
    ])

    F_L = np.array([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])
    F_R = np.array([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    flux_sum = sum(dv[j] * abs(lmda[j]) * r[j] for j in range(3))
    return 0.5 * (F_L + F_R) - 0.5 * flux_sum


def run_simulation_gnn(
    model: SimpleEulerGNN,
    stats: Dict,
    gamma: float,
    nx: int = 100,
    t_final: float = 0.00075,
    cfl: float = 0.5,
    device: str = 'cpu'
) -> Dict:
    """Run 1D Euler simulation using GNN for flux computation."""

    model = model.to(device)
    model.eval()

    dx = 1.0 / nx
    x = np.linspace(dx / 2, 1.0 - dx / 2, nx)

    # Initialize Sod shock tube
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(nx)
    p = np.where(x < 0.5, 100000.0, 10000.0)

    t = 0.0
    n_steps = 0
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=device)

    while t < t_final:
        # CFL timestep
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t

        # Build interface states (nx+1 interfaces)
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

        # Convert to tensors
        x_left = torch.tensor(
            np.column_stack([rho_L, u_L, p_L]),
            dtype=torch.float32, device=device
        )
        x_right = torch.tensor(
            np.column_stack([rho_R, u_R, p_R]),
            dtype=torch.float32, device=device
        )

        # Normalize inputs
        X = torch.cat([x_left, x_right, gamma_tensor.expand(nx + 1, 1)], dim=-1)
        X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32, device=device)
        X_std = torch.tensor(stats['X_std'], dtype=torch.float32, device=device)
        X_norm = (X - X_mean) / X_std

        # Compute flux using GNN's flux network directly
        with torch.no_grad():
            flux_norm = model.flux_layer.flux_net.mlp(X_norm)

        # Denormalize outputs
        Y_mean = torch.tensor(stats['Y_mean'], dtype=torch.float32, device=device)
        Y_std = torch.tensor(stats['Y_std'], dtype=torch.float32, device=device)
        flux = (flux_norm * Y_std + Y_mean).cpu().numpy()

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


def run_simulation_roe(gamma: float, nx: int = 100, t_final: float = 0.00075, cfl: float = 0.5) -> Dict:
    """Run 1D Euler simulation using analytical Roe flux."""

    dx = 1.0 / nx
    x = np.linspace(dx / 2, 1.0 - dx / 2, nx)

    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(nx)
    p = np.where(x < 0.5, 100000.0, 10000.0)

    t = 0.0
    n_steps = 0

    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t

        flux = np.zeros((nx + 1, 3))
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

        E = p / (gamma - 1) + 0.5 * rho * u**2
        W = np.array([rho, rho * u, E])

        for i in range(nx):
            W[:, i] = W[:, i] - dt / dx * (flux[i + 1] - flux[i])

        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            break

        rho = W[0]
        u = W[1] / rho
        E = W[2]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        if np.any(p <= 0):
            break

        t += dt
        n_steps += 1

    return {'x': x, 'rho': rho, 'u': u, 'p': p, 't': t, 'n_steps': n_steps, 'gamma': gamma}


def compute_errors(gnn_result: Dict, roe_result: Dict) -> Dict:
    """Compute error metrics."""
    return {
        'rho_mae': np.mean(np.abs(gnn_result['rho'] - roe_result['rho'])),
        'u_mae': np.mean(np.abs(gnn_result['u'] - roe_result['u'])),
        'p_mae': np.mean(np.abs(gnn_result['p'] - roe_result['p'])),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GNN simulation comparison')
    parser.add_argument('--model', type=str, default='models/flux_mlp_gamma.pt')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--t-final', type=float, default=0.00075)
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='gnn_gamma_comparison.png')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    print("=" * 70)
    print("SimpleEulerGNN vs Roe Solver Comparison")
    print("=" * 70)

    # Training gamma values
    train_gammas = [1.2, 1.256, 1.311, 1.367, 1.422, 1.478, 1.533, 1.589, 1.644, 1.7]
    test_gammas = [(train_gammas[i] + train_gammas[i + 1]) / 2 for i in range(len(train_gammas) - 1)]

    print(f"\nTesting gamma values: {[f'{g:.3f}' for g in test_gammas]}")

    # Load model
    print(f"\nLoading GNN model from {args.model}...")
    model = SimpleEulerGNN(hidden_dim=256, n_layers=5, input_dim=7)
    stats = load_uniform_weights(model, args.model)

    # Run simulations
    results = []
    print(f"\nRunning simulations (nx={args.nx}, t_final={args.t_final})...")
    print("-" * 70)

    for gamma in test_gammas:
        print(f"\nGamma = {gamma:.4f}:")

        gnn_result = run_simulation_gnn(model, stats, gamma, args.nx, args.t_final, args.cfl, args.device)
        print(f"  GNN: {gnn_result['n_steps']} steps")

        roe_result = run_simulation_roe(gamma, args.nx, args.t_final, args.cfl)
        print(f"  Roe: {roe_result['n_steps']} steps")

        errors = compute_errors(gnn_result, roe_result)
        print(f"  Errors - rho: {errors['rho_mae']:.2e}, u: {errors['u_mae']:.2e}, p: {errors['p_mae']:.2e}")

        results.append({'gamma': gamma, 'gnn': gnn_result, 'roe': roe_result, 'errors': errors})

    # Create plot
    print(f"\nCreating comparison plot...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plot_indices = [0, 4, 8]
    var_names = ['rho', 'u', 'p']
    var_labels = [r'Density $\rho$ [kg/m³]', r'Velocity $u$ [m/s]', r'Pressure $p$ [Pa]']

    for col, idx in enumerate(plot_indices):
        r = results[idx]
        x = r['gnn']['x']
        gamma = r['gamma']

        for row, (var, label) in enumerate(zip(var_names, var_labels)):
            ax = axes[row, col]
            ax.plot(x, r['roe'][var], 'b-', label='Roe', linewidth=2)
            ax.plot(x, r['gnn'][var], 'r--', label='GNN', linewidth=1.5)

            ax.set_xlabel('x [m]')
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(f'$\\gamma$ = {gamma:.3f}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()

    plt.suptitle(f'Sod Shock Tube: SimpleEulerGNN vs Roe Solver (Interpolated $\\gamma$ values)\n'
                 f't = {args.t_final} s, {args.nx} cells', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {args.output}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Absolute Errors (GNN vs Roe)")
    print("=" * 70)
    print(f"{'Gamma':>8} | {'Density MAE':>12} | {'Velocity MAE':>12} | {'Pressure MAE':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['gamma']:>8.4f} | {r['errors']['rho_mae']:>12.4e} | "
              f"{r['errors']['u_mae']:>12.4e} | {r['errors']['p_mae']:>12.4e}")

    print("-" * 70)
    avg_rho = np.mean([r['errors']['rho_mae'] for r in results])
    avg_u = np.mean([r['errors']['u_mae'] for r in results])
    avg_p = np.mean([r['errors']['p_mae'] for r in results])
    print(f"{'Average':>8} | {avg_rho:>12.4e} | {avg_u:>12.4e} | {avg_p:>12.4e}")

    print("\nDone!")
    plt.show()


if __name__ == '__main__':
    main()
