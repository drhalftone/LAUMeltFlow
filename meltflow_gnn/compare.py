"""
Compare original MeltFlow solver with GNN-based solver side by side.

Runs both simulations and displays results in a synchronized visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meltflow.functions.parameters import create_parameters, apply_defaults
from meltflow.functions.grid import grid_setup
from meltflow.functions.state_var import state_var
from meltflow.functions.ghost_fluid import ghost_GFM, real_GFM, extrp_vel
from meltflow.functions.level_set import advc
from meltflow.functions.solver import run_slvr, timestep
from meltflow.input.configs import load_config

from .model import EulerGNN
from .graph import create_1d_graph
from .data_generator import compute_roe_flux_1d, prim_to_cons_1d


def load_trained_model(model_path: str = 'flux_model.pt'):
    """Load trained GNN model and normalizer."""
    checkpoint = torch.load(model_path, weights_only=False)

    # Check if antisymmetric flag is stored in checkpoint
    antisymmetric = checkpoint.get('antisymmetric', True)

    model = EulerGNN(
        n_var=3,
        n_edge_features=2,
        hidden_dim=256,
        n_layers=5,
        antisymmetric=antisymmetric
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    normalizer = checkpoint['normalizer']

    return model, normalizer


def gnn_flux_step(model, normalizer, x, U, phi, dt, gamma):
    """
    Perform one timestep using GNN-predicted flux.

    Parameters
    ----------
    model : EulerGNN
        Trained GNN model
    normalizer : dict
        Normalization parameters
    x : np.ndarray
        Grid coordinates
    U : np.ndarray
        Primitive variables [rho, u, p]
    phi : np.ndarray
        Level set
    dt : float
        Timestep
    gamma : float
        Specific heat ratio

    Returns
    -------
    np.ndarray
        Updated primitive variables
    """
    n = len(x)
    n_var = U.shape[0]

    # Create graph
    graph = create_1d_graph(x, U, phi)

    # Normalize
    node_mean = normalizer['node_mean']
    node_std = normalizer['node_std']
    flux_mean = normalizer['flux_mean']
    flux_std = normalizer['flux_std']

    graph.x = (graph.x - node_mean) / node_std

    # Predict flux
    with torch.no_grad():
        pred_flux_norm = model.compute_flux(graph)
        # Only positive direction edges (every other edge)
        mask = graph.edge_attr[:, 1] > 0
        pred_flux_norm = pred_flux_norm[mask]

        # Denormalize
        pred_flux = pred_flux_norm * flux_std + flux_mean
        pred_flux = pred_flux.numpy()

    # Convert to conserved variables
    W = np.zeros_like(U)
    for i in range(n):
        W[:, i] = prim_to_cons_1d(U[:, i], gamma)

    # Update using flux differencing
    dx = x[1] - x[0]
    W_new = W.copy()

    for i in range(1, n - 1):
        # F_{i+1/2} - F_{i-1/2}
        W_new[:, i] = W[:, i] - dt / dx * (pred_flux[i, :] - pred_flux[i-1, :])

    # Boundary conditions (copy)
    W_new[:, 0] = W_new[:, 1]
    W_new[:, -1] = W_new[:, -2]

    # Convert back to primitive
    U_new = np.zeros_like(U)
    for i in range(n):
        rho = W_new[0, i]
        u = W_new[1, i] / rho
        E = W_new[2, i]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)
        U_new[:, i] = [rho, u, p]

    return U_new


def run_comparison(config_name: str = 'in_1Dsod1fl', model_path: str = 'flux_model.pt'):
    """
    Run original solver and GNN solver side by side.

    Parameters
    ----------
    config_name : str
        Configuration name
    model_path : str
        Path to trained model
    """
    print("=" * 60)
    print("MeltFlow vs GNN Comparison")
    print("=" * 60)

    # Load model
    print("\nLoading trained GNN model...")
    try:
        model, normalizer = load_trained_model(model_path)
        print("   Model loaded successfully")
    except FileNotFoundError:
        print(f"   Error: Model file '{model_path}' not found.")
        print("   Please run training first: python -m meltflow_gnn.train")
        return

    # Load configuration
    print(f"\nLoading configuration: {config_name}")
    config, init_func = load_config(config_name)

    # Grid setup (for both)
    n_var, n, x, dx, U_orig, phi_orig = grid_setup(config)
    _, _, _, _, U_gnn, phi_gnn = grid_setup(config)

    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx
    config = apply_defaults(config, config['n_dim'])
    prm = create_parameters(config)
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Apply initial conditions
    U_orig, phi_orig = init_func(x, U_orig, phi_orig)
    U_gnn, phi_gnn = init_func(x, U_gnn, phi_gnn)

    gamma = prm.c_EoS[0]  # Use first fluid's gamma

    # Set up figure
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])
    ax_rho_err = fig.add_subplot(gs[1, 0])
    ax_u_err = fig.add_subplot(gs[1, 1])
    ax_p_err = fig.add_subplot(gs[1, 2])

    fig.suptitle('MeltFlow (blue) vs GNN (red dashed)', fontsize=14)

    # Time stepping
    t = 0.0
    it = 0
    dt_fixed = 0.1336323e-05  # Match MATLAB

    print(f"\nRunning simulation to t = {prm.t_f}")
    print("-" * 60)

    while t < prm.t_f:
        dt = dt_fixed
        if t + dt > prm.t_f:
            dt = prm.t_f - t

        t += dt

        # === Original MeltFlow solver ===
        W_orig = state_var(prm, "cons", n_var, phi_orig, U_orig)
        UU, WW = ghost_GFM(prm, x, phi_orig, U_orig)
        WW = run_slvr(prm, dt, x, phi_orig, UU, WW)
        V = extrp_vel(prm, x, phi_orig, WW)
        phi_orig = advc(prm, dt, V, phi_orig)
        W_orig = real_GFM(prm, phi_orig, WW)
        U_orig = state_var(prm, "prim", n_var, phi_orig, W_orig)

        # === GNN solver ===
        U_gnn = gnn_flux_step(model, normalizer, x, U_gnn, phi_gnn, dt, gamma)

        it += 1

        # Update plots every 50 iterations
        if it % 50 == 0 or t >= prm.t_f:
            # Clear axes
            for ax in [ax_rho, ax_u, ax_p, ax_rho_err, ax_u_err, ax_p_err]:
                ax.clear()

            # Plot comparisons
            ax_rho.plot(x, U_orig[0, :], 'b-', label='MeltFlow', linewidth=2)
            ax_rho.plot(x, U_gnn[0, :], 'r--', label='GNN', linewidth=2)
            ax_rho.set_ylabel(r'$\rho$ [kg/m³]')
            ax_rho.set_title('Density')
            ax_rho.legend()
            ax_rho.grid(True)

            ax_u.plot(x, U_orig[1, :], 'b-', linewidth=2)
            ax_u.plot(x, U_gnn[1, :], 'r--', linewidth=2)
            ax_u.set_ylabel(r'$u$ [m/s]')
            ax_u.set_title('Velocity')
            ax_u.grid(True)

            ax_p.plot(x, U_orig[2, :], 'b-', linewidth=2)
            ax_p.plot(x, U_gnn[2, :], 'r--', linewidth=2)
            ax_p.set_ylabel(r'$p$ [Pa]')
            ax_p.set_title('Pressure')
            ax_p.grid(True)

            # Plot errors
            err_rho = U_gnn[0, :] - U_orig[0, :]
            err_u = U_gnn[1, :] - U_orig[1, :]
            err_p = U_gnn[2, :] - U_orig[2, :]

            ax_rho_err.plot(x, err_rho, 'k-', linewidth=1.5)
            ax_rho_err.set_ylabel(r'$\Delta\rho$')
            ax_rho_err.set_xlabel('x [m]')
            ax_rho_err.set_title('Density Error')
            ax_rho_err.grid(True)

            ax_u_err.plot(x, err_u, 'k-', linewidth=1.5)
            ax_u_err.set_ylabel(r'$\Delta u$')
            ax_u_err.set_xlabel('x [m]')
            ax_u_err.set_title('Velocity Error')
            ax_u_err.grid(True)

            ax_p_err.plot(x, err_p, 'k-', linewidth=1.5)
            ax_p_err.set_ylabel(r'$\Delta p$')
            ax_p_err.set_xlabel('x [m]')
            ax_p_err.set_title('Pressure Error')
            ax_p_err.grid(True)

            fig.suptitle(f'MeltFlow vs GNN  |  t = {t:.6f} s  |  iter = {it}', fontsize=14)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Print progress
            rms_rho = np.sqrt(np.mean(err_rho**2))
            rms_u = np.sqrt(np.mean(err_u**2))
            rms_p = np.sqrt(np.mean(err_p**2))
            print(f"t={t:.6f}, it={it:4d}, RMS errors: rho={rms_rho:.4e}, u={rms_u:.4e}, p={rms_p:.4e}")

    print("-" * 60)
    print("Simulation complete!")

    # Final statistics
    err_rho = U_gnn[0, :] - U_orig[0, :]
    err_u = U_gnn[1, :] - U_orig[1, :]
    err_p = U_gnn[2, :] - U_orig[2, :]

    print(f"\nFinal RMS Errors:")
    print(f"   Density:  {np.sqrt(np.mean(err_rho**2)):.6e}")
    print(f"   Velocity: {np.sqrt(np.mean(err_u**2)):.6e}")
    print(f"   Pressure: {np.sqrt(np.mean(err_p**2)):.6e}")

    # Save results to CSV
    csv_path = 'data/gnn_flux_comparison.csv'
    os.makedirs('data', exist_ok=True)

    # Create data array with all results
    data_out = np.column_stack([
        x,
        U_orig[0, :], U_orig[1, :], U_orig[2, :],
        U_gnn[0, :], U_gnn[1, :], U_gnn[2, :],
        err_rho, err_u, err_p
    ])

    header = 'x,rho_meltflow,u_meltflow,p_meltflow,rho_gnn,u_gnn,p_gnn,err_rho,err_u,err_p'
    np.savetxt(csv_path, data_out, delimiter=',', header=header, comments='',
               fmt='%.10e')

    # Also save summary statistics
    summary_path = 'data/gnn_flux_comparison_summary.csv'
    with open(summary_path, 'w') as f:
        f.write('metric,density,velocity,pressure\n')
        f.write(f'rms_error,{np.sqrt(np.mean(err_rho**2)):.10e},{np.sqrt(np.mean(err_u**2)):.10e},{np.sqrt(np.mean(err_p**2)):.10e}\n')
        f.write(f'max_error,{np.max(np.abs(err_rho)):.10e},{np.max(np.abs(err_u)):.10e},{np.max(np.abs(err_p)):.10e}\n')
        f.write(f'mean_error,{np.mean(err_rho):.10e},{np.mean(err_u):.10e},{np.mean(err_p):.10e}\n')

    print(f"\nResults saved to:")
    print(f"   {csv_path}")
    print(f"   {summary_path}")

    plt.ioff()
    plt.show()

    return U_orig, U_gnn


if __name__ == '__main__':
    run_comparison()
