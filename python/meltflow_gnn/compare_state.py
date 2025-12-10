"""
Compare MeltFlow solver with state-prediction GNN side by side.
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
from meltflow.functions.solver import run_slvr
from meltflow.input.configs import load_config

from .model_state import FullStateGNN
from .graph import create_1d_graph


def load_state_model(model_path: str = 'state_model.pt'):
    """Load trained state prediction model."""
    checkpoint = torch.load(model_path, weights_only=False)

    model = FullStateGNN(
        n_var=3,
        n_node_features=5,
        n_edge_features=2,
        hidden_dim=256,
        n_layers=4,
        n_message_passing=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    normalizer = checkpoint['normalizer']
    return model, normalizer


def gnn_state_step(model, normalizer, x, U, phi):
    """
    Perform one timestep using state-prediction GNN.
    """
    # Create graph
    graph = create_1d_graph(x, U, phi)

    # Normalize
    graph.x = (graph.x - normalizer['node_mean']) / normalizer['node_std']

    # Predict delta
    with torch.no_grad():
        pred_delta_norm = model(graph)
        # Denormalize
        pred_delta = pred_delta_norm * normalizer['delta_std'] + normalizer['delta_mean']
        pred_delta = pred_delta.numpy()

    # Apply update: U^{n+1} = U^n + delta
    U_new = U.copy()
    U_new = U + pred_delta.T  # delta is (n, 3), U is (3, n)

    return U_new


def run_comparison(config_name: str = 'in_1Dsod1fl', model_path: str = 'state_model.pt'):
    """Run comparison between MeltFlow and state-prediction GNN."""
    print("=" * 60)
    print("MeltFlow vs State-Prediction GNN Comparison")
    print("=" * 60)

    # Load model
    print("\nLoading trained state model...")
    try:
        model, normalizer = load_state_model(model_path)
        print("   Model loaded successfully")
    except FileNotFoundError:
        print(f"   Error: Model file '{model_path}' not found.")
        print("   Please run training first: python -m meltflow_gnn.train_state")
        return

    # Load configuration
    print(f"\nLoading configuration: {config_name}")
    config, init_func = load_config(config_name)

    # Grid setup
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

    fig.suptitle('MeltFlow (blue) vs State-GNN (red dashed)', fontsize=14)

    # Time stepping
    t = 0.0
    it = 0
    dt_fixed = 0.1336323e-05

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

        # === State-prediction GNN solver ===
        U_gnn = gnn_state_step(model, normalizer, x, U_gnn, phi_gnn)

        it += 1

        # Update plots every 50 iterations
        if it % 50 == 0 or t >= prm.t_f:
            for ax in [ax_rho, ax_u, ax_p, ax_rho_err, ax_u_err, ax_p_err]:
                ax.clear()

            ax_rho.plot(x, U_orig[0, :], 'b-', label='MeltFlow', linewidth=2)
            ax_rho.plot(x, U_gnn[0, :], 'r--', label='State-GNN', linewidth=2)
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

            fig.suptitle(f'MeltFlow vs State-GNN  |  t = {t:.6f} s  |  iter = {it}', fontsize=14)

            fig.canvas.draw()
            fig.canvas.flush_events()

            rms_rho = np.sqrt(np.mean(err_rho**2))
            rms_u = np.sqrt(np.mean(err_u**2))
            rms_p = np.sqrt(np.mean(err_p**2))
            print(f"t={t:.6f}, it={it:4d}, RMS errors: rho={rms_rho:.4e}, u={rms_u:.4e}, p={rms_p:.4e}")

    print("-" * 60)
    print("Simulation complete!")

    err_rho = U_gnn[0, :] - U_orig[0, :]
    err_u = U_gnn[1, :] - U_orig[1, :]
    err_p = U_gnn[2, :] - U_orig[2, :]

    print(f"\nFinal RMS Errors:")
    print(f"   Density:  {np.sqrt(np.mean(err_rho**2)):.6e}")
    print(f"   Velocity: {np.sqrt(np.mean(err_u**2)):.6e}")
    print(f"   Pressure: {np.sqrt(np.mean(err_p**2)):.6e}")

    plt.ioff()
    plt.show()

    return U_orig, U_gnn


if __name__ == '__main__':
    run_comparison()
