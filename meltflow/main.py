"""
MeltFlow - Ghost Fluid Method Solver

Main entry point for running simulations.

Usage:
    python -m meltflow.main --config in_1Dsod1fl
    python -m meltflow.main --config in_2Dcdrop
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .functions.parameters import create_parameters, Parameters
from .functions.grid import grid_setup
from .functions.state_var import state_var
from .functions.ghost_fluid import ghost_GFM, real_GFM, extrp_vel
from .functions.level_set import advc, reinit_fast
from .functions.solver import run_slvr, timestep
from .functions.io import interpolate, wrt_data
from .functions.plotting import plt_setup, plot
from .input.configs import load_config, CONFIGS


def print_term(prm: Parameters, flag: int, t: float = 0.0, t_wall: float = 0.0,
               it: int = 0, it_r: int = 0) -> None:
    """
    Print relevant data to terminal.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    flag : int
        Which header/line to print (1-9)
    t : float
        Current simulation time
    t_wall : float
        Wall clock time
    it : int
        Current iteration
    it_r : int
        Reinitialization iteration
    """
    n_dim = prm.n_dim
    n = prm.n
    dx = prm.dx
    x_min = prm.x_min
    x_max = prm.x_max
    t_f = prm.t_f
    cfl = prm.cfl
    n_r = prm.n_r
    e_r = prm.e_r
    ICs_hdr = prm.ICs_hdr

    # Headers
    headers = {
        1: "%======================= Ghost-fluid Method Solver =======================%",
        2: ICs_hdr,
        3: "%----- Final Time ---------------- CFL --------------- Grid Size --------%",
        4: "%------------------ dx -------------------- Grid Points -----------------%",
        5: "%------------- # Iter/Reinit ------------ Reinit Tolerance --------------%",
        6: "%--- Iteration --- # Reinit ----- Simulation Time ----- Wall Time [s] ---%",
        9: "%============================= End of Output ============================%",
    }

    skip_prev = [1, 0, 0, 0, 0, 0, 0, 0, 1]
    skip_next = [0, 1, 0, 0, 0, 0, 0, 1, 1]
    print_hdr = [1, 1, 1, 1, 1, 1, 0, 0, 1]
    print_ln = [0, 0, 1, 1, 1, 0, 1, 0, 0]

    idx = flag - 1  # Convert to 0-indexed

    if skip_prev[idx]:
        print()

    if print_hdr[idx] and flag in headers:
        print(headers[flag])

    if print_ln[idx]:
        if n_dim == 1:
            if flag == 3:
                grid_size = x_max - x_min
                print(f"{t_f:16f} {cfl:24f} {grid_size:22f}")
            elif flag == 4:
                print(f"{dx:26f} {n:25d}")
            elif flag == 5:
                print(f"{n_r:24d} {e_r:31e}")
            elif flag == 7:
                print(f"{it:12d} {it_r:13d} {t:19f} {t_wall:20f}")
        elif n_dim == 2:
            n = np.atleast_1d(n)
            dx = np.atleast_1d(dx)
            x_min = np.atleast_1d(x_min)
            x_max = np.atleast_1d(x_max)
            if flag == 3:
                print(f"{t_f:17f} {cfl:23f} {x_max[0]-x_min[0]:18f} x {x_max[1]-x_min[1]}")
            elif flag == 4:
                print(f"{dx[0]:17f} {dx[1]:23f} {n[0]:19d} x {n[1]}")
            elif flag == 5:
                print(f"{n_r:24d} {e_r:31e}")
            elif flag == 7:
                print(f"{it:12d} {it_r:13d} {t:19f} {t_wall:20f}")

    if skip_next[idx]:
        print()


def run_simulation(config_name: str = 'in_1Dsod1fl', show_plot: bool = True) -> dict:
    """
    Run a MeltFlow simulation.

    Parameters
    ----------
    config_name : str
        Name of the configuration to run
    show_plot : bool
        Whether to show plots

    Returns
    -------
    dict
        Results dictionary containing X, U, phi, and prm
    """
    # Start timer
    start_time = time.time()

    # Load configuration
    config, init_func = load_config(config_name)

    # Grid setup
    n_var, n, x, dx, U, phi = grid_setup(config)
    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx

    # Create parameters
    from .functions.parameters import apply_defaults
    config = apply_defaults(config, config['n_dim'])
    prm = create_parameters(config)
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Create mesh
    n_dim = prm.n_dim
    if n_dim == 1:
        X = x
    elif n_dim == 2:
        n = np.atleast_1d(n)
        xx = x[0, :n[0]]
        yy = x[1, :n[1]]
        XX, YY = np.meshgrid(xx, yy, indexing='ij')
        X = np.zeros((2, n[0], n[1]))
        X[0, :, :] = XX
        X[1, :, :] = YY

    # Apply initial conditions
    U, phi = init_func(x if n_dim == 1 else X, U, phi)

    # Initialize counters
    t = 0.0
    it = 0
    it_r = 0
    cntr_it = 0
    cntr_r = 0
    cntr_a = 0

    # Print headers
    print_term(prm, 1)
    print_term(prm, 2)
    print_term(prm, 3, t)
    print_term(prm, 4)

    # Set up plotting
    fig = None
    if prm.flg_plt and show_plot:
        plt.ion()
        fig = plt_setup(prm)
        if prm.flg_intrp:
            X_out, U_out, phi_out = interpolate(prm, X, U, phi)
        else:
            X_out, U_out, phi_out = X, U, phi

    # Calculate initial conserved variables
    W = state_var(prm, "cons", n_var, phi, U)

    # Main time loop
    if prm.n_r > 0:
        print_term(prm, 5)
    print_term(prm, 6)

    while t < prm.t_f:
        # Compute speed of sound
        a = state_var(prm, "SoS", 1, phi, U)

        # Calculate time step
        dt = timestep(prm, U, a)
        if t + dt > prm.t_f:
            dt = prm.t_f - t

        # Override dt for debugging (matching MATLAB)
        # dt = 0.1336323e-05

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
        it_r = 0
        if cntr_r >= prm.n_r and prm.n_r > 0:
            cntr_r = 0
            phi = reinit_fast(prm, phi)

        # Reassemble real domain
        W = real_GFM(prm, phi, WW)

        # Compute primitive variables
        U = state_var(prm, "prim", n_var, phi, W)

        # Update iteration counters
        it += 1
        cntr_it += 1

        # Print progress
        if it == 1 or cntr_it >= prm.n_disp or t >= prm.t_f:
            cntr_it = 0
            t_wall = time.time() - start_time
            print_term(prm, 7, t, t_wall, it, it_r)

        # Animation
        cntr_a += 1
        if prm.flg_anmt and fig is not None and (it == 1 or cntr_a >= prm.n_anmt):
            cntr_a = 0
            if prm.flg_intrp:
                X_out, U_out, phi_out = interpolate(prm, X, U, phi)
            else:
                X_out, U_out, phi_out = X, U, phi
            plot(fig, prm, X_out, U_out, phi_out)
            plt.pause(prm.t_anmt)

    print_term(prm, 8)

    # Post-processing
    # Interpolate
    if prm.flg_intrp:
        if n_dim == 1:
            print(f'Interpolating flow field ({prm.n_out} points)...')
        elif n_dim == 2:
            n_out = np.atleast_1d(prm.n_out)
            print(f'Interpolating flow field ({n_out[0]} x {n_out[1]} points)...')
        X_out, U_out, phi_out = interpolate(prm, X, U, phi)
    else:
        X_out, U_out, phi_out = X, U, phi

    # Write results
    if prm.flg_wrt:
        wrt_fl = f"{prm.wrt_prfx}{prm.wrt_nm}{prm.wrt_sfx}"
        wrt_data(prm, X_out, U_out, phi_out, wrt_fl)

    # Final plot
    if prm.flg_plt and fig is not None:
        print('Plotting flow field...')
        plot(fig, prm, X_out, U_out, phi_out)

    print_term(prm, 9)

    if show_plot:
        plt.ioff()
        plt.show()

    return {
        'X': X_out,
        'U': U_out,
        'phi': phi_out,
        'prm': prm
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='MeltFlow - Ghost Fluid Method Solver')
    parser.add_argument('--config', '-c', type=str, default='in_1Dsod1fl',
                       choices=list(CONFIGS.keys()),
                       help='Configuration name to run')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')

    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:")
        for name in CONFIGS.keys():
            print(f"  {name}")
        return

    run_simulation(args.config, show_plot=not args.no_plot)


if __name__ == '__main__':
    main()
