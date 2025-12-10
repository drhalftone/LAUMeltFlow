"""
Plotting functions for MeltFlow solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameters import Parameters


def plt_setup(prm: 'Parameters') -> Figure:
    """
    Set up plot for primitive variable data U over grid (X,Y) and level set phi.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    opt_plt = prm.opt_plt
    plt_ps = prm.plt_ps
    plt_wn = prm.plt_wn

    # Convert position/size to inches (approximate conversion from pixels)
    figsize = (plt_ps[2] / 100, plt_ps[3] / 100)

    if n_dim == 1:
        # 1D Case
        U_labels = [r"$T$ [K]", r"$u$ [m/s]", r"$p$ [Pa]", r"$\phi$ [m]"]
        U_titles = ["Temperature", "Velocity", "Pressure", "Level Set"]

        plt_hn = int(np.ceil((n_var + 1) / plt_wn))
        fig, axes = plt.subplots(plt_hn, plt_wn, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        fig.suptitle('Flow Field')
        fig.supxlabel(r"$x$ [m]")

        for k in range(n_var + 1):
            if k < len(axes):
                axes[k].set_ylabel(U_labels[k])
                axes[k].set_title(U_titles[k])

        plt.tight_layout()

    elif n_dim == 2:
        # 2D Case
        if opt_plt == 1:
            # Contour with velocity vectors
            plt_hn = n_var - plt_wn
            fig, axes = plt.subplots(plt_hn, plt_wn, figsize=figsize)
            fig.suptitle('Flow Field')
            fig.supxlabel(r"$x$")
            fig.supylabel(r"$y$")

        elif opt_plt == 2:
            # Surfaces
            plt_hn = int(np.floor((n_var + 2) / plt_wn))
            fig = plt.figure(figsize=figsize)

            for k in range(n_var + 1):
                ax = fig.add_subplot(plt_hn, plt_wn, k + 1, projection='3d')
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")

            fig.suptitle('Flow Field')
        else:
            fig, axes = plt.subplots(figsize=figsize)

        plt.tight_layout()

    return fig


def plot(fig: Figure, prm: 'Parameters', X: np.ndarray, U: np.ndarray,
         phi: np.ndarray) -> None:
    """
    Plot primitive variable data U and level set phi over grid (X,Y).

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    U : np.ndarray
        Primitive variables [rho, u, (v), p]^T
    phi : np.ndarray
        Level set function
    """
    n_dim = prm.n_dim
    n_var = prm.n_var
    n_out = prm.n_out
    opt_plt = prm.opt_plt
    flg_fld = prm.flg_fld
    c_EoS = prm.c_EoS
    flg_vec = prm.flg_vec
    n_vec = prm.n_vec
    x_min = prm.x_min
    x_max = prm.x_max

    gamma = c_EoS  # Specific heat ratios

    if n_dim == 1:
        # 1D Case
        axes = fig.get_axes()

        # Compute temperature
        cp = [1005, 1378]  # Specific heat capacities
        T = np.zeros(n_out)
        for i in range(n_out):
            if phi[i] > 0:
                flag = 0
            else:
                flag = 1
            R = (gamma[flag] - 1) / gamma[flag] * cp[flag]
            T[i] = U[2, i] / (R * U[0, i])

        # Plot each variable
        for k in range(n_var + 1):
            if k < len(axes):
                axes[k].clear()
                if k == 0:
                    axes[k].plot(X, T, '-o')
                    axes[k].set_ylabel(r"$T$ [K]")
                    axes[k].set_title("Temperature")
                elif k < n_var:
                    axes[k].plot(X, U[k, :], '-o')
                else:
                    axes[k].plot(X, phi)
                    axes[k].set_ylabel(r"$\phi$ [m]")
                    axes[k].set_title("Level Set")

    elif n_dim == 2:
        # 2D Case
        Y = X[1, :, :]
        X_grid = X[0, :, :]

        if opt_plt == 1:
            # Contour Map
            axes = fig.get_axes()

            # Velocity vectors
            if flg_vec > 0:
                u = U[1, :, :]
                v = U[2, :, :]
                clr_vec = [0.75, 0.75, 0.75]

                if flg_vec == 1:
                    X_vec, Y_vec = X_grid, Y
                    u_vec, v_vec = u, v
                elif flg_vec > 1:
                    n_vec = np.atleast_1d(n_vec).astype(int)
                    x_min = np.atleast_1d(x_min)
                    x_max = np.atleast_1d(x_max)
                    x_vec = np.linspace(x_min[0], x_max[0], n_vec[0])
                    y_vec = np.linspace(x_min[1], x_max[1], n_vec[1])
                    X_vec, Y_vec = np.meshgrid(x_vec, y_vec, indexing='ij')
                    # Interpolate velocities to vector grid
                    from scipy.interpolate import RegularGridInterpolator
                    f_u = RegularGridInterpolator(
                        (X_grid[:, 0], Y[0, :]), u, bounds_error=False, fill_value=0
                    )
                    f_v = RegularGridInterpolator(
                        (X_grid[:, 0], Y[0, :]), v, bounds_error=False, fill_value=0
                    )
                    pts = np.column_stack([X_vec.ravel(), Y_vec.ravel()])
                    u_vec = f_u(pts).reshape(X_vec.shape)
                    v_vec = f_v(pts).reshape(X_vec.shape)

            clr_lb = [r"$\rho$", r"$p$", r"$\phi$"]
            k_prp = [0, 3, -1]  # Indices for rho, p, phi

            for j, ax in enumerate(axes[:3]):
                ax.clear()
                if j == 2:
                    cf = ax.contourf(X_grid, Y, phi)
                else:
                    cf = ax.contourf(X_grid, Y, U[k_prp[j], :, :])

                plt.colorbar(cf, ax=ax, label=clr_lb[j])

                if flg_vec > 0:
                    ax.quiver(X_vec, Y_vec, u_vec, v_vec, color=clr_vec)

        elif opt_plt == 2:
            # Surfaces
            axes = fig.get_axes()
            U_labels = [r"$\rho$", r"$u$", r"$v$", r"$p$", r"$\phi$"]

            for k, ax in enumerate(axes[:n_var + 1]):
                ax.clear()
                if k == 4:
                    ax.plot_surface(X_grid, Y, phi, cmap='viridis')
                else:
                    ax.plot_surface(X_grid, Y, U[k, :, :], cmap='viridis')
                ax.set_zlabel(U_labels[k])

    fig.canvas.draw()
    plt.pause(0.01)


def animate(fig: Figure, prm: 'Parameters', X: np.ndarray, U: np.ndarray,
            phi: np.ndarray, t_pause: float = 0.05) -> None:
    """
    Animate the plot with a pause.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object
    prm : Parameters
        Simulation parameters
    X : np.ndarray
        Grid coordinates
    U : np.ndarray
        Primitive variables
    phi : np.ndarray
        Level set function
    t_pause : float
        Pause time in seconds
    """
    plot(fig, prm, X, U, phi)
    plt.pause(t_pause)
