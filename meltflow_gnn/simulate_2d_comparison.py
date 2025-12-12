"""
2D Sod Shock Tube Simulation: MLP vs Analytical Roe Comparison
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '.')
from meltflow_gnn.grid_sampler_2d import roe_flux2D, prim_to_cons_2d
from meltflow_gnn.train_uniform_2d_cuda import FluxMLP2D

print('=' * 60)
print('2D Sod Shock Tube: MLP vs Analytical Roe Comparison')
print('=' * 60)

# Load model
print('\nLoading 2D MLP model...')
checkpoint = torch.load('flux_model_2d_cuda.pt', weights_only=False)
model = FluxMLP2D(hidden_dim=256, n_layers=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

stats = checkpoint['stats']
X_mean = torch.tensor(stats['X_mean'])
X_std = torch.tensor(stats['X_std'])
Y_mean = torch.tensor(stats['Y_mean'])
Y_std = torch.tensor(stats['Y_std'])

# Simulation parameters
gamma = 1.4
nx, ny = 50, 25  # Grid size
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 1.0
dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
t_final = 0.0004
cfl = 0.3
diaphragm = 0.5

print(f'Grid: {nx} x {ny} cells')
print(f'Final time: {t_final}')

# Cell centers
x = np.linspace(x_min + dx/2, x_max - dx/2, nx)
y = np.linspace(y_min + dy/2, y_max - dy/2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')


def init_sod():
    rho = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if x[i] < diaphragm:
                rho[i, j], u[i, j], v[i, j], p[i, j] = 1.0, 0.0, 0.0, 100000.0
            else:
                rho[i, j], u[i, j], v[i, j], p[i, j] = 0.125, 0.0, 0.0, 10000.0
    return rho, u, v, p


def cons_to_prim(W, gamma):
    rho = W[0]
    u = W[1] / rho
    v = W[2] / rho
    E = W[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


def run_simulation_analytical():
    rho, u, v, p = init_sod()
    t = 0.0
    n_steps = 0

    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a) + np.max(np.abs(v) + a)
        dt = cfl * min(dx, dy) / max_speed
        if t + dt > t_final:
            dt = t_final - t

        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        W = np.stack([rho, rho * u, rho * v, E], axis=0)

        # X-direction fluxes
        F_x = np.zeros((4, nx + 1, ny))
        for i in range(nx + 1):
            for j in range(ny):
                if i == 0:
                    i_L, i_R = 0, 0
                elif i == nx:
                    i_L, i_R = nx - 1, nx - 1
                else:
                    i_L, i_R = i - 1, i
                F_x[:, i, j] = roe_flux2D(dim=2, gam=gamma, W_L=W[:, i_L, j], W_R=W[:, i_R, j])

        # Y-direction fluxes
        G_y = np.zeros((4, nx, ny + 1))
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    j_B, j_T = 0, 0
                elif j == ny:
                    j_B, j_T = ny - 1, ny - 1
                else:
                    j_B, j_T = j - 1, j
                G_y[:, i, j] = roe_flux2D(dim=1, gam=gamma, W_L=W[:, i, j_B], W_R=W[:, i, j_T])

        # Update
        for i in range(nx):
            for j in range(ny):
                W[:, i, j] -= dt / dx * (F_x[:, i + 1, j] - F_x[:, i, j]) + dt / dy * (G_y[:, i, j + 1] - G_y[:, i, j])

        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            break

        rho, u, v, p = cons_to_prim(W, gamma)
        if np.any(p <= 0):
            break

        t += dt
        n_steps += 1

    return rho, u, v, p, t, n_steps


def mlp_flux_x(rho_L, u_L, v_L, p_L, rho_R, u_R, v_R, p_R):
    """Compute x-flux using MLP."""
    # For x-interface: approximate 5-cell stencil with available data
    inp = np.array([rho_L, u_L, v_L, p_L,  # W
                    rho_L, u_L, v_L, p_L,  # C
                    rho_R, u_R, v_R, p_R,  # E
                    rho_L, u_L, v_L, p_L,  # S
                    rho_L, u_L, v_L, p_L])  # N
    with torch.no_grad():
        x_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
        x_norm = (x_tensor - X_mean) / X_std
        y_norm = model(x_norm)
        y = y_norm * Y_std + Y_mean
    # Return F_e (east face flux) - indices 4:8
    return y[0, 4:8].numpy()


def mlp_flux_y(rho_B, u_B, v_B, p_B, rho_T, u_T, v_T, p_T):
    """Compute y-flux using MLP."""
    inp = np.array([rho_B, u_B, v_B, p_B,  # W
                    rho_B, u_B, v_B, p_B,  # C
                    rho_B, u_B, v_B, p_B,  # E
                    rho_B, u_B, v_B, p_B,  # S
                    rho_T, u_T, v_T, p_T])  # N
    with torch.no_grad():
        x_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
        x_norm = (x_tensor - X_mean) / X_std
        y_norm = model(x_norm)
        y = y_norm * Y_std + Y_mean
    # Return G_n (north face flux) - indices 12:16
    return y[0, 12:16].numpy()


def run_simulation_mlp():
    rho, u, v, p = init_sod()
    t = 0.0
    n_steps = 0

    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a) + np.max(np.abs(v) + a)
        dt = cfl * min(dx, dy) / max_speed
        if t + dt > t_final:
            dt = t_final - t

        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        W = np.stack([rho, rho * u, rho * v, E], axis=0)

        # X-direction fluxes using MLP
        F_x = np.zeros((4, nx + 1, ny))
        for i in range(nx + 1):
            for j in range(ny):
                if i == 0:
                    i_L, i_R = 0, 0
                elif i == nx:
                    i_L, i_R = nx - 1, nx - 1
                else:
                    i_L, i_R = i - 1, i
                rho_L, u_L, v_L = rho[i_L, j], u[i_L, j], v[i_L, j]
                p_L = p[i_L, j]
                rho_R, u_R, v_R = rho[i_R, j], u[i_R, j], v[i_R, j]
                p_R = p[i_R, j]
                F_x[:, i, j] = mlp_flux_x(rho_L, u_L, v_L, p_L, rho_R, u_R, v_R, p_R)

        # Y-direction fluxes using MLP
        G_y = np.zeros((4, nx, ny + 1))
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    j_B, j_T = 0, 0
                elif j == ny:
                    j_B, j_T = ny - 1, ny - 1
                else:
                    j_B, j_T = j - 1, j
                rho_B, u_B, v_B = rho[i, j_B], u[i, j_B], v[i, j_B]
                p_B = p[i, j_B]
                rho_T, u_T, v_T = rho[i, j_T], u[i, j_T], v[i, j_T]
                p_T = p[i, j_T]
                G_y[:, i, j] = mlp_flux_y(rho_B, u_B, v_B, p_B, rho_T, u_T, v_T, p_T)

        # Update
        for i in range(nx):
            for j in range(ny):
                W[:, i, j] -= dt / dx * (F_x[:, i + 1, j] - F_x[:, i, j]) + dt / dy * (G_y[:, i, j + 1] - G_y[:, i, j])

        if np.any(W[0] <= 0) or np.any(np.isnan(W)):
            print(f'  MLP: Invalid state at step {n_steps}')
            break

        rho, u, v, p = cons_to_prim(W, gamma)
        if np.any(p <= 0):
            print(f'  MLP: Negative pressure at step {n_steps}')
            break

        t += dt
        n_steps += 1
        if n_steps % 10 == 0:
            print(f'  MLP Step {n_steps}: t = {t:.6f}')

    return rho, u, v, p, t, n_steps


# Run both simulations
print('\n1. Running analytical Roe simulation...')
start = time.time()
rho_ana, u_ana, v_ana, p_ana, t_ana, n_ana = run_simulation_analytical()
time_ana = time.time() - start
print(f'   Completed: {n_ana} steps, t = {t_ana:.6f}, time = {time_ana:.1f}s')

print('\n2. Running MLP simulation...')
start = time.time()
rho_mlp, u_mlp, v_mlp, p_mlp, t_mlp, n_mlp = run_simulation_mlp()
time_mlp = time.time() - start
print(f'   Completed: {n_mlp} steps, t = {t_mlp:.6f}, time = {time_mlp:.1f}s')

# Comparison metrics
print('\n3. Comparison metrics:')
rho_err = np.abs(rho_mlp - rho_ana).mean()
u_err = np.abs(u_mlp - u_ana).mean()
p_err = np.abs(p_mlp - p_ana).mean()
print(f'   Density MAE: {rho_err:.6f}')
print(f'   Velocity MAE: {u_err:.2f}')
print(f'   Pressure MAE: {p_err:.0f}')

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

j_mid = ny // 2

# Row 1: 2D contour plots
im0 = axes[0, 0].contourf(X, Y, rho_ana, levels=30, cmap='viridis')
axes[0, 0].set_title('Density - Analytical Roe')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0, 0])
axes[0, 0].set_aspect('equal')

im1 = axes[0, 1].contourf(X, Y, rho_mlp, levels=30, cmap='viridis')
axes[0, 1].set_title('Density - MLP')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 1])
axes[0, 1].set_aspect('equal')

im2 = axes[0, 2].contourf(X, Y, np.abs(rho_mlp - rho_ana), levels=30, cmap='hot')
axes[0, 2].set_title('Density Error |MLP - Roe|')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('y')
plt.colorbar(im2, ax=axes[0, 2])
axes[0, 2].set_aspect('equal')

# Row 2: 1D slices
axes[1, 0].plot(x, rho_ana[:, j_mid], 'b-', label='Analytical', linewidth=2)
axes[1, 0].plot(x, rho_mlp[:, j_mid], 'r--', label='MLP', linewidth=2)
axes[1, 0].set_xlabel('x [m]')
axes[1, 0].set_ylabel('Density [kg/m³]')
axes[1, 0].set_title(f'Density at y = {y[j_mid]:.2f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, u_ana[:, j_mid], 'b-', label='Analytical', linewidth=2)
axes[1, 1].plot(x, u_mlp[:, j_mid], 'r--', label='MLP', linewidth=2)
axes[1, 1].set_xlabel('x [m]')
axes[1, 1].set_ylabel('Velocity [m/s]')
axes[1, 1].set_title(f'X-Velocity at y = {y[j_mid]:.2f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(x, p_ana[:, j_mid], 'b-', label='Analytical', linewidth=2)
axes[1, 2].plot(x, p_mlp[:, j_mid], 'r--', label='MLP', linewidth=2)
axes[1, 2].set_xlabel('x [m]')
axes[1, 2].set_ylabel('Pressure [Pa]')
axes[1, 2].set_title(f'Pressure at y = {y[j_mid]:.2f}')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f'2D Sod Shock Tube: MLP vs Analytical Roe (t = {t_ana:.5f} s)', fontsize=14)
plt.tight_layout()
plt.savefig('simulation_2d_comparison.png', dpi=150)
print('\nSaved comparison to simulation_2d_comparison.png')
