"""Run a 2D Sod shock tube using SodMPGNN_2D for flux.

Adapts simulate_2d_comparison.py to use the message-passing GNN. Each
timestep:
  1. Build (W, C, E, S, N) stencils for every interior cell in a single
     batched tensor (with transmissive ghost padding at boundaries).
  2. One MPGNN forward pass yields 4 fluxes per cell.
  3. Assemble F_x and G_y arrays from the per-cell predictions
     (use F_e for east faces, G_n for north faces, plus F_w at i=0 and
     G_s at j=0 for the boundary faces).
  4. Finite-volume update via dimensional splitting.

Compares against analytical Roe 2D reference and produces:
  - Static figure: 3-panel density heatmaps (Roe, MPGNN, |error|)
                   + 1D profile slice along y = center
  - Animation: density heatmap evolving for both methods

Usage:
    python sod_simulate_mpgnn_2d.py
"""

import os
import argparse
import time
import numpy as np
import torch

from sod_mpgnn_2d import SodMPGNN_2D
from sod_train_mpgnn_2d import build_edge_attrs
from grid_sampler_2d import roe_flux2D


def load_mpgnn_2d(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = SodMPGNN_2D(
        node_dim=4, edge_dim=3, output_dim=4,
        hidden_dim=a["hidden_dim"],
        n_message_passes=a["n_message_passes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded SodMPGNN_2D: h={a['hidden_dim']}, K={a['n_message_passes']}, "
          f"val_loss={ckpt['val_loss']:.4e}")
    return model


def init_sod_2d(nx, ny, x_max=2.0, y_max=1.0, diaphragm=1.0):
    """Standard 2D Sod tube: high state left of diaphragm, low state right."""
    dx = x_max / nx
    dy = y_max / ny
    x = np.linspace(dx / 2, x_max - dx / 2, nx)
    y = np.linspace(dy / 2, y_max - dy / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = np.where(X < diaphragm, 1.0, 0.125)
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.where(X < diaphragm, 1.0e5, 1.0e4)
    return x, y, dx, dy, rho, u, v, p


def cons_to_prim(W, gamma):
    rho = W[0]
    u = W[1] / rho
    v = W[2] / rho
    E = W[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u ** 2 + v ** 2))
    return rho, u, v, p


def step_roe_2d(rho, u, v, p, dx, dy, dt, gamma):
    nx, ny = rho.shape
    E = p / (gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2)
    W = np.stack([rho, rho * u, rho * v, E], axis=0)  # (4, nx, ny)

    # X-direction fluxes at (i+1/2, j) for i = 0..nx
    F_x = np.zeros((4, nx + 1, ny))
    for i in range(nx + 1):
        iL = max(i - 1, 0)
        iR = min(i, nx - 1)
        for j in range(ny):
            F_x[:, i, j] = roe_flux2D(dim=2, gam=gamma,
                                      W_L=W[:, iL, j], W_R=W[:, iR, j])

    # Y-direction fluxes at (i, j+1/2)
    G_y = np.zeros((4, nx, ny + 1))
    for i in range(nx):
        for j in range(ny + 1):
            jB = max(j - 1, 0)
            jT = min(j, ny - 1)
            G_y[:, i, j] = roe_flux2D(dim=1, gam=gamma,
                                      W_L=W[:, i, jB], W_R=W[:, i, jT])

    # Update
    W_new = W.copy()
    W_new -= dt / dx * (F_x[:, 1:, :] - F_x[:, :-1, :])
    W_new -= dt / dy * (G_y[:, :, 1:] - G_y[:, :, :-1])
    return cons_to_prim(W_new, gamma)


def step_mpgnn_2d(rho, u, v, p, dx, dy, dt, gamma, model, edge_attrs_cached, device):
    """One FV timestep using the MPGNN to predict per-cell fluxes."""
    nx, ny = rho.shape

    # Pad with one cell of transmissive ghosts on each side
    rho_p = np.pad(rho, 1, mode="edge")
    u_p = np.pad(u, 1, mode="edge")
    v_p = np.pad(v, 1, mode="edge")
    p_p = np.pad(p, 1, mode="edge")

    # Build stencils for every interior cell: shape (nx*ny, 5, 4)
    # Cell ordering: W, C, E, S, N
    states = np.stack([rho_p, u_p, v_p, p_p], axis=-1)  # (nx+2, ny+2, 4)
    W = states[0:nx,     1:ny+1]    # (nx, ny, 4)
    C = states[1:nx+1,   1:ny+1]
    E = states[2:nx+2,   1:ny+1]
    S = states[1:nx+1,   0:ny]
    N = states[1:nx+1,   2:ny+2]
    stencils = np.stack([W, C, E, S, N], axis=2)  # (nx, ny, 5, 4)
    stencils_flat = stencils.reshape(-1, 5, 4).astype(np.float32)
    n_cells = stencils_flat.shape[0]

    nodes_t = torch.from_numpy(stencils_flat).to(device)
    edges_t = edge_attrs_cached[:n_cells].to(device) \
        if edge_attrs_cached.shape[0] >= n_cells \
        else build_edge_attrs(n_cells, dx, dy).to(device)

    # Normalize and predict
    nodes_n = (nodes_t - model.node_mean) / model.node_std
    edges_n = (edges_t - model.edge_mean) / model.edge_std
    with torch.no_grad():
        flux_n = model.forward_stencil(nodes_n, edges_n)  # (n_cells, 4, 4)
    flux = (flux_n * model.y_std + model.y_mean).cpu().numpy()
    flux = flux.reshape(nx, ny, 4, 4)  # (nx, ny, 4_faces, 4_components)

    # Flux indices (matches model output ordering): 0=F_w, 1=F_e, 2=G_s, 3=G_n
    F_e = flux[:, :, 1, :]  # (nx, ny, 4)   east face of each cell
    F_w = flux[:, :, 0, :]
    G_n = flux[:, :, 3, :]
    G_s = flux[:, :, 2, :]

    # Assemble F_x (nx+1, ny, 4)
    F_x = np.zeros((nx + 1, ny, 4))
    F_x[0, :, :] = F_w[0, :, :]           # left boundary: F_w of leftmost cell
    F_x[1:, :, :] = F_e[:, :, :]           # interior + right boundary: F_e of each cell
    F_x = np.transpose(F_x, (2, 0, 1))     # (4, nx+1, ny)

    G_y = np.zeros((nx, ny + 1, 4))
    G_y[:, 0, :] = G_s[:, 0, :]
    G_y[:, 1:, :] = G_n[:, :, :]
    G_y = np.transpose(G_y, (2, 0, 1))

    # FV update
    E_arr = p / (gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2)
    W_arr = np.stack([rho, rho * u, rho * v, E_arr], axis=0)
    W_new = W_arr.copy()
    W_new -= dt / dx * (F_x[:, 1:, :] - F_x[:, :-1, :])
    W_new -= dt / dy * (G_y[:, :, 1:] - G_y[:, :, :-1])
    return cons_to_prim(W_new, gamma)


def run_sim(stepper, nx, ny, t_final, cfl, gamma, label, sample_every=2):
    x, y, dx, dy, rho, u, v, p = init_sod_2d(nx, ny)
    t = 0.0
    n_steps = 0
    times = [t]
    rhos = [rho.copy()]
    t0 = time.time()
    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a) + np.max(np.abs(v) + a)
        dt = cfl * min(dx, dy) / max_speed
        if t + dt > t_final:
            dt = t_final - t
        rho, u, v, p = stepper(rho, u, v, p, dx, dy, dt)
        if np.any(rho <= 0) or np.any(p <= 0) or np.any(np.isnan(rho)):
            print(f"  [{label}] invalid state at t={t:.4e}, stopping")
            break
        t += dt
        n_steps += 1
        if n_steps % sample_every == 0 or t >= t_final:
            times.append(t)
            rhos.append(rho.copy())
    elapsed = time.time() - t0
    print(f"  [{label}] {n_steps} steps, {elapsed:.2f} s, "
          f"final t={t:.4e}, {len(times)} frames")
    return dict(x=x, y=y, dx=dx, dy=dy, t=np.array(times),
                rho_frames=np.stack(rhos),
                rho=rho, u=u, v=v, p=p, n_steps=n_steps,
                wall_time_s=elapsed)


def plot_static(roe, mpgnn, save_path, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                         "legend.fontsize": 10})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = roe["x"]
    y = roe["y"]
    extent = [x[0], x[-1], y[0], y[-1]]

    rho_min = min(roe["rho"].min(), mpgnn["rho"].min())
    rho_max = max(roe["rho"].max(), mpgnn["rho"].max())

    im0 = axes[0, 0].imshow(roe["rho"].T, origin="lower", extent=extent,
                            aspect="auto", cmap="viridis",
                            vmin=rho_min, vmax=rho_max)
    axes[0, 0].set_title("Density - Roe (reference)")
    axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(mpgnn["rho"].T, origin="lower", extent=extent,
                            aspect="auto", cmap="viridis",
                            vmin=rho_min, vmax=rho_max)
    axes[0, 1].set_title("Density - MPGNN")
    axes[0, 1].set_xlabel("x [m]"); axes[0, 1].set_ylabel("y [m]")
    plt.colorbar(im1, ax=axes[0, 1])

    err = np.abs(mpgnn["rho"] - roe["rho"])
    im2 = axes[0, 2].imshow(err.T, origin="lower", extent=extent,
                            aspect="auto", cmap="hot")
    axes[0, 2].set_title("|Error| Density")
    axes[0, 2].set_xlabel("x [m]"); axes[0, 2].set_ylabel("y [m]")
    plt.colorbar(im2, ax=axes[0, 2])

    # 1D slice along y = mid
    j_mid = roe["rho"].shape[1] // 2
    axes[1, 0].plot(x, roe["rho"][:, j_mid], "k-", linewidth=2,
                    label="Roe (reference)")
    axes[1, 0].plot(x, mpgnn["rho"][:, j_mid], "b--", linewidth=2, label="MPGNN")
    axes[1, 0].set_xlabel("x [m]"); axes[1, 0].set_ylabel("Density [kg/m³]")
    axes[1, 0].set_title(f"Density slice at y={y[j_mid]:.2f}")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, roe["u"][:, j_mid], "k-", linewidth=2,
                    label="Roe (reference)")
    axes[1, 1].plot(x, mpgnn["u"][:, j_mid], "b--", linewidth=2, label="MPGNN")
    axes[1, 1].set_xlabel("x [m]"); axes[1, 1].set_ylabel("u [m/s]")
    axes[1, 1].set_title(f"x-Velocity at y={y[j_mid]:.2f}")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(x, roe["p"][:, j_mid], "k-", linewidth=2,
                    label="Roe (reference)")
    axes[1, 2].plot(x, mpgnn["p"][:, j_mid], "b--", linewidth=2, label="MPGNN")
    axes[1, 2].set_xlabel("x [m]"); axes[1, 2].set_ylabel("p [Pa]")
    axes[1, 2].set_title(f"Pressure at y={y[j_mid]:.2f}")
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f"2D Sod Shock Tube: MPGNN vs Analytical Roe at t = "
                 f"{t_final*1e3:.3f} ms", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def animate_2d(roe, mpgnn, save_path, fps, t_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    n_frames = min(roe["rho_frames"].shape[0], mpgnn["rho_frames"].shape[0])
    x = roe["x"]; y = roe["y"]
    extent = [x[0], x[-1], y[0], y[-1]]

    rho_min = min(roe["rho_frames"].min(), mpgnn["rho_frames"].min())
    rho_max = max(roe["rho_frames"].max(), mpgnn["rho_frames"].max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im_r = axes[0].imshow(roe["rho_frames"][0].T, origin="lower",
                          extent=extent, aspect="auto", cmap="viridis",
                          vmin=rho_min, vmax=rho_max)
    axes[0].set_title("Roe (reference)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    plt.colorbar(im_r, ax=axes[0])

    im_m = axes[1].imshow(mpgnn["rho_frames"][0].T, origin="lower",
                          extent=extent, aspect="auto", cmap="viridis",
                          vmin=rho_min, vmax=rho_max)
    axes[1].set_title("MPGNN")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")
    plt.colorbar(im_m, ax=axes[1])

    suptitle = fig.suptitle("", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        im_r.set_data(roe["rho_frames"][frame].T)
        im_m.set_data(mpgnn["rho_frames"][frame].T)
        suptitle.set_text(f"2D Sod density at t = {roe['t'][frame]*1e3:.3f} ms")
        return im_r, im_m, suptitle

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved {save_path}")
    except Exception as e:
        gif = os.path.splitext(save_path)[0] + ".gif"
        print(f"ffmpeg unavailable ({e}); writing .gif")
        anim.save(gif, writer="pillow", fps=fps, dpi=100)
        print(f"Saved {gif}")
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_mpgnn_2d(args.mpgnn_ckpt, device)

    # Pre-build edge attrs cache (constant across all cells/steps)
    _, _, dx, dy, _, _, _, _ = init_sod_2d(args.nx, args.ny)
    edge_cache = build_edge_attrs(args.nx * args.ny, dx, dy)

    def step_r(rho, u, v, p, dx, dy, dt):
        return step_roe_2d(rho, u, v, p, dx, dy, dt, args.gamma)

    def step_m(rho, u, v, p, dx, dy, dt):
        return step_mpgnn_2d(rho, u, v, p, dx, dy, dt, args.gamma,
                             model, edge_cache, device)

    print("\nRoe (reference):")
    roe = run_sim(step_r, args.nx, args.ny, args.t_final, args.cfl,
                  args.gamma, "Roe", sample_every=args.sample_every)
    print("\nMPGNN:")
    mp = run_sim(step_m, args.nx, args.ny, args.t_final, args.cfl,
                 args.gamma, "MPGNN", sample_every=args.sample_every)

    rho_mae = float(np.mean(np.abs(mp["rho"] - roe["rho"])))
    u_mae = float(np.mean(np.abs(mp["u"] - roe["u"])))
    v_mae = float(np.mean(np.abs(mp["v"] - roe["v"])))
    p_mae = float(np.mean(np.abs(mp["p"] - roe["p"])))
    print("\n=== MAE vs analytical Roe (at t_final) ===")
    print(f"  rho: {rho_mae:.4e} kg/m^3")
    print(f"  u:   {u_mae:.4e} m/s")
    print(f"  v:   {v_mae:.4e} m/s")
    print(f"  p:   {p_mae:.4e} Pa")

    os.makedirs(os.path.dirname(args.static_out) or ".", exist_ok=True)
    plot_static(roe, mp, args.static_out, args.t_final)
    animate_2d(roe, mp, args.anim_out, args.fps, args.t_final)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mpgnn_ckpt", type=str,
                   default="outputs/sod_mpgnn_2d_best.pt")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=50)
    p.add_argument("--t_final", type=float, default=4.0e-4)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--static_out", type=str,
                   default="outputs/sod2d_comparison.png")
    p.add_argument("--anim_out", type=str,
                   default="outputs/sod2d_evolution.mp4")
    args = p.parse_args()
    main(args)
