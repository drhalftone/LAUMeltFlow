"""Run a 1D Sod shock tube using the trained SodMPGNN for flux.

Mirrors simulate_with_mlp.py's structure but uses our message-passing GNN
as the flux predictor. All interior interfaces are computed in a single
batched forward pass per timestep.

Also runs the analytical Roe solver (reference) and optionally the
existing MLP (baseline) for comparison. Plots all three on density,
velocity, pressure at t = 7.5e-4 s.

Usage:
    python sod_simulate_mpgnn.py
    python sod_simulate_mpgnn.py --mlp_npz ../models/flux_model_cuda.npz
"""

import os
import argparse
import time
import numpy as np
import torch

from sod_mpgnn import SodMPGNN
from grid_sampler import roe_flux1D, prim_to_cons


def load_mpgnn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = SodMPGNN(
        node_dim=3, edge_dim=1, output_dim=3,
        hidden_dim=a["hidden_dim"],
        n_message_passes=a["n_message_passes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded SodMPGNN: hidden_dim={a['hidden_dim']}, K={a['n_message_passes']}, "
          f"val_loss={ckpt['val_loss']:.4e}")
    return model


def sod_initial_conditions(nx, x_min=0.0, x_max=1.0):
    """Standard 1D Sod shock tube: left high state, right low state."""
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(nx)
    p = np.where(x < 0.5, 1.0e5, 1.0e4)
    return x, dx, rho, u, p


def step_mpgnn(rho, u, p, dx, dt, gamma, model, device):
    """One FV timestep using the MPGNN as flux predictor."""
    nx = rho.shape[0]
    state = torch.from_numpy(
        np.stack([rho, u, p], axis=1).astype(np.float32)
    ).to(device)

    # Pad with one ghost cell each side (transmissive)
    state_padded = torch.cat([state[:1], state, state[-1:]], dim=0)  # (nx+2, 3)

    # MPGNN: interior interfaces = (nx+1) interfaces between nx+2 cells
    with torch.no_grad():
        flux_t = model.predict_flux_at_interfaces(state_padded, dx)
    flux = flux_t.cpu().numpy()  # (nx+1, 3)

    # FV update
    E = p / (gamma - 1) + 0.5 * rho * u ** 2
    W = np.stack([rho, rho * u, E], axis=1)  # (nx, 3)
    W_new = W - dt / dx * (flux[1:] - flux[:-1])

    rho_n = W_new[:, 0]
    u_n = W_new[:, 1] / rho_n
    E_n = W_new[:, 2]
    p_n = (gamma - 1) * (E_n - 0.5 * rho_n * u_n ** 2)
    return rho_n, u_n, p_n


def step_roe(rho, u, p, dx, dt, gamma):
    """One FV timestep using the analytical Roe flux (reference)."""
    nx = rho.shape[0]
    flux = np.zeros((nx + 1, 3))
    for i in range(nx + 1):
        if i == 0:
            iL, iR = 0, 0
        elif i == nx:
            iL, iR = nx - 1, nx - 1
        else:
            iL, iR = i - 1, i
        W_L = prim_to_cons(rho[iL], u[iL], p[iL], gamma)
        W_R = prim_to_cons(rho[iR], u[iR], p[iR], gamma)
        flux[i] = roe_flux1D(n_dim=1, gam=gamma, W_L=W_L, W_R=W_R)

    E = p / (gamma - 1) + 0.5 * rho * u ** 2
    W = np.stack([rho, rho * u, E], axis=1)
    W_new = W - dt / dx * (flux[1:] - flux[:-1])

    rho_n = W_new[:, 0]
    u_n = W_new[:, 1] / rho_n
    E_n = W_new[:, 2]
    p_n = (gamma - 1) * (E_n - 0.5 * rho_n * u_n ** 2)
    return rho_n, u_n, p_n


def run_sim(stepper, nx, t_final, cfl, gamma, label):
    """Drive a Sod simulation with the given per-step stepper."""
    x, dx, rho, u, p = sod_initial_conditions(nx)
    t = 0.0
    n_steps = 0
    t0 = time.time()

    while t < t_final:
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        dt = cfl * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t

        rho, u, p = stepper(rho, u, p, dx, dt)

        if np.any(rho <= 0) or np.any(p <= 0) or np.any(np.isnan(rho)):
            print(f"  [{label}] WARNING: invalid state at t={t:.6e}, step={n_steps}")
            print(f"    min rho={rho.min():.4e}, min p={p.min():.4e}")
            break

        t += dt
        n_steps += 1

    elapsed = time.time() - t0
    print(f"  [{label}] {n_steps} steps, t={t:.6e}, {elapsed:.2f} s")
    return dict(x=x, rho=rho, u=u, p=p, t=t, n_steps=n_steps,
                wall_time_s=elapsed)


def plot_comparison(results, save_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                         "axes.labelsize": 13, "legend.fontsize": 11})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    field_names = [("rho", "Density [kg/m³]"),
                   ("u", "Velocity [m/s]"),
                   ("p", "Pressure [Pa]")]
    colors = {"Roe (reference)": "k", "MPGNN": "tab:blue",
              "MLP (baseline)": "tab:orange"}
    styles = {"Roe (reference)": "-", "MPGNN": "--", "MLP (baseline)": ":"}

    for ax, (field, ylabel) in zip(axes, field_names):
        for name, r in results.items():
            ax.plot(r["x"], r[field], styles.get(name, "-"),
                    label=name, color=colors.get(name, None),
                    linewidth=2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" [")[0])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def compute_mae(results, reference_name="Roe (reference)"):
    """Mean absolute error vs. the reference solution for each method."""
    ref = results[reference_name]
    errs = {}
    for name, r in results.items():
        if name == reference_name:
            continue
        errs[name] = dict(
            rho_mae=float(np.mean(np.abs(r["rho"] - ref["rho"]))),
            u_mae=float(np.mean(np.abs(r["u"] - ref["u"]))),
            p_mae=float(np.mean(np.abs(r["p"] - ref["p"]))),
        )
    return errs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load MPGNN
    model = load_mpgnn(args.mpgnn_ckpt, device)

    # Optionally load MLP baseline
    mlp = None
    if args.mlp_npz and os.path.exists(args.mlp_npz):
        from train_uniform import FluxMLP
        mlp, mlp_stats, mlp_config = FluxMLP.load(args.mlp_npz)
        print(f"Loaded MLP baseline from {args.mlp_npz}")

    results = {}

    # Reference: analytical Roe
    print("\nRoe (reference):")
    results["Roe (reference)"] = run_sim(
        lambda rho, u, p, dx, dt: step_roe(rho, u, p, dx, dt, args.gamma),
        args.nx, args.t_final, args.cfl, args.gamma, "Roe")

    # MPGNN
    print("\nMPGNN:")

    def step_mpgnn_wrapper(rho, u, p, dx, dt):
        return step_mpgnn(rho, u, p, dx, dt, args.gamma, model, device)

    results["MPGNN"] = run_sim(step_mpgnn_wrapper, args.nx, args.t_final,
                               args.cfl, args.gamma, "MPGNN")

    # MLP baseline (if available)
    if mlp is not None:
        from simulate_with_mlp import mlp_flux

        def step_mlp(rho, u, p, dx, dt):
            nx = rho.shape[0]
            flux = np.zeros((nx + 1, 3))
            for i in range(nx + 1):
                if i == 0:
                    iL, iR = 0, 0
                elif i == nx:
                    iL, iR = nx - 1, nx - 1
                else:
                    iL, iR = i - 1, i
                flux[i] = mlp_flux(mlp, mlp_stats, rho[iL], u[iL], p[iL],
                                   rho[iR], u[iR], p[iR])
            E = p / (args.gamma - 1) + 0.5 * rho * u ** 2
            W = np.stack([rho, rho * u, E], axis=1)
            W_new = W - dt / dx * (flux[1:] - flux[:-1])
            rho_n = W_new[:, 0]
            u_n = W_new[:, 1] / rho_n
            E_n = W_new[:, 2]
            p_n = (args.gamma - 1) * (E_n - 0.5 * rho_n * u_n ** 2)
            return rho_n, u_n, p_n

        print("\nMLP (baseline):")
        results["MLP (baseline)"] = run_sim(step_mlp, args.nx, args.t_final,
                                            args.cfl, args.gamma, "MLP")

    # Error metrics
    errs = compute_mae(results)
    print("\n=== MAE vs. analytical Roe ===")
    for name, e in errs.items():
        print(f"  {name}:  rho={e['rho_mae']:.4e}, "
              f"u={e['u_mae']:.4e}, p={e['p_mae']:.4e}")

    # Save data
    np.savez(args.output_npz,
             **{f"{k}_{f}": v[f] for k, v in results.items()
                for f in ("x", "rho", "u", "p")},
             errors=errs)
    print(f"\nSaved {args.output_npz}")

    # Plot
    plot_comparison(results, args.output_png,
                    title=f"Sod shock tube at t = {args.t_final*1e3:.3f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mpgnn_ckpt", type=str,
                   default="outputs/sod_mpgnn_best.pt")
    p.add_argument("--mlp_npz", type=str, default="",
                   help="Path to existing MLP checkpoint (.npz) for baseline comparison.")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--t_final", type=float, default=7.5e-4)
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--output_npz", type=str,
                   default="outputs/sod_comparison.npz")
    p.add_argument("--output_png", type=str,
                   default="outputs/sod_comparison.png")
    args = p.parse_args()
    main(args)
