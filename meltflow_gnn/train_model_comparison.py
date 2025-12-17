"""
Train and compare different model sizes for flux prediction.

Tests accuracy vs model size tradeoff.
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size, lr, device, verbose=True):
    """Train model with manual batching (fast, no DataLoader overhead)."""
    model = model.to(device)
    print(f"  Model on: {next(model.parameters()).device}", flush=True)

    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training with manual batching (GPU shuffle)
        model.train()
        perm = torch.randperm(n_train, device=device)

        train_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            idx = perm[start_idx:end_idx]

            X_batch = X_train[idx]
            Y_batch = Y_train[idx]

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= n_batches

        # Validation (single batch for speed)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_val)
            val_loss = criterion(Y_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.2e}, val_loss={val_loss:.2e}", flush=True)

    # Restore best model
    model.load_state_dict(best_state)
    return best_val_loss, model


def evaluate_simulation(model, stats, device, gamma=1.4):
    """Run Sod shock tube and compute error vs analytical."""
    model.eval()

    # Grid setup
    n_cells = 101
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells
    x = np.linspace(x_min + dx/2, x_max - dx/2, n_cells)

    # Initial conditions
    U_cons = np.zeros((3, n_cells))
    for i in range(n_cells):
        if x[i] < 0.5:
            rho, u, p = 1.0, 0.0, 1e5
        else:
            rho, u, p = 0.125, 0.0, 1e4
        U_cons[0, i] = rho
        U_cons[1, i] = rho * u
        U_cons[2, i] = p / (gamma - 1) + 0.5 * rho * u**2

    # Time stepping
    t_final = 0.00025
    t = 0.0
    cfl = 0.5

    while t < t_final:
        rho = U_cons[0, :]
        u = U_cons[1, :] / rho
        E = U_cons[2, :]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        c = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + c)
        if t + dt > t_final:
            dt = t_final - t

        F = np.zeros((3, n_cells + 1))

        for i in range(n_cells + 1):
            if i == 0:
                U_L = np.array([rho[0], u[0], p[0]])
                U_R = np.array([rho[0], u[0], p[0]])
            elif i == n_cells:
                U_L = np.array([rho[-1], u[-1], p[-1]])
                U_R = np.array([rho[-1], u[-1], p[-1]])
            else:
                U_L = np.array([rho[i-1], u[i-1], p[i-1]])
                U_R = np.array([rho[i], u[i], p[i]])

            # MLP flux
            X = np.array([[U_L[0], U_L[1], U_L[2], U_R[0], U_R[1], U_R[2], gamma]], dtype=np.float32)
            X_norm = (X - stats['X_mean']) / stats['X_std']
            with torch.no_grad():
                X_t = torch.from_numpy(X_norm).to(device)
                Y_pred = model(X_t).cpu().numpy()
            F[:, i] = Y_pred[0] * stats['Y_std'] + stats['Y_mean']

        for i in range(n_cells):
            U_cons[:, i] = U_cons[:, i] - dt / dx * (F[:, i+1] - F[:, i])

        t += dt

    # Convert to primitive
    rho_mlp = U_cons[0, :]
    u_mlp = U_cons[1, :] / rho_mlp
    p_mlp = (gamma - 1) * (U_cons[2, :] - 0.5 * rho_mlp * u_mlp**2)

    # Run analytical Roe for comparison
    U_cons_roe = np.zeros((3, n_cells))
    for i in range(n_cells):
        if x[i] < 0.5:
            rho, u, p = 1.0, 0.0, 1e5
        else:
            rho, u, p = 0.125, 0.0, 1e4
        U_cons_roe[0, i] = rho
        U_cons_roe[1, i] = rho * u
        U_cons_roe[2, i] = p / (gamma - 1) + 0.5 * rho * u**2

    t = 0.0
    while t < t_final:
        rho = U_cons_roe[0, :]
        u = U_cons_roe[1, :] / rho
        E = U_cons_roe[2, :]
        p = (gamma - 1) * (E - 0.5 * rho * u**2)

        c = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + c)
        if t + dt > t_final:
            dt = t_final - t

        F = np.zeros((3, n_cells + 1))

        for i in range(n_cells + 1):
            if i == 0:
                U_L = np.array([rho[0], u[0], p[0]])
                U_R = np.array([rho[0], u[0], p[0]])
            elif i == n_cells:
                U_L = np.array([rho[-1], u[-1], p[-1]])
                U_R = np.array([rho[-1], u[-1], p[-1]])
            else:
                U_L = np.array([rho[i-1], u[i-1], p[i-1]])
                U_R = np.array([rho[i], u[i], p[i]])

            F[:, i] = analytical_roe_flux(U_L, U_R, gamma)

        for i in range(n_cells):
            U_cons_roe[:, i] = U_cons_roe[:, i] - dt / dx * (F[:, i+1] - F[:, i])

        t += dt

    rho_roe = U_cons_roe[0, :]
    u_roe = U_cons_roe[1, :] / rho_roe
    p_roe = (gamma - 1) * (U_cons_roe[2, :] - 0.5 * rho_roe * u_roe**2)

    # Compute errors
    rho_mae = np.mean(np.abs(rho_mlp - rho_roe))
    u_mae = np.mean(np.abs(u_mlp - u_roe))
    p_mae = np.mean(np.abs(p_mlp - p_roe))

    return rho_mae, u_mae, p_mae


def analytical_roe_flux(U_L, U_R, gamma):
    """Compute Roe flux analytically."""
    rho_L, u_L, p_L = U_L
    rho_R, u_R, p_R = U_R

    H_L = gamma * p_L / ((gamma - 1) * rho_L) + 0.5 * u_L**2
    H_R = gamma * p_R / ((gamma - 1) * rho_R) + 0.5 * u_R**2

    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe**2))
    rho_roe = sqrt_rho_L * sqrt_rho_R

    lambda1 = u_roe - c_roe
    lambda2 = u_roe
    lambda3 = u_roe + c_roe

    drho = rho_R - rho_L
    du = u_R - u_L
    dp = p_R - p_L

    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    F_L = np.array([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])

    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
    F_R = np.array([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    R = np.array([
        [1, 1, 1],
        [u_roe - c_roe, u_roe, u_roe + c_roe],
        [H_roe - u_roe * c_roe, 0.5 * u_roe**2, H_roe + u_roe * c_roe]
    ])

    alpha = np.array([
        0.5 * (dp - rho_roe * c_roe * du) / c_roe**2,
        drho - dp / c_roe**2,
        0.5 * (dp + rho_roe * c_roe * du) / c_roe**2
    ])

    lambda_abs = np.array([abs(lambda1), abs(lambda2), abs(lambda3)])

    F_roe = 0.5 * (F_L + F_R)
    for k in range(3):
        F_roe -= 0.5 * alpha[k] * lambda_abs[k] * R[:, k]

    return F_roe


def main():
    parser = argparse.ArgumentParser(description='Compare model sizes')
    parser.add_argument('--data', type=str, default='data/multiphase_flux_data.npz')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--subset', type=int, default=2000000,
                        help='Use subset of data for faster training')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation evaluation (faster)')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("Model Size Comparison")
    print("=" * 70)
    print(f"Device: {args.device}")

    if args.device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        # Quick GPU test
        test_tensor = torch.randn(1000, 1000, device='cuda')
        _ = test_tensor @ test_tensor
        print("GPU test passed!", flush=True)
    print(f"Epochs: {args.epochs}")
    print(f"Data subset: {args.subset:,} samples")

    # Load data
    print("\nLoading data...")
    data = np.load(args.data)
    X = data['inputs'].astype(np.float32)
    Y = data['outputs'].astype(np.float32)

    # Use subset for faster comparison
    if args.subset and args.subset < len(X):
        idx = np.random.choice(len(X), args.subset, replace=False)
        X = X[idx]
        Y = Y[idx]

    print(f"Using {len(X):,} samples")

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)

    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    # Split
    n_train = int(0.9 * len(X))
    X_train, X_val = X_norm[:n_train], X_norm[n_train:]
    Y_train, Y_val = Y_norm[:n_train], Y_norm[n_train:]

    # Move all data to GPU upfront (no DataLoader - manual batching is faster)
    if args.device == 'cuda':
        print("Moving data to GPU...", flush=True)
        X_train_t = torch.from_numpy(X_train).cuda()
        Y_train_t = torch.from_numpy(Y_train).cuda()
        X_val_t = torch.from_numpy(X_val).cuda()
        Y_val_t = torch.from_numpy(Y_val).cuda()
        print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    else:
        X_train_t = torch.from_numpy(X_train)
        Y_train_t = torch.from_numpy(Y_train)
        X_val_t = torch.from_numpy(X_val)
        Y_val_t = torch.from_numpy(Y_val)

    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }

    # Architectures to compare
    architectures = [
        ('tiny_2x32', [32, 32]),
        ('small_3x64', [64, 64, 64]),
        ('medium_3x128', [128, 128, 128]),
        ('medium_4x128', [128, 128, 128, 128]),
        ('large_5x256', [256, 256, 256, 256, 256]),
    ]

    results = []

    print("\n" + "=" * 70)
    print("Training models...")
    print("=" * 70)

    for name, hidden_dims in architectures:
        print(f"\n{name}:", flush=True)
        model = FluxMLP(input_dim=7, output_dim=3, hidden_dims=hidden_dims, activation='gelu')
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}", flush=True)
        print(f"  Training...", flush=True)

        # Train
        start_time = time.time()
        best_val_loss, model = train_model(
            model, X_train_t, Y_train_t, X_val_t, Y_val_t,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device
        )
        train_time = time.time() - start_time

        print(f"  Best val_loss: {best_val_loss:.2e}")
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on simulation (optional - slow on CPU)
        if not args.skip_sim:
            rho_mae, u_mae, p_mae = evaluate_simulation(model, stats, args.device)
            print(f"  Simulation - rho_MAE: {rho_mae:.2e}, u_MAE: {u_mae:.2e}, p_MAE: {p_mae:.2e}")
        else:
            rho_mae, u_mae, p_mae = 0.0, 0.0, 0.0

        # Save model
        output_path = os.path.join(args.output_dir, f'flux_mlp_{name}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim': 7,
                'output_dim': 3,
                'hidden_dims': hidden_dims,
                'activation': 'gelu'
            },
            'stats': stats
        }, output_path)

        results.append({
            'name': name,
            'hidden_dims': hidden_dims,
            'params': n_params,
            'val_loss': best_val_loss,
            'train_time': train_time,
            'rho_mae': rho_mae,
            'u_mae': u_mae,
            'p_mae': p_mae
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} | {'Params':>10} | {'Val Loss':>10} | {'Time':>8} | {'rho MAE':>10} | {'u MAE':>10} | {'p MAE':>10}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<15} | {r['params']:>10,} | {r['val_loss']:>10.2e} | {r['train_time']:>7.1f}s | {r['rho_mae']:>10.2e} | {r['u_mae']:>10.2e} | {r['p_mae']:>10.2e}")

    # Find smallest model with acceptable accuracy
    large_rho = results[-1]['rho_mae']  # Use large model as baseline
    print(f"\nBaseline (large model) rho_MAE: {large_rho:.2e}")
    print("\nModels within 10x of baseline accuracy:")
    for r in results:
        if r['rho_mae'] < large_rho * 10:
            speedup = results[-1]['params'] / r['params']
            print(f"  {r['name']}: {r['params']:,} params ({speedup:.1f}x smaller)")


if __name__ == '__main__':
    main()
