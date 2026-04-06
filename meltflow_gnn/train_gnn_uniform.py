"""
Train the GNN FluxMLP on uniform sampled data.

This adapts the 1M uniform samples to the GNN's expected input format:
- Node features: [rho, rho*u, E, phi, x_coord] (5 features per node)
- Edge attributes: [dx, normal] (2 features per edge)

The uniform data has primitive variables [rho, u, p] which we convert
to conserved variables [rho, rho*u, E].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import sys
sys.path.insert(0, '.')

from meltflow_gnn.model import FluxMLP


def prim_to_cons(rho, u, p, gamma=1.4):
    """Convert primitive to conserved variables."""
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, rho * u, E


def load_uniform_data(data_path: str, device: torch.device, gamma: float = 1.4):
    """
    Load uniform sampled data and convert to GNN format.

    The uniform data has:
    - inputs: [rho_L, u_L, p_L, rho_R, u_R, p_R] (6 features)
    - outputs: [flux_rho, flux_rhou, flux_E] (3 features)

    GNN FluxMLP expects:
    - x_i, x_j: [rho, rho*u, E, phi, x_coord] (5 features each)
    - edge_attr: [dx, normal] (2 features)
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    inputs = data['inputs']  # Shape: (N, 6)
    outputs = data['outputs']  # Shape: (N, 3)

    n_samples = inputs.shape[0]
    print(f"  Loaded {n_samples:,} samples")

    # Extract primitive variables
    rho_L = inputs[:, 0]
    u_L = inputs[:, 1]
    p_L = inputs[:, 2]
    rho_R = inputs[:, 3]
    u_R = inputs[:, 4]
    p_R = inputs[:, 5]

    # Convert to conserved variables
    rho_L_cons, rhou_L, E_L = prim_to_cons(rho_L, u_L, p_L, gamma)
    rho_R_cons, rhou_R, E_R = prim_to_cons(rho_R, u_R, p_R, gamma)

    # Create node features: [rho, rho*u, E, phi, x_coord]
    # phi = 1.0 (single fluid), x_coord = 0.5 (dummy, centered)
    phi_L = np.ones(n_samples)
    phi_R = np.ones(n_samples)
    x_L = np.full(n_samples, 0.45)  # Left cell center
    x_R = np.full(n_samples, 0.55)  # Right cell center

    x_i = np.stack([rho_L_cons, rhou_L, E_L, phi_L, x_L], axis=1)  # Shape: (N, 5)
    x_j = np.stack([rho_R_cons, rhou_R, E_R, phi_R, x_R], axis=1)  # Shape: (N, 5)

    # Create edge attributes: [dx, normal]
    dx = np.full(n_samples, 0.01)  # Grid spacing
    normal = np.ones(n_samples)  # Normal direction (+1 for right-facing)
    edge_attr = np.stack([dx, normal], axis=1)  # Shape: (N, 2)

    # Convert to tensors
    x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
    x_j = torch.tensor(x_j, dtype=torch.float32, device=device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    Y = torch.tensor(outputs, dtype=torch.float32, device=device)

    # Normalize outputs
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_norm = (Y - Y_mean) / Y_std

    stats = {
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }

    return x_i, x_j, edge_attr, Y_norm, stats


def plot_training_curve(history, save_path='gnn_training_curve.png'):
    """Save training curve plot."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.semilogy(history['train_loss'], 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('GNN FluxMLP Training Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'][-100:] if len(history['train_loss']) > 100 else history['train_loss'], 'b-', linewidth=1)
    plt.xlabel('Epoch (last 100)')
    plt.ylabel('Loss (MSE)')
    plt.title('Recent Training Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def train(
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    edge_attr: torch.Tensor,
    Y: torch.Tensor,
    hidden_dim: int = 256,
    n_layers: int = 5,
    epochs: int = 500,
    batch_size: int = 16384,
    lr: float = 1e-3,
    antisymmetric: bool = False,
    use_residual: bool = True,
    device: torch.device = None,
    plot_interval: int = 10
):
    """Train the GNN FluxMLP."""
    n_samples = x_i.shape[0]
    print(f"\nTraining GNN FluxMLP on {n_samples:,} samples")
    print(f"Architecture: 12 -> {hidden_dim}x{n_layers} -> 3")
    print(f"Antisymmetric: {antisymmetric}, Residual: {use_residual}")

    # Create model
    model = FluxMLP(
        n_var=3,
        n_edge_features=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        antisymmetric=antisymmetric,
        use_residual=use_residual
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    start_time = time.time()
    history = {'train_loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        x_i_shuf = x_i[perm]
        x_j_shuf = x_j[perm]
        edge_attr_shuf = edge_attr[perm]
        Y_shuf = Y[perm]

        for i in range(0, n_samples, batch_size):
            x_i_batch = x_i_shuf[i:i+batch_size]
            x_j_batch = x_j_shuf[i:i+batch_size]
            edge_attr_batch = edge_attr_shuf[i:i+batch_size]
            Y_batch = Y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            Y_pred = model(x_i_batch, x_j_batch, edge_attr_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)

        # Save plot periodically
        if (epoch + 1) % plot_interval == 0:
            plot_training_curve(history)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_current = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:4d}/{epochs}: loss={avg_loss:.6f}, "
                  f"lr={lr_current:.2e}, time={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/epochs:.3f}s/epoch)")

    # Final plot
    plot_training_curve(history)

    return model, history


def evaluate(model, x_i, x_j, edge_attr, Y, Y_std, device):
    """Evaluate model and compute errors."""
    model.eval()
    with torch.no_grad():
        Y_pred = model(x_i, x_j, edge_attr)
        mse = ((Y_pred - Y) ** 2).mean().item()

        # Denormalize
        Y_pred_denorm = Y_pred * Y_std
        Y_denorm = Y * Y_std

        mae = (Y_pred_denorm - Y_denorm).abs().mean(dim=0)
        rel_error = mae / Y_std.abs()

    print(f"\nPer-variable errors:")
    names = ['flux_rho', 'flux_rhou', 'flux_E']
    for i, name in enumerate(names):
        print(f"  {name}: MAE={mae[i].item():.2e}, rel={100*rel_error[i].item():.2f}%")

    avg_rel = rel_error.mean().item()
    print(f"  Average relative error: {100*avg_rel:.2f}%")

    return mse, avg_rel


def save_model(model, stats, history, save_path: str, config: dict):
    """Save model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': {k: v.cpu().numpy() for k, v in stats.items()},
        'history': history,
        'config': config
    }, save_path)
    print(f"Saved model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train GNN FluxMLP on uniform data')
    parser.add_argument('--data', type=str, default='data/uniform_flux_data.npz')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16384)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--antisymmetric', action='store_true')
    parser.add_argument('--no-residual', action='store_true')
    parser.add_argument('--output', type=str, default='models/gnn_flux_uniform.pt')

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load data
    x_i, x_j, edge_attr, Y, stats = load_uniform_data(args.data, device)

    # Train
    model, history = train(
        x_i, x_j, edge_attr, Y,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        antisymmetric=args.antisymmetric,
        use_residual=not args.no_residual,
        device=device
    )

    # Evaluate
    evaluate(model, x_i, x_j, edge_attr, Y, stats['Y_std'], device)

    # Save
    config = {
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'antisymmetric': args.antisymmetric,
        'use_residual': not args.no_residual
    }
    save_model(model, stats, history, args.output, config)


if __name__ == '__main__':
    main()
