"""
PyTorch CUDA training for 2D flux MLP.

Architecture:
- Input: 20 features (5 cells × 4 primitives: W, C, E, S, N)
- Output: 16 fluxes (4 faces × 4 flux components: F_w, F_e, G_s, G_n)
- Hidden: 256×5 or configurable

Based on the 1D training script with adaptations for 2D.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
import os


class FluxMLP2D(nn.Module):
    """MLP for predicting 2D Roe fluxes from 5-cell stencil."""

    def __init__(self, hidden_dim: int = 256, n_layers: int = 5):
        super().__init__()

        layers = []
        # Input layer: 20 features (5 cells × 4 primitives)
        layers.append(nn.Linear(20, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer: 16 fluxes (4 faces × 4 components)
        layers.append(nn.Linear(hidden_dim, 16))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_data(data_path: str, device: torch.device):
    """Load and normalize training data."""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    inputs = data['inputs']
    outputs = data['outputs']

    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")

    # Convert to tensors
    X = torch.tensor(inputs, dtype=torch.float32, device=device)
    Y = torch.tensor(outputs, dtype=torch.float32, device=device)

    # Normalize inputs
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std

    # Normalize outputs
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_norm = (Y - Y_mean) / Y_std

    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }

    return X_norm, Y_norm, stats


def train(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    hidden_dim: int = 256,
    n_layers: int = 5,
    epochs: int = 500,
    batch_size: int = 16384,
    lr: float = 1e-3,
    device: torch.device = None
):
    """Train the 2D flux MLP on GPU."""
    n_train = X_train.shape[0]
    print(f"\nTraining on {n_train:,} samples")
    print(f"Architecture: 20 -> {hidden_dim}x{n_layers} -> 16")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Create model
    model = FluxMLP2D(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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

        # Shuffle indices on GPU
        perm = torch.randperm(n_train, device=device)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        for i in range(0, n_train, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)

        # Progress report
        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_current = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:4d}/{epochs}: train_loss={avg_loss:.6f}, "
                  f"lr={lr_current:.2e}, time={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/epochs:.3f}s/epoch)")

    return model, history


def evaluate_model(model, X, Y, Y_std, device):
    """Evaluate model and compute per-variable errors."""
    model.eval()
    with torch.no_grad():
        Y_pred = model(X)
        mse = ((Y_pred - Y) ** 2).mean().item()

        # Denormalize for absolute errors
        Y_pred_denorm = Y_pred * Y_std
        Y_denorm = Y * Y_std

        # Per-flux-component errors
        mae = (Y_pred_denorm - Y_denorm).abs().mean(dim=0)
        rel_error = mae / Y_std.abs()

    face_names = ['F_w', 'F_e', 'G_s', 'G_n']
    flux_names = ['mass', 'mom_x', 'mom_y', 'energy']

    print(f"\nPer-component errors (MAE / std = relative):")
    for f, face in enumerate(face_names):
        print(f"  {face}:")
        for fl, fname in enumerate(flux_names):
            idx = f * 4 + fl
            print(f"    {fname}: MAE={mae[idx].item():.3e}, "
                  f"rel={100*rel_error[idx].item():.2f}%")

    avg_rel_error = rel_error.mean().item()
    print(f"\n  Average relative error: {100*avg_rel_error:.2f}%")

    return mse, avg_rel_error


def save_model(model, stats, history, save_path: str):
    """Save model weights and normalization stats."""
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': {k: v.cpu().numpy() for k, v in stats.items()},
        'history': history
    }, save_path.replace('.npz', '.pt'))
    print(f"Saved PyTorch model to {save_path.replace('.npz', '.pt')}")

    # Save NumPy format for simulation
    weights = {}
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            weights[f'W{i}'] = layer.weight.data.cpu().numpy()
            weights[f'b{i}'] = layer.bias.data.cpu().numpy()

    np.savez(
        save_path,
        **weights,
        X_mean=stats['X_mean'].cpu().numpy(),
        X_std=stats['X_std'].cpu().numpy(),
        Y_mean=stats['Y_mean'].cpu().numpy(),
        Y_std=stats['Y_std'].cpu().numpy()
    )
    print(f"Saved NumPy weights to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train 2D flux MLP with CUDA')
    parser.add_argument('--data', type=str, default='data/uniform_flux_data_2d.npz',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16384,
                        help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--n-layers', type=int, default=5,
                        help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='flux_model_2d_cuda.npz',
                        help='Output model path')

    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # Load data
    X, Y, stats = load_data(args.data, device)

    # Train on all data (no validation split for uniform sampling)
    model, history = train(
        X, Y,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # Evaluate
    evaluate_model(model, X, Y, stats['Y_std'], device)

    # Save model
    save_model(model, stats, history, args.output)


if __name__ == '__main__':
    main()
