"""
Training script for the air/water droplet flux MLP.

Configured specifically for the in_1Dcdrop case with high density ratios.

Usage:
    # Step 1: Generate training data (CPU only, ~2 minutes)
    python meltflow_gnn/grid_sampler_droplet.py --n-samples 2000000 --output data/droplet_flux_data.npz

    # Step 2: Train the model (requires GPU)
    python meltflow_gnn/train_droplet.py --data data/droplet_flux_data.npz --output models/droplet_flux_model.pt

    # Step 3: Test with simulation
    python meltflow_gnn/simulate_droplet_with_mlp.py --model models/droplet_flux_model.npz
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import time
import argparse


class FluxMLP(nn.Module):
    """
    MLP to learn the Roe flux function for high density ratio flows.

    Input: [rho_L, u_L, p_L, rho_R, u_R, p_R] (6 features)
    Output: [flux_rho, flux_rhou, flux_E] (3 features)

    Architecture tuned for the 1000:1 density ratio case:
    - Deeper network (6 layers) for complex flux patterns
    - GELU activation for smooth gradients
    - Wider layers (384) to capture high dynamic range
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = 'gelu',
    ):
        super().__init__()

        if hidden_dims is None:
            # Default architecture for droplet case: wider and deeper
            hidden_dims = [384, 384, 384, 384, 384, 384]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Activation function
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn)

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_data(
    data_path: str,
    val_split: float = 0.1,
    normalize: bool = True,
    log_normalize_density: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Load and prepare data for PyTorch training.

    For high density ratio cases, we optionally log-normalize density
    to handle the 1000:1 ratio better.
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data['inputs'].astype(np.float32)
    Y = data['outputs'].astype(np.float32)

    print(f"  Loaded {len(X):,} samples")
    print(f"  Input range - rho: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  Input range - u:   [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print(f"  Input range - p:   [{X[:, 2].min():.0f}, {X[:, 2].max():.0f}]")

    # Compute normalization stats
    stats = {}

    if normalize:
        X_orig = X.copy()

        # Optionally log-transform density for better numerical conditioning
        if log_normalize_density:
            print("  Using log-normalization for density columns")
            # Log-transform density columns (indices 0 and 3)
            X[:, 0] = np.log10(X[:, 0])
            X[:, 3] = np.log10(X[:, 3])
            stats['log_density'] = True
        else:
            stats['log_density'] = False

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-8] = 1.0

        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_std[Y_std < 1e-8] = 1.0

        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std

        stats['X_mean'] = X_mean
        stats['X_std'] = X_std
        stats['Y_mean'] = Y_mean
        stats['Y_std'] = Y_std

        print(f"  Normalized: X mean={X.mean():.4f}, std={X.std():.4f}")
        print(f"              Y mean={Y.mean():.4f}, std={Y.std():.4f}")

    stats['input_dim'] = X.shape[1]
    stats['output_dim'] = Y.shape[1]

    # Shuffle and split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    return X_train_t, Y_train_t, X_val_t, Y_val_t, stats


def train(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cuda',
    print_every: int = 10
) -> Dict:
    """Train the model with PyTorch."""

    model = model.to(device)

    # Move ALL data to GPU upfront
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch_time': []
    }

    best_val_loss = float('inf')
    best_state_dict = None

    print(f"\nTraining on {device}...")
    print(f"  Samples: {n_train:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches/epoch: {n_batches}")
    print(f"  Learning rate: {lr}")
    print()

    for epoch in range(n_epochs):
        epoch_start = time.time()
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

        # Validation
        model.eval()
        with torch.no_grad():
            Y_val_pred = model(X_val)
            val_loss = criterion(Y_val_pred, Y_val).item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"lr={current_lr:.2e}, time={epoch_time:.2f}s")

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nRestored best model with val_loss={best_val_loss:.6f}")

    return history


def save_model(model: nn.Module, path: str, stats: Dict, config: Dict, history: Dict = None):
    """Save model in PyTorch format."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'config': config,
    }
    if history:
        save_dict['history'] = history

    torch.save(save_dict, path)
    print(f"Saved model to {path}")


def save_numpy_compatible(model: nn.Module, path: str, stats: Dict, config: Dict, history: Dict = None):
    """Save model in NumPy-compatible format for inference without PyTorch."""

    save_dict = {}

    # Extract weights and biases
    state_dict = model.state_dict()
    layer_idx = 0

    for name, param in state_dict.items():
        if 'weight' in name:
            save_dict[f'W{layer_idx}'] = param.cpu().numpy().T
        elif 'bias' in name:
            save_dict[f'b{layer_idx}'] = param.cpu().numpy()
            layer_idx += 1

    save_dict['n_layers'] = layer_idx

    # Save stats
    for key, val in stats.items():
        if isinstance(val, np.ndarray):
            save_dict[f'stats_{key}'] = val
        else:
            save_dict[f'stats_{key}'] = np.array(val)

    # Save config
    save_dict['config_input_dim'] = config.get('input_dim', 6)
    save_dict['config_output_dim'] = config.get('output_dim', 3)
    save_dict['config_hidden_dims'] = np.array(config.get('hidden_dims', [384]*6))
    save_dict['config_activation'] = config.get('activation', 'gelu')
    save_dict['config_log_density'] = config.get('log_density', True)

    if history:
        save_dict['history_train_loss'] = np.array(history.get('train_loss', []))
        save_dict['history_val_loss'] = np.array(history.get('val_loss', []))

    np.savez(path, **save_dict)
    print(f"Saved NumPy-compatible model to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train flux MLP for air/water droplet')
    parser.add_argument('--data', type=str, default='data/droplet_flux_data.npz',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--hidden-dim', type=int, default=384,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu', 'silu'], help='Activation function')
    parser.add_argument('--no-log-density', action='store_true',
                        help='Disable log-normalization of density')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print progress every N epochs')
    parser.add_argument('--output', type=str, default='models/droplet_flux_model.pt',
                        help='Output model path (.pt for PyTorch)')
    parser.add_argument('--output-npz', type=str, default='models/droplet_flux_model.npz',
                        help='Output NumPy-compatible model path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("Air/Water Droplet Flux MLP Training")
    print("=" * 70)
    print()
    print("This model learns the Roe flux for high density ratio (1000:1) flows.")
    print("Parameter ranges: rho=[0.1, 1200] kg/m³, u=[-20, 120] m/s, p=[48k, 166k] Pa")
    print()

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Auto-generate training data if it doesn't exist
    if not os.path.exists(args.data):
        print(f"\nTraining data not found at {args.data}")
        print("Generating training data (2M samples, this may take a minute)...")
        print()

        # Import and run the data generator
        from grid_sampler_droplet import generate_droplet_samples_vectorized, get_droplet_parameter_ranges

        # Generate data
        inputs, outputs = generate_droplet_samples_vectorized(
            n_samples=2000000,
            gamma=1.4,
            log_sampling=True,
        )

        # Save
        data_dir = os.path.dirname(args.data)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

        ranges = get_droplet_parameter_ranges()
        np.savez(args.data,
                 inputs=inputs,
                 outputs=outputs,
                 ranges=ranges,
                 gamma=1.4,
                 config='in_1Dcdrop')

        print(f"\nSaved training data to {args.data}")
        print(f"  Samples: {len(inputs):,}")
        print()

    # Load data
    print("\n1. Loading data...")
    X_train, Y_train, X_val, Y_val, stats = load_data(
        args.data,
        val_split=args.val_split,
        normalize=True,
        log_normalize_density=not args.no_log_density
    )

    # Create model
    print("\n2. Creating model...")
    hidden_dims = [args.hidden_dim] * args.n_layers
    input_dim = stats['input_dim']
    output_dim = stats['output_dim']

    model = FluxMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=args.activation
    )

    print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Activation: {args.activation}")
    print(f"  Log-density normalization: {not args.no_log_density}")

    # Train
    print("\n3. Training...")
    start_time = time.time()

    history = train(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        print_every=args.print_every
    )

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average epoch time: {np.mean(history['epoch_time']):.2f}s")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")

    # Save model
    print("\n4. Saving model...")
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': args.activation,
        'log_density': not args.no_log_density,
        'case': 'in_1Dcdrop',
        'description': 'Air/water droplet flux model (1000:1 density ratio)'
    }

    save_model(model, args.output, stats, config, history)
    save_numpy_compatible(model, args.output_npz, stats, config, history)

    print("\nDone!")
    print("\nTo use this model, run:")
    print(f"  python meltflow_gnn/simulate_droplet_with_mlp.py --model {args.output_npz}")


if __name__ == '__main__':
    main()
