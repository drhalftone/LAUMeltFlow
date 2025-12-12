"""
Training script for the flux MLP using uniform grid-sampled data.

Pure NumPy implementation - no PyTorch required.
Uses a simple MLP with GELU activation and optional residual connections.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import time
import os


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU activation."""
    # Approximate derivative
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(np.float64)


class FluxMLP:
    """
    Simple MLP to learn the Roe flux function.

    Input: [rho_L, u_L, p_L, rho_R, u_R, p_R] (6 features)
    Output: [flux_rho, flux_rhou, flux_E] (3 features)
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256, 256, 256],
        activation: str = 'relu'
    ):
        """
        Initialize MLP with He initialization.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        hidden_dims : List[int]
            List of hidden layer dimensions
        activation : str
            Activation function ('relu' or 'gelu')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = gelu
            self.activation_derivative = gelu_derivative

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # He initialization
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

        # For Adam optimizer
        self.m_weights = [np.zeros_like(W) for W in self.weights]
        self.v_weights = [np.zeros_like(W) for W in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.layer_inputs = [X]
        self.pre_activations = []

        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            self.pre_activations.append(z)

            # Apply activation to all but last layer
            if i < len(self.weights) - 1:
                h = self.activation(z)
            else:
                h = z  # Linear output

            self.layer_inputs.append(h)

        return h

    def backward(self, X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass - compute gradients."""
        batch_size = X.shape[0]

        # Output layer gradient (MSE loss derivative)
        delta = 2 * (Y_pred - Y) / batch_size

        grad_weights = []
        grad_biases = []

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            h = self.layer_inputs[i]
            grad_W = h.T @ delta
            grad_b = delta.sum(axis=0)

            grad_weights.insert(0, grad_W)
            grad_biases.insert(0, grad_b)

            # Backpropagate delta (skip for input layer)
            if i > 0:
                delta = delta @ self.weights[i].T
                # Apply activation derivative
                delta = delta * self.activation_derivative(self.pre_activations[i-1])

        return grad_weights, grad_biases

    def update_adam(
        self,
        grad_weights: List[np.ndarray],
        grad_biases: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update weights using Adam optimizer."""
        self.t += 1

        for i in range(len(self.weights)):
            # Update moments for weights
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * grad_weights[i]**2

            # Bias correction
            m_hat = self.m_weights[i] / (1 - beta1**self.t)
            v_hat = self.v_weights[i] / (1 - beta2**self.t)

            # Update weights
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Update moments for biases
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_biases[i]
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * grad_biases[i]**2

            # Bias correction
            m_hat = self.m_biases[i] / (1 - beta1**self.t)
            v_hat = self.v_biases[i] / (1 - beta2**self.t)

            # Update biases
            self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: str, stats: Dict = None, config: Dict = None):
        """Save model to file."""
        # Save weights and biases as separate arrays to avoid inhomogeneous array issue
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'W{i}'] = W
            save_dict[f'b{i}'] = b
        save_dict['n_layers'] = len(self.weights)

        # Save stats
        if stats:
            for key, val in stats.items():
                save_dict[f'stats_{key}'] = val

        # Save config
        if config:
            save_dict['config_input_dim'] = config.get('input_dim', 6)
            save_dict['config_output_dim'] = config.get('output_dim', 3)
            save_dict['config_hidden_dims'] = np.array(config.get('hidden_dims', [256]*5))
            save_dict['config_activation'] = config.get('activation', 'relu')

        np.savez(path, **save_dict)

    def save_with_history(self, path: str, stats: Dict = None, config: Dict = None, history: Dict = None):
        """Save model with training history."""
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'W{i}'] = W
            save_dict[f'b{i}'] = b
        save_dict['n_layers'] = len(self.weights)

        if stats:
            for key, val in stats.items():
                save_dict[f'stats_{key}'] = val

        if config:
            save_dict['config_input_dim'] = config.get('input_dim', 6)
            save_dict['config_output_dim'] = config.get('output_dim', 3)
            save_dict['config_hidden_dims'] = np.array(config.get('hidden_dims', [256]*5))
            save_dict['config_activation'] = config.get('activation', 'relu')

        if history:
            save_dict['history_train_loss'] = np.array(history.get('train_loss', []))
            save_dict['history_val_loss'] = np.array(history.get('val_loss', []))
            save_dict['history_epoch_time'] = np.array(history.get('epoch_time', []))

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> Tuple['FluxMLP', Dict, Dict]:
        """Load model from file."""
        data = np.load(path, allow_pickle=True)

        # Load config
        config = {
            'input_dim': int(data['config_input_dim']) if 'config_input_dim' in data else 6,
            'output_dim': int(data['config_output_dim']) if 'config_output_dim' in data else 3,
            'hidden_dims': list(data['config_hidden_dims']) if 'config_hidden_dims' in data else [256]*5,
            'activation': str(data['config_activation']) if 'config_activation' in data else 'relu'
        }

        # Load stats
        stats = {}
        for key in ['X_mean', 'X_std', 'Y_mean', 'Y_std']:
            if f'stats_{key}' in data:
                stats[key] = data[f'stats_{key}']

        # Create model
        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation']
        )

        # Load weights
        n_layers = int(data['n_layers'])
        model.weights = [data[f'W{i}'] for i in range(n_layers)]
        model.biases = [data[f'b{i}'] for i in range(n_layers)]

        return model, stats, config


def load_data(
    data_path: str,
    val_split: float = 0.1,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load uniform grid data.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, stats
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X = data['inputs'].astype(np.float64)
    Y = data['outputs'].astype(np.float64)

    print(f"  Loaded {len(X):,} samples")

    # Compute normalization stats
    stats = {}
    if normalize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-8] = 1.0

        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_std[Y_std < 1e-8] = 1.0

        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std

        stats = {
            'X_mean': X_mean,
            'X_std': X_std,
            'Y_mean': Y_mean,
            'Y_std': Y_std
        }

        print(f"  Normalized: X mean={X.mean():.4f}, std={X.std():.4f}")
        print(f"              Y mean={Y.mean():.4f}, std={Y.std():.4f}")

    # Shuffle and split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    return X_train, Y_train, X_val, Y_val, stats


def train(
    model: FluxMLP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """
    Train the model.

    Returns
    -------
    Dict
        Training history
    """
    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_time': []
    }

    best_val_loss = float('inf')
    best_weights = None
    best_biases = None

    print(f"\nTraining...", flush=True)
    print(f"  Samples: {n_train:,}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Batches/epoch: {n_batches}", flush=True)
    print(f"  Learning rate: {lr}", flush=True)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Shuffle training data
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Training
        train_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # Forward pass
            Y_pred = model.forward(X_batch)

            # Compute loss
            batch_loss = np.mean((Y_pred - Y_batch) ** 2)
            train_loss += batch_loss

            # Backward pass
            grad_W, grad_b = model.backward(X_batch, Y_batch, Y_pred)

            # Update weights
            model.update_adam(grad_W, grad_b, lr=lr)

        train_loss /= n_batches

        # Validation
        Y_val_pred = model.forward(X_val)
        val_loss = np.mean((Y_val_pred - Y_val) ** 2)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [W.copy() for W in model.weights]
            best_biases = [b.copy() for b in model.biases]

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"time={epoch_time:.2f}s", flush=True)

    # Restore best model
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases
        print(f"\nRestored best model with val_loss={best_val_loss:.6f}")

    return history


def evaluate(
    model: FluxMLP,
    X: np.ndarray,
    Y: np.ndarray,
    stats: Dict
) -> Dict:
    """Evaluate model and return metrics in original scale."""
    Y_pred = model.forward(X)

    # Convert back to original scale
    Y_pred_orig = Y_pred * stats['Y_std'] + stats['Y_mean']
    Y_orig = Y * stats['Y_std'] + stats['Y_mean']

    mse = np.mean((Y_pred_orig - Y_orig) ** 2)
    mae = np.mean(np.abs(Y_pred_orig - Y_orig))

    # Per-variable
    var_names = ['flux_rho', 'flux_rhou', 'flux_E']
    per_var_mae = {}
    for i, name in enumerate(var_names):
        per_var_mae[name] = np.mean(np.abs(Y_pred_orig[:, i] - Y_orig[:, i]))

    return {
        'mse': mse,
        'mae': mae,
        'per_var_mae': per_var_mae
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train flux MLP (NumPy)')
    parser.add_argument('--data', type=str, default='data/uniform_flux_data.npz',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=5,
                        help='Number of hidden layers')
    parser.add_argument('--output', type=str, default='flux_model_uniform.npz',
                        help='Output model path')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'], help='Activation function')

    args = parser.parse_args()

    print("=" * 60)
    print("Flux MLP Training (NumPy)")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    print("\n1. Loading data...")
    X_train, Y_train, X_val, Y_val, stats = load_data(
        args.data, val_split=0.1, normalize=True
    )

    # Create model
    print("\n2. Creating model...")
    hidden_dims = [args.hidden_dim] * args.n_layers
    model = FluxMLP(
        input_dim=6,
        output_dim=3,
        hidden_dims=hidden_dims,
        activation=args.activation
    )

    n_params = sum(W.size + b.size for W, b in zip(model.weights, model.biases))
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: 6 -> {' -> '.join(map(str, hidden_dims))} -> 3")
    print(f"  Activation: {args.activation}")

    # Train
    print("\n3. Training...")
    history = train(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True
    )

    # Evaluate
    print("\n4. Evaluating...")
    metrics = evaluate(model, X_val, Y_val, stats)

    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Per-variable MAE:")
    for name, mae in metrics['per_var_mae'].items():
        print(f"    {name}: {mae:.4f}")

    # Save model
    print(f"\n5. Saving model to {args.output}...")
    config = {
        'input_dim': 6,
        'output_dim': 3,
        'hidden_dims': hidden_dims,
        'activation': args.activation
    }
    model.save_with_history(args.output, stats=stats, config=config, history=history)

    print("\nDone!")

    return model, history, metrics, stats


if __name__ == '__main__':
    main()
