"""
Simplified GNN model for learning numerical flux.

This is a stripped-down version that:
1. Uses only primitive variables (ρ, u, p) as node features
2. Removes edge attributes (assumes uniform mesh)
3. Removes antisymmetric constraint (degraded performance)
4. Keeps message-passing structure for generalization potential

The FluxMLP is now compatible with the uniform sampling training data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional


class SimpleFluxMLP(nn.Module):
    """
    Simplified MLP for numerical flux prediction.

    Input: [ρ_L, u_L, p_L, ρ_R, u_R, p_R] (6 features) or
           [ρ_L, u_L, p_L, ρ_R, u_R, p_R, γ] (7 features with gamma)
    Output: [flux_ρ, flux_ρu, flux_E] (3 fluxes)

    This is compatible with the uniform sampling training data.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 5,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_gamma = (input_dim == 7)

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor,
                gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute flux from left and right primitive states.

        Parameters
        ----------
        x_left : torch.Tensor
            Left state [ρ, u, p], shape (n_edges, 3)
        x_right : torch.Tensor
            Right state [ρ, u, p], shape (n_edges, 3)
        gamma : torch.Tensor, optional
            Specific heat ratio, shape (n_edges,) or scalar.
            Required if model was trained with gamma as input (input_dim=7).

        Returns
        -------
        torch.Tensor
            Numerical flux [flux_ρ, flux_ρu, flux_E], shape (n_edges, 3)
        """
        if self.has_gamma:
            if gamma is None:
                raise ValueError("Model requires gamma input (input_dim=7)")
            # Handle scalar gamma
            if gamma.dim() == 0:
                gamma = gamma.expand(x_left.shape[0])
            elif gamma.dim() == 1 and gamma.shape[0] == 1:
                gamma = gamma.expand(x_left.shape[0])
            # Concatenate left, right, and gamma
            inputs = torch.cat([x_left, x_right, gamma.unsqueeze(-1)], dim=-1)
        else:
            # Concatenate left and right states only
            inputs = torch.cat([x_left, x_right], dim=-1)
        return self.mlp(inputs)


class SimpleFluxGNN(MessagePassing):
    """
    Simplified GNN layer for flux computation.

    Message passing:
    1. message(): Compute numerical flux at each interface using SimpleFluxMLP
    2. aggregate(): Sum incoming fluxes
    3. The result is the flux divergence for each cell
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 5,
        activation: str = 'gelu',
        input_dim: int = 6
    ):
        super().__init__(aggr='add')

        self.flux_net = SimpleFluxMLP(
            input_dim=input_dim,
            output_dim=3,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation
        )
        self.has_gamma = (input_dim == 7)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dx: float = 1.0,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flux divergence for each node.

        Parameters
        ----------
        x : torch.Tensor
            Node features [ρ, u, p], shape (n_nodes, 3)
        edge_index : torch.Tensor
            Edge connectivity, shape (2, n_edges)
        dx : float
            Grid spacing (uniform)
        gamma : torch.Tensor, optional
            Specific heat ratio. Required if model has gamma input.

        Returns
        -------
        torch.Tensor
            Flux divergence for each node, shape (n_nodes, 3)
        """
        self._dx = dx
        self._gamma = gamma
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Compute message (numerical flux) for each edge.

        In GNN convention:
        - x_i: target node (receives message)
        - x_j: source node (sends message)

        For flux computation:
        - We treat x_j as "left" and x_i as "right" for edges j -> i
        - The flux is scaled by 1/dx for finite volume update
        """
        # Compute flux: F(left=x_j, right=x_i, gamma)
        if self.has_gamma:
            # Expand gamma to match number of edges
            n_edges = x_i.shape[0]
            if self._gamma.dim() == 0:
                gamma_expanded = self._gamma.expand(n_edges)
            else:
                gamma_expanded = self._gamma.expand(n_edges)
            flux = self.flux_net(x_j, x_i, gamma_expanded)
        else:
            flux = self.flux_net(x_j, x_i)

        # Scale by 1/dx for flux divergence
        return flux / self._dx


class SimpleEulerGNN(nn.Module):
    """
    Simplified GNN for solving 1D Euler equations.

    Predicts state update:
        U^{n+1} = U^n - dt * div(F)

    where div(F) is computed by message passing.

    Key simplifications:
    - Only primitive variables (ρ, u, p) as node features
    - No edge attributes (uniform mesh assumed)
    - No antisymmetric constraint
    - Single message passing layer

    Supports gamma-parameterized flux models (input_dim=7) for different gases.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 5,
        activation: str = 'gelu',
        input_dim: int = 6
    ):
        super().__init__()

        self.flux_layer = SimpleFluxGNN(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            input_dim=input_dim
        )
        self.has_gamma = (input_dim == 7)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float,
        dx: float,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict state at next timestep.

        Parameters
        ----------
        x : torch.Tensor
            Primitive variables [ρ, u, p], shape (n_nodes, 3)
        edge_index : torch.Tensor
            Edge connectivity, shape (2, n_edges)
        dt : float
            Timestep
        dx : float
            Grid spacing
        gamma : torch.Tensor, optional
            Specific heat ratio. Required if model has gamma input.

        Returns
        -------
        torch.Tensor
            Updated primitive variables, shape (n_nodes, 3)
        """
        # Compute flux divergence
        div_flux = self.flux_layer(x, edge_index, dx, gamma)

        # Update (note: this operates on primitives, may need conversion)
        return x - dt * div_flux

    def compute_flux(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Directly compute flux (for training with uniform data).

        Parameters
        ----------
        x_left : torch.Tensor
            Left primitive states, shape (batch, 3)
        x_right : torch.Tensor
            Right primitive states, shape (batch, 3)
        gamma : torch.Tensor, optional
            Specific heat ratio, shape (batch,) or scalar.
            Required if model has gamma input.

        Returns
        -------
        torch.Tensor
            Flux values, shape (batch, 3)
        """
        return self.flux_layer.flux_net(x_left, x_right, gamma)


def load_uniform_weights(model: SimpleEulerGNN, path: str) -> dict:
    """
    Load weights from uniformly-trained MLP into the GNN's flux network.

    Supports both .npz files (from train_uniform.py) and .pt files (from train_uniform_cuda.py).

    Parameters
    ----------
    model : SimpleEulerGNN
        The GNN model to load weights into
    path : str
        Path to the .npz or .pt file from training

    Returns
    -------
    dict
        Normalization statistics {'X_mean', 'X_std', 'Y_mean', 'Y_std'}
    """
    import numpy as np

    if path.endswith('.pt'):
        # Load PyTorch checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Load state dict directly into flux MLP
        flux_mlp = model.flux_layer.flux_net.mlp

        # Map checkpoint keys to our model
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # Keys are like 'network.0.weight', 'network.0.bias', etc.
            new_key = key.replace('network.', '')
            new_state_dict[new_key] = value

        flux_mlp.load_state_dict(new_state_dict)

        # Extract stats
        stats = checkpoint['stats']
        config = checkpoint['config']

        print(f"Loaded PyTorch model from {path}")
        print(f"  Input dim: {config['input_dim']}, Output dim: {config['output_dim']}")
        print(f"  Has gamma: {config.get('has_gamma', False)}")

        return {
            'X_mean': stats['X_mean'],
            'X_std': stats['X_std'],
            'Y_mean': stats['Y_mean'],
            'Y_std': stats['Y_std']
        }

    else:
        # Load NumPy .npz file
        data = np.load(path)

        # Get the flux MLP
        flux_mlp = model.flux_layer.flux_net.mlp

        # Load weights layer by layer
        layer_idx = 0
        for i, module in enumerate(flux_mlp):
            if isinstance(module, nn.Linear):
                W = torch.from_numpy(data[f'W{layer_idx}']).float()
                b = torch.from_numpy(data[f'b{layer_idx}']).float()

                # NumPy MLP stores W as (in, out), PyTorch uses (out, in)
                module.weight.data = W.T
                module.bias.data = b

                layer_idx += 1

        print(f"Loaded {layer_idx} layers from {path}")

        # Extract stats
        return {
            'X_mean': data['stats_X_mean'],
            'X_std': data['stats_X_std'],
            'Y_mean': data['stats_Y_mean'],
            'Y_std': data['stats_Y_std']
        }


def create_1d_graph(n_cells: int) -> torch.Tensor:
    """
    Create edge_index for a 1D path graph.

    For finite volume method on 1D grid:
    - Each interior cell i has edges to neighbors i-1 and i+1
    - Boundary cells have one neighbor

    Parameters
    ----------
    n_cells : int
        Number of cells in the grid

    Returns
    -------
    torch.Tensor
        Edge index, shape (2, n_edges)
    """
    edges = []

    # Interior edges: i -> i+1 and i+1 -> i
    for i in range(n_cells - 1):
        edges.append([i, i + 1])  # left to right
        edges.append([i + 1, i])  # right to left

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return edge_index


if __name__ == "__main__":
    # Test the simplified model
    print("=" * 60)
    print("Testing SimpleEulerGNN")
    print("=" * 60)

    # Test 1: Standard model (6 inputs, no gamma)
    print("\n1. Testing standard model (input_dim=6)...")
    model = SimpleEulerGNN(hidden_dim=256, n_layers=5, input_dim=6)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    n_cells = 10
    x = torch.randn(n_cells, 3)
    edge_index = create_1d_graph(n_cells)

    dt, dx = 0.001, 0.01
    x_new = model(x, edge_index, dt, dx)
    print(f"   Forward pass: {x.shape} -> {x_new.shape}")

    x_left = torch.randn(100, 3)
    x_right = torch.randn(100, 3)
    flux = model.compute_flux(x_left, x_right)
    print(f"   Direct flux: {flux.shape}")

    # Test 2: Gamma-parameterized model (7 inputs)
    print("\n2. Testing gamma model (input_dim=7)...")
    model_gamma = SimpleEulerGNN(hidden_dim=256, n_layers=5, input_dim=7)
    print(f"   Parameters: {sum(p.numel() for p in model_gamma.parameters()):,}")
    print(f"   Has gamma: {model_gamma.has_gamma}")

    gamma = torch.tensor(1.4)
    x_new_gamma = model_gamma(x, edge_index, dt, dx, gamma)
    print(f"   Forward pass with gamma: {x.shape} -> {x_new_gamma.shape}")

    flux_gamma = model_gamma.compute_flux(x_left, x_right, gamma)
    print(f"   Direct flux with gamma: {flux_gamma.shape}")

    # Test 3: Load pre-trained weights
    print("\n3. Testing weight loading...")

    # Try loading gamma model
    try:
        model_loaded = SimpleEulerGNN(hidden_dim=256, n_layers=5, input_dim=7)
        stats = load_uniform_weights(model_loaded, "models/flux_mlp_gamma.pt")
        print("   Loaded gamma model from models/flux_mlp_gamma.pt")

        # Test inference with loaded model
        gamma_test = torch.tensor(1.4)
        flux_loaded = model_loaded.compute_flux(x_left, x_right, gamma_test)
        print(f"   Inference test: flux shape = {flux_loaded.shape}")
        print(f"   Stats keys: {list(stats.keys())}")
    except FileNotFoundError:
        print("   No pre-trained gamma model found (models/flux_mlp_gamma.pt)")

    # Try loading standard model
    try:
        model_std = SimpleEulerGNN(hidden_dim=256, n_layers=5, input_dim=6)
        stats = load_uniform_weights(model_std, "flux_model_cuda.npz")
        print("   Loaded standard model from flux_model_cuda.npz")
    except FileNotFoundError:
        print("   No pre-trained standard model found (flux_model_cuda.npz)")

    print("\nAll tests passed!")
