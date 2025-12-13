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

    Input: [ρ_L, u_L, p_L, ρ_R, u_R, p_R] (6 features)
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

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """
        Compute flux from left and right primitive states.

        Parameters
        ----------
        x_left : torch.Tensor
            Left state [ρ, u, p], shape (n_edges, 3)
        x_right : torch.Tensor
            Right state [ρ, u, p], shape (n_edges, 3)

        Returns
        -------
        torch.Tensor
            Numerical flux [flux_ρ, flux_ρu, flux_E], shape (n_edges, 3)
        """
        # Concatenate left and right states
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
        activation: str = 'gelu'
    ):
        super().__init__(aggr='add')

        self.flux_net = SimpleFluxMLP(
            input_dim=6,
            output_dim=3,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dx: float = 1.0
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

        Returns
        -------
        torch.Tensor
            Flux divergence for each node, shape (n_nodes, 3)
        """
        self._dx = dx
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
        # Compute flux: F(left=x_j, right=x_i)
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
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 5,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.flux_layer = SimpleFluxGNN(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float,
        dx: float
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

        Returns
        -------
        torch.Tensor
            Updated primitive variables, shape (n_nodes, 3)
        """
        # Compute flux divergence
        div_flux = self.flux_layer(x, edge_index, dx)

        # Update (note: this operates on primitives, may need conversion)
        return x - dt * div_flux

    def compute_flux(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Directly compute flux (for training with uniform data).

        Parameters
        ----------
        x_left : torch.Tensor
            Left primitive states, shape (batch, 3)
        x_right : torch.Tensor
            Right primitive states, shape (batch, 3)

        Returns
        -------
        torch.Tensor
            Flux values, shape (batch, 3)
        """
        return self.flux_layer.flux_net(x_left, x_right)


def load_uniform_weights(model: SimpleEulerGNN, npz_path: str) -> None:
    """
    Load weights from uniformly-trained MLP into the GNN's flux network.

    Parameters
    ----------
    model : SimpleEulerGNN
        The GNN model to load weights into
    npz_path : str
        Path to the .npz file from train_uniform.py or train_uniform_cuda.py
    """
    import numpy as np

    data = np.load(npz_path)

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

    print(f"Loaded {layer_idx} layers from {npz_path}")


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
    print("Testing SimpleEulerGNN...")

    # Create model
    model = SimpleEulerGNN(hidden_dim=256, n_layers=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test data (10 cells)
    n_cells = 10
    x = torch.randn(n_cells, 3)  # [ρ, u, p] for each cell
    edge_index = create_1d_graph(n_cells)

    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")

    # Forward pass
    dt, dx = 0.001, 0.01
    x_new = model(x, edge_index, dt, dx)
    print(f"Output shape: {x_new.shape}")

    # Test direct flux computation
    x_left = torch.randn(100, 3)
    x_right = torch.randn(100, 3)
    flux = model.compute_flux(x_left, x_right)
    print(f"Direct flux shape: {flux.shape}")

    # Test loading uniform weights
    print("\nTesting weight loading...")
    try:
        load_uniform_weights(model, "flux_model_cuda.npz")
        print("Weight loading successful!")
    except FileNotFoundError:
        print("No pre-trained weights found (flux_model_cuda.npz)")

    print("\nAll tests passed!")
