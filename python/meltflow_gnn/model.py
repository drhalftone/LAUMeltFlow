"""
GNN model for learning numerical flux.

Architecture follows the message-passing framework:
1. Edge function: Compute flux from left/right states
2. Aggregation: Sum fluxes at each node
3. Node update: Update conserved variables

The key learning task is the edge function (numerical flux).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple


class FluxMLP(nn.Module):
    """
    MLP to learn the numerical flux function.

    Approximates: F* = f(U_left, U_right, edge_features)

    Input: Concatenation of left state, right state, and edge features
    Output: Numerical flux vector
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3
    ):
        """
        Parameters
        ----------
        n_var : int
            Number of conserved variables (3 for 1D Euler: rho, rho*u, E)
        n_edge_features : int
            Number of edge features (dx, normal)
        hidden_dim : int
            Hidden layer dimension
        n_layers : int
            Number of hidden layers
        """
        super().__init__()

        self.n_var = n_var

        # Input: [U_left (n_var+2), U_right (n_var+2), edge_attr (n_edge_features)]
        # +2 for phi and x_coord in node features
        input_dim = 2 * (n_var + 2) + n_edge_features

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output: flux for each conserved variable
        layers.append(nn.Linear(hidden_dim, n_var))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flux from neighboring states.

        Parameters
        ----------
        x_i : torch.Tensor
            Features of source nodes, shape (n_edges, n_features)
        x_j : torch.Tensor
            Features of target nodes, shape (n_edges, n_features)
        edge_attr : torch.Tensor
            Edge attributes, shape (n_edges, n_edge_features)

        Returns
        -------
        torch.Tensor
            Numerical flux, shape (n_edges, n_var)
        """
        # Concatenate left state, right state, and edge features
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(inputs)


class FluxGNN(MessagePassing):
    """
    GNN layer that mimics one timestep of the DG/FV method.

    Message passing:
    1. message(): Compute numerical flux at each interface
    2. aggregate(): Sum incoming fluxes (with sign from normal)
    3. update(): Compute dU/dt from flux divergence
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3
    ):
        super().__init__(aggr='add')  # Sum aggregation

        self.n_var = n_var
        self.flux_net = FluxMLP(n_var, n_edge_features, hidden_dim, n_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flux divergence for each node.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (n_nodes, n_features)
        edge_index : torch.Tensor
            Edge connectivity, shape (2, n_edges)
        edge_attr : torch.Tensor
            Edge attributes, shape (n_edges, n_edge_features)

        Returns
        -------
        torch.Tensor
            Flux divergence for each node, shape (n_nodes, n_var)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute message (numerical flux) for each edge.

        Parameters
        ----------
        x_i : torch.Tensor
            Features of source nodes
        x_j : torch.Tensor
            Features of target nodes
        edge_attr : torch.Tensor
            Edge attributes [dx, normal]

        Returns
        -------
        torch.Tensor
            Flux contribution, shape (n_edges, n_var)
        """
        # Compute numerical flux
        flux = self.flux_net(x_i, x_j, edge_attr)

        # Scale by normal direction (for proper flux differencing)
        # normal is the second edge attribute
        normal = edge_attr[:, 1:2]
        dx = edge_attr[:, 0:1]

        # Return flux * normal / dx (contribution to flux divergence)
        return flux * normal / dx


class EulerGNN(nn.Module):
    """
    Full GNN model for solving 1D Euler equations.

    Predicts the state at the next timestep:
        U^{n+1} = U^n - dt * div(F)

    where div(F) is computed by the FluxGNN layer.
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_message_passing: int = 1
    ):
        """
        Parameters
        ----------
        n_var : int
            Number of conserved variables
        n_edge_features : int
            Number of edge features
        hidden_dim : int
            Hidden layer dimension
        n_layers : int
            Number of hidden layers in flux MLP
        n_message_passing : int
            Number of message passing iterations
        """
        super().__init__()

        self.n_var = n_var
        self.n_message_passing = n_message_passing

        # Flux computation layers
        self.flux_layers = nn.ModuleList([
            FluxGNN(n_var, n_edge_features, hidden_dim, n_layers)
            for _ in range(n_message_passing)
        ])

    def forward(
        self,
        data: Data,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Predict state at next timestep.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object with node/edge features
        dt : float
            Timestep size

        Returns
        -------
        torch.Tensor
            Updated node features (only conserved variables), shape (n_nodes, n_var)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Extract current conserved variables
        U = x[:, :self.n_var]

        # Compute flux divergence through message passing
        for layer in self.flux_layers:
            div_flux = layer(x, edge_index, edge_attr)
            # Update conserved variables
            U = U - dt * div_flux

        return U

    def compute_flux(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Compute numerical flux at interfaces (for training/analysis).

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object

        Returns
        -------
        torch.Tensor
            Flux at each edge, shape (n_edges, n_var)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Get source and target node features
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        # Compute flux using the first layer's flux network
        return self.flux_layers[0].flux_net(x_i, x_j, edge_attr)
