"""
MeltFlow GNN - Graph Neural Network for learning numerical flux.

This package implements a GNN-based approach to learn the numerical flux
function for the compressible Euler equations, following the Discontinuous
Galerkin framework.

Usage:
    python -m meltflow_gnn.train
"""

__version__ = "0.1.0"

from .graph import create_1d_graph, create_flux_training_data, graph_to_arrays
from .model import FluxMLP, FluxGNN, EulerGNN
from .data_generator import (
    generate_trajectory,
    trajectory_to_graphs,
    generate_training_dataset,
    FluxDataset
)
from .train import train_flux_model, evaluate_model
