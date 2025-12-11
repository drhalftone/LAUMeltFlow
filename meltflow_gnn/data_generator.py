"""
Training data generator for the GNN flux learner.

Runs meltflow simulations and extracts (state, flux) pairs at each timestep
for supervised learning of the numerical flux function.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

# Add meltflow to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meltflow.functions.parameters import create_parameters, Parameters
from meltflow.functions.grid import grid_setup
from meltflow.functions.state_var import state_var
from meltflow.functions.ghost_fluid import ghost_GFM, real_GFM, extrp_vel
from meltflow.functions.level_set import advc
from meltflow.functions.solver import timestep
from meltflow.functions.flux import roe_flux1D
from meltflow.input.configs import load_config

from .graph import create_1d_graph


def prim_to_cons_1d(U: np.ndarray, gam: float) -> np.ndarray:
    """
    Convert primitive variables to conserved variables for 1D Euler.

    Parameters
    ----------
    U : np.ndarray
        Primitive variables [rho, u, p]
    gam : float
        Specific heat ratio

    Returns
    -------
    np.ndarray
        Conserved variables [rho, rho*u, E]
    """
    rho = U[0]
    u = U[1]
    p = U[2]
    E = p / (gam - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])


def compute_roe_flux_1d(
    prm: Parameters,
    U: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Compute Roe flux at all interfaces for 1D grid.

    Parameters
    ----------
    prm : Parameters
        Simulation parameters
    U : np.ndarray
        Primitive variables [rho, u, p], shape (n_var, n)
    phi : np.ndarray
        Level set function, shape (n,)

    Returns
    -------
    np.ndarray
        Numerical flux at interfaces, shape (n_var, n-1)
    """
    n = U.shape[1]
    n_var = U.shape[0]
    n_dim = prm.n_dim
    gamma = prm.c_EoS

    flux = np.zeros((n_var, n - 1))

    for i in range(n - 1):
        # Left and right primitive states
        U_L = U[:, i]
        U_R = U[:, i + 1]

        # Determine which fluid (based on phi at interface)
        phi_interface = 0.5 * (phi[i] + phi[i + 1])
        if phi_interface > 0:
            gam = gamma[0]
        else:
            gam = gamma[1]

        # Convert to conserved variables
        W_L = prim_to_cons_1d(U_L, gam)
        W_R = prim_to_cons_1d(U_R, gam)

        # Compute Roe flux
        flux[:, i] = roe_flux1D(n_dim, gam, W_L, W_R)

    return flux


def generate_trajectory(
    config_name: str = 'in_1Dsod1fl',
    n_steps: Optional[int] = None,
    save_interval: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate a simulation trajectory with state and flux data.

    Parameters
    ----------
    config_name : str
        Name of the configuration to run
    n_steps : int, optional
        Number of timesteps to run (default: run to t_f)
    save_interval : int
        Save data every n timesteps

    Returns
    -------
    List[Dict]
        List of dictionaries with keys:
        - 'x': Grid coordinates
        - 'U': Primitive variables
        - 'phi': Level set
        - 'flux': Numerical flux at interfaces
        - 't': Current time
        - 'dt': Timestep used
    """
    # Load configuration
    config, init_func = load_config(config_name)

    # Grid setup
    n_var, n, x, dx, U, phi = grid_setup(config)
    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx

    # Create parameters
    from meltflow.functions.parameters import apply_defaults
    config = apply_defaults(config, config['n_dim'])
    prm = create_parameters(config)
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Apply initial conditions
    U, phi = init_func(x, U, phi)

    # Initialize
    t = 0.0
    it = 0
    trajectory = []

    # Calculate conserved variables
    W = state_var(prm, "cons", n_var, phi, U)

    # Use fixed dt to match comparison script
    dt_fixed = 0.1336323e-05

    while True:
        # Use fixed timestep for consistency
        dt = dt_fixed
        if t + dt > prm.t_f:
            dt = prm.t_f - t

        # Compute and save flux before update
        if it % save_interval == 0:
            flux = compute_roe_flux_1d(prm, U, phi)
            trajectory.append({
                'x': x.copy(),
                'U': U.copy(),
                'phi': phi.copy(),
                'flux': flux.copy(),
                't': t,
                'dt': dt
            })

        # Check termination
        if n_steps is not None and it >= n_steps:
            break
        if t >= prm.t_f:
            break

        t += dt

        # Ghost Fluid Method
        UU, WW = ghost_GFM(prm, x, phi, U)

        # Run solvers (simplified - just use Roe flux directly for 1D)
        from meltflow.functions.solver import run_slvr
        WW = run_slvr(prm, dt, x, phi, UU, WW)

        # Extrapolate velocity field
        V = extrp_vel(prm, x, phi, WW)

        # Advect level set
        phi = advc(prm, dt, V, phi)

        # Reassemble real domain
        W = real_GFM(prm, phi, WW)

        # Compute primitive variables
        U = state_var(prm, "prim", n_var, phi, W)

        it += 1

    return trajectory


def trajectory_to_graphs(
    trajectory: List[Dict[str, Any]],
    include_next_state: bool = False
) -> List[Data]:
    """
    Convert a trajectory to a list of PyTorch Geometric graphs.

    Parameters
    ----------
    trajectory : List[Dict]
        Output from generate_trajectory()
    include_next_state : bool
        If True, include next state's conserved variables for conservation loss

    Returns
    -------
    List[Data]
        List of graph Data objects
    """
    graphs = []
    for i, step in enumerate(trajectory):
        graph = create_1d_graph(step['x'], step['U'], step['phi'])
        # Add flux as target
        graph.flux = torch.tensor(step['flux'].T, dtype=torch.float32)
        graph.t = step['t']
        graph.dt = step['dt']

        # Add next state for conservation loss computation
        if include_next_state and i < len(trajectory) - 1:
            next_step = trajectory[i + 1]
            # Store next state's primitive variables
            graph.U_next = torch.tensor(next_step['U'].T, dtype=torch.float32)
            graph.has_next = True
        else:
            graph.has_next = False

        graphs.append(graph)
    return graphs


def generate_training_dataset(
    config_names: List[str] = ['in_1Dsod1fl'],
    n_trajectories_per_config: int = 1,
    perturbation_scale: float = 0.0,
    save_interval: int = 10,
    include_next_state: bool = False
) -> List[Data]:
    """
    Generate a training dataset from multiple configurations.

    Parameters
    ----------
    config_names : List[str]
        List of configuration names to use
    n_trajectories_per_config : int
        Number of trajectories per configuration
    perturbation_scale : float
        Scale of random perturbations to initial conditions
    save_interval : int
        Save data every n timesteps
    include_next_state : bool
        If True, include next state for conservation loss computation

    Returns
    -------
    List[Data]
        Combined list of graph Data objects
    """
    all_graphs = []

    for config_name in config_names:
        for traj_idx in range(n_trajectories_per_config):
            print(f"Generating trajectory {traj_idx+1}/{n_trajectories_per_config} "
                  f"for {config_name}...")

            trajectory = generate_trajectory(
                config_name=config_name,
                save_interval=save_interval
            )
            graphs = trajectory_to_graphs(trajectory, include_next_state=include_next_state)
            all_graphs.extend(graphs)

            print(f"  Generated {len(graphs)} samples")

    print(f"Total samples: {len(all_graphs)}")
    return all_graphs


class FluxDataset(Dataset):
    """
    PyTorch Geometric Dataset for flux learning.
    """

    def __init__(
        self,
        graphs: List[Data],
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.graphs = graphs

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]
