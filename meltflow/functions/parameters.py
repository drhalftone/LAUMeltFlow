"""
Parameter handling for MeltFlow solver.

This module defines the Parameters dataclass that holds all simulation parameters,
replacing the MATLAB cell array 'prm'.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class Parameters:
    """
    Container for all simulation parameters.

    This replaces the MATLAB prm cell array with a more Pythonic approach.
    """
    # Grid parameters
    n_dim: int = 1                          # Number of spatial dimensions
    n_var: int = 3                          # Number of primitive/conserved variables
    n: Union[int, np.ndarray] = 100         # Number of grid points
    dx: Union[float, np.ndarray] = 0.01     # Grid spacing
    x_min: Union[float, np.ndarray] = 0.0   # Grid minimum boundary
    x_max: Union[float, np.ndarray] = 1.0   # Grid maximum boundary

    # Time parameters
    t_f: float = 1.0                        # Final simulation time [s]
    cfl: float = 0.9                        # CFL number
    t_0: float = 0.0                        # Initial time

    # Fluid parameters
    flg_fld: np.ndarray = field(default_factory=lambda: np.array([0, 1]))  # 0=gas, 1=liquid
    EoS: List[str] = field(default_factory=lambda: ["perfect", "none"])    # Equation of state
    c_EoS: List[float] = field(default_factory=lambda: [1.4, 1.0])         # EoS parameters (gamma)
    slvr: List[str] = field(default_factory=lambda: ["roe_perfect", "none"])  # Solver selection

    # Boundary conditions
    flg_BCs: np.ndarray = field(default_factory=lambda: np.array([1, 1]))  # BC flags

    # Parallel processing
    n_nds: int = 0                          # Number of parallel nodes

    # Display/output
    ICs_hdr: str = "%------------------ (Simulation Description Unspecified) ----------------%"
    n_disp: int = 10                        # Display interval
    n_out: Union[int, np.ndarray] = 51      # Output grid points
    flg_intrp: bool = False                 # Interpolation flag

    # File I/O
    flg_wrt: bool = False                   # Write flag
    wrt_nm: str = "flow_unnamed"            # Write filename
    wrt_prfx: str = "data/"                 # Write prefix
    wrt_sfx: str = ".d"                     # Write suffix

    # Restart
    n_rstrt: int = 0                        # Restart interval
    rd_nm: str = "flow_unnamed"             # Read filename
    rd_prfx: str = "data/"                  # Read prefix
    rd_sfx: str = ".flo"                    # Read suffix

    # Plotting
    flg_plt: bool = False                   # Plot flag
    opt_plt: int = 0                        # Plot option
    plt_ps: np.ndarray = field(default_factory=lambda: np.array([10, 10, 750, 600]))  # Plot position/size
    plt_wn: int = 2                         # Plot window

    # Animation
    flg_anmt: bool = False                  # Animation flag
    n_anmt: int = 5                         # Iterations per animation frame
    t_anmt: float = 0.05                    # Time between animation frames [s]

    # Reinitialization
    n_r: int = 0                            # Reinitialization interval
    e_r: float = 0.0                        # Reinitialization tolerance

    # Vector field
    flg_vec: int = 0                        # Vector field flag
    n_vec: Union[int, np.ndarray] = 0       # Vector field points

    # Debug
    flg_dbg: bool = False                   # Debug flag

    # DG Method parameters
    method: str = 'fvm'                     # 'fvm' (finite volume) or 'dg' (discontinuous Galerkin)
    dg_order: int = 0                       # DG polynomial order (p=0 is equivalent to FVM)

    def __post_init__(self):
        """Convert lists to numpy arrays where appropriate."""
        if isinstance(self.flg_fld, list):
            self.flg_fld = np.array(self.flg_fld)
        if isinstance(self.flg_BCs, list):
            self.flg_BCs = np.array(self.flg_BCs)
        if isinstance(self.plt_ps, list):
            self.plt_ps = np.array(self.plt_ps)


def apply_defaults(params: dict, n_dim: int) -> dict:
    """
    Apply default values to parameters if not specified.

    Parameters
    ----------
    params : dict
        Dictionary of user-specified parameters
    n_dim : int
        Number of spatial dimensions

    Returns
    -------
    dict
        Parameters with defaults applied
    """
    # Required parameters (will raise error if not provided)
    required = ['n_dim', 'dx', 'x_min', 'x_max', 'flg_fld', 'EoS', 'c_EoS', 'slvr']
    for key in required:
        if key not in params:
            raise ValueError(f"defaults: {key} not specified")

    # Set defaults
    defaults = {
        't_f': 1.0,
        'cfl': 0.9,
        'flg_BCs': 0,
        'n_nds': 0,
        'ICs_hdr': "%------------------ (Simulation Description Unspecified) ----------------%",
        'n_disp': 10,
        'flg_anmt': False,
        'n_anmt': 5,
        't_anmt': 0.05,
        'wrt_prfx': "data/",
        'wrt_sfx': ".d",
        'n_rstrt': 0,
        'rd_nm': "flow_unnamed",
        'rd_prfx': "data/",
        'rd_sfx': ".flo",
        'plt_wn': 2,
        'flg_vec': 0,
        'n_vec': 0,
        't_0': 0.0,
        'flg_dbg': False,
        'method': 'fvm',
        'dg_order': 0
    }

    for key, value in defaults.items():
        if key not in params:
            params[key] = value

    # Handle flg_BCs expansion
    if isinstance(params['flg_BCs'], (int, float)):
        if n_dim == 1:
            params['flg_BCs'] = np.ones(2) * params['flg_BCs']
        elif n_dim == 2:
            params['flg_BCs'] = np.ones(4) * params['flg_BCs']

    # Handle n_out for 2D
    if n_dim == 2:
        if 'n_out' in params:
            if isinstance(params['n_out'], (int, float)):
                params['n_out'] = np.array([params['n_out'], params['n_out']])
        elif 'dx_out' in params:
            dx_out = params['dx_out']
            if isinstance(dx_out, (int, float)):
                dx_out = np.array([dx_out, dx_out])
            x_min = params['x_min']
            x_max = params['x_max']
            params['n_out'] = np.array([
                int((x_max[0] - x_min[0]) / dx_out[0] + 1),
                int((x_max[1] - x_min[1]) / dx_out[1] + 1)
            ])

    # Set interpolation flag
    params['flg_intrp'] = 'n_out' in params
    if not params['flg_intrp']:
        params['n_out'] = 0

    # Set write flag
    params['flg_wrt'] = 'wrt_nm' in params

    # Set plot flag
    if 'opt_plt' in params and params['opt_plt'] > 0:
        params['flg_plt'] = True
    else:
        params['flg_plt'] = False
        params['opt_plt'] = 0

    # Set reinitialization defaults
    if 'n_r' not in params:
        params['n_r'] = 0
        params['e_r'] = 0.0

    # Animation flag depends on plot flag
    if not params['flg_plt']:
        params['flg_anmt'] = False

    # Plot position/size defaults
    if 'plt_ps' not in params:
        if n_dim == 1:
            params['plt_ps'] = np.array([10, 10, 750, 600])
        elif n_dim == 2:
            params['plt_ps'] = np.array([10, 10, 1350, 750])

    # Vector field
    if 'n_vec' in params:
        params['flg_vec'] = 2
        if isinstance(params['n_vec'], (int, float)):
            params['n_vec'] = np.array([params['n_vec'], params['n_vec']])

    return params


def create_parameters(config: dict) -> Parameters:
    """
    Create a Parameters object from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary from input file

    Returns
    -------
    Parameters
        Initialized Parameters object
    """
    n_dim = config['n_dim']
    config = apply_defaults(config, n_dim)

    return Parameters(
        n_dim=config['n_dim'],
        n_var=config.get('n_var', n_dim + 2),
        n=config.get('n', 100),
        dx=config['dx'],
        x_min=config['x_min'],
        x_max=config['x_max'],
        t_f=config['t_f'],
        cfl=config['cfl'],
        t_0=config.get('t_0', 0.0),
        flg_fld=np.array(config['flg_fld']),
        EoS=config['EoS'],
        c_EoS=config['c_EoS'],
        slvr=config['slvr'],
        flg_BCs=np.array(config['flg_BCs']),
        n_nds=config['n_nds'],
        ICs_hdr=config['ICs_hdr'],
        n_disp=config['n_disp'],
        n_out=config['n_out'],
        flg_intrp=config['flg_intrp'],
        flg_wrt=config['flg_wrt'],
        wrt_nm=config.get('wrt_nm', 'flow_unnamed'),
        wrt_prfx=config['wrt_prfx'],
        wrt_sfx=config['wrt_sfx'],
        n_rstrt=config['n_rstrt'],
        rd_nm=config['rd_nm'],
        rd_prfx=config['rd_prfx'],
        rd_sfx=config['rd_sfx'],
        flg_plt=config['flg_plt'],
        opt_plt=config['opt_plt'],
        plt_ps=np.array(config['plt_ps']),
        plt_wn=config['plt_wn'],
        flg_anmt=config['flg_anmt'],
        n_anmt=config['n_anmt'],
        t_anmt=config['t_anmt'],
        n_r=config['n_r'],
        e_r=config.get('e_r', 0.0),
        flg_vec=config['flg_vec'],
        n_vec=config['n_vec'],
        flg_dbg=config['flg_dbg'],
        method=config['method'],
        dg_order=config['dg_order']
    )
