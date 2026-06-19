# MeltFlow - Ghost Fluid Method Solver

A Python framework for simulating multi-phase compressible and incompressible flows using the Ghost Fluid Method (GFM). This solver handles two-fluid systems with sharp interfaces in 1D and 2D domains.

> **Note:** This is a Python translation of the original MATLAB implementation.

## Overview

MeltFlow solves the Euler equations for multi-fluid systems, enabling simulation of:

- Shock tubes with multiple fluids
- Droplet dynamics and oscillations
- Bubble and cavitation dynamics
- Gas-liquid interface interactions

The solver uses the Ghost Fluid Method to enforce physical jump conditions at fluid interfaces while avoiding numerical smearing.

## Features

- **1D and 2D simulations** with structured rectangular grids
- **Multi-phase flow** support for gas-liquid systems
- **Compressible flows** using Roe's approximate Riemann solver
- **Discontinuous Galerkin (DG) method** with selectable polynomial order (p=0,1,2)
- **Incompressible liquid** treatment option
- **Level set interface tracking** with WENO reinitialization
- **Real-time visualization** and animation
- **Multiple equation of state** support (perfect gas)
- **FVM/DG comparison tools** for method validation

## Installation

### Prerequisites

- Python 3.8 or later
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:drhalftone/LAUMeltFlow.git
   cd LAUMeltFlow
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or use the automated setup script:
   ```bash
   ./run_simulation.sh --help
   ```
   This creates a virtual environment and installs all dependencies automatically.

## Quick Start

### Running from Command Line

```bash
# Run the default 1D Sod shock tube
python -m meltflow --config in_1Dsod1fl

# Run a 2D droplet simulation
python -m meltflow --config in_2Dcdrop

# List available configurations
python -m meltflow --list-configs

# Run without plotting (faster)
python -m meltflow --config in_1Dsod1fl --no-plot
```

### Running from Python

```python
from meltflow import run_simulation

# Run simulation and get results
results = run_simulation('in_1Dsod1fl')

# Access results
X = results['X']       # Grid coordinates
U = results['U']       # Primitive variables [rho, u, (v), p]
phi = results['phi']   # Level set function
prm = results['prm']   # Simulation parameters
```

### Custom Simulations

```python
from meltflow.input.configs import load_config
from meltflow.functions import (
    grid_setup, create_parameters, state_var,
    ghost_GFM, run_slvr, real_GFM, timestep
)

# Load and modify a configuration
config, init_func = load_config('in_1Dsod1fl')
config['t_f'] = 1e-3  # Change final time
config['cfl'] = 0.5   # Change CFL number

# ... continue with simulation setup
```

## Project Structure

```
LAUMeltFlow/
├── README.md
├── requirements.txt
├── .gitignore
└── meltflow/
    ├── __init__.py              # Package exports
    ├── __main__.py              # CLI entry point
    ├── main.py                  # Main simulation runner
    │
    ├── functions/               # Core solver functions
    │   ├── thermodynamics.py    # EoS functions (pressure, entropy, etc.)
    │   ├── parameters.py        # Parameters dataclass
    │   ├── grid.py              # Grid setup and meshing
    │   ├── state_var.py         # State variable computations
    │   ├── flux.py              # Roe flux calculations
    │   ├── roe_perfect.py       # Roe solver for perfect gas
    │   ├── ghost_fluid.py       # Ghost Fluid Method implementation
    │   ├── level_set.py         # Level set advection and reinitialization
    │   ├── solver.py            # Solver dispatch and time stepping
    │   ├── io.py                # File I/O and interpolation
    │   ├── plotting.py          # Visualization functions
    │   └── geometry/            # Geometry functions (future)
    │
    ├── input/                   # Test case configurations
    │   └── configs.py           # Predefined simulation setups
    │
    ├── exact/                   # Exact solutions for validation
    ├── testing/                 # Unit tests
    └── visualization/           # Additional visualization tools
```

## Test Cases

| Configuration | Description |
|---------------|-------------|
| `in_1Dsod1fl` | 1D Sod shock tube (FVM baseline) |
| `in_1Dsod1fl_dg0` | 1D Sod shock tube (DG p=0, identical to FVM) |
| `in_1Dsod1fl_dg` | 1D Sod shock tube (DG p=1, 2nd order) |
| `in_1Dsod1fl_dg2` | 1D Sod shock tube (DG p=2, 3rd order) |
| `in_1Dsod2fl` | 1D Sod shock tube (two different gases) |
| `in_1Dcdrop` | 1D centered liquid droplet in gas |
| `in_2Dcdrop` | 2D circular liquid droplet in gas |
| `in_2Dsod1fl` | 2D Sod shock tube |

## Configuration Parameters

Test cases are defined in `meltflow/input/configs.py`. Key parameters include:

```python
config = {
    # Grid parameters
    'n_dim': 1,                    # Spatial dimensions (1 or 2)
    'dx': 0.01,                    # Grid spacing [m]
    'x_min': 0.0,                  # Domain minimum [m]
    'x_max': 1.0,                  # Domain maximum [m]

    # Fluid properties
    'flg_fld': [0, 1],             # Fluid type: 0=gas, 1=liquid
    'EoS': ["perfect", "none"],    # Equation of state
    'c_EoS': [1.4, 1.0],           # EoS parameters (gamma for perfect gas)
    'slvr': ["roe_perfect", "none"], # Solver selection

    # Time integration
    'cfl': 0.9,                    # CFL number
    't_f': 7.5e-4,                 # Final time [s]

    # Boundary conditions
    'flg_BCs': 1,                  # BC type: 0=Dirichlet, 1=Neumann, 2=Periodic

    # Output
    'n_out': 51,                   # Output grid points
    'wrt_nm': "flow_output",       # Output filename
    'opt_plt': 1,                  # Plot option
}
```

For 2D cases, use arrays for grid parameters:
```python
config = {
    'n_dim': 2,
    'dx': np.array([0.02, 0.02]),           # [dx, dy]
    'x_min': np.array([0.0, 0.0]),          # [x_min, y_min]
    'x_max': np.array([1.0, 1.0]),          # [x_max, y_max]
    # ...
}
```

## Variables

| Variable | Description |
|----------|-------------|
| `phi` | Level set function (positive = fluid 1, negative = fluid 2) |
| `U` | Primitive variables `[rho, u, (v), p]^T` |
| `W` | Conserved variables `[rho, rho*u, (rho*v), E]^T` |
| `X` | Grid coordinates `[X, (Y)]` |

## Numerical Methods

### Ghost Fluid Method

The GFM separates the domain into real and ghost regions for each fluid:

1. Compute entropy in real regions
2. Extrapolate entropy to ghost regions
3. Copy velocity and pressure across interface
4. Compute ghost density from extrapolated entropy
5. Solve each fluid independently

### Flux Solver

- **Roe's approximate Riemann solver** for compressible flows
- Dimensional splitting for 2D (x-sweep, then y-sweep)
- Supports periodic, Dirichlet, and Neumann boundary conditions

### Discontinuous Galerkin Method

The DG solver (`dg_perfect.py`) implements a higher-order accurate method:

- **p=0**: Piecewise constant (mathematically identical to FVM)
- **p=1**: Piecewise linear (2nd order accurate)
- **p=2**: Piecewise quadratic (3rd order accurate)

Uses the same Roe numerical flux as FVM for fair comparison. Select the method via config:

```python
config = {
    'slvr': ["dg_perfect", "none"],  # Use DG solver
    'method': 'dg',
    'dg_order': 1,  # Polynomial order (0, 1, or 2)
}
```

### Level Set

- **Advection**: Godunov upwind scheme with artificial viscosity
- **Reinitialization**: 5th-order WENO finite differences

### Time Integration

- Forward Euler with CFL-based adaptive time stepping
- `dt = CFL * dx / (|V|_max + a_max)`

## Output

Results can be written to files in the `data/` directory:

```python
# Output format for 1D:
# x rho u p phi

# Output format for 2D:
# x y rho u v p phi
```

## Adding New Test Cases

1. Add a new function in `meltflow/input/configs.py`:

```python
def in_mycase():
    config = {
        'n_dim': 1,
        'dx': 0.01,
        'x_min': 0.0,
        'x_max': 1.0,
        # ... other parameters
    }

    def init_func(x, U, phi):
        # Set initial conditions
        n = len(x)
        for i in range(n):
            if x[i] < 0.5:
                U[:, i] = [1.0, 0.0, 1e5]  # [rho, u, p]
            else:
                U[:, i] = [0.125, 0.0, 1e4]
        phi[:] = 1  # Single fluid
        return U, phi

    return config, init_func
```

2. Register it in the `CONFIGS` dictionary:

```python
CONFIGS = {
    # ... existing configs
    'in_mycase': in_mycase,
}
```

3. Run your new case:

```bash
python -m meltflow --config in_mycase
```

## Dependencies

- **NumPy** >= 1.24.0 - Array operations
- **SciPy** >= 1.11.0 - Interpolation and scientific computing
- **Matplotlib** >= 3.7.0 - Plotting and visualization
- **Numba** >= 0.58.0 - JIT compilation (optional, for future performance)

## Comparison Tools

Scripts are provided to compare FVM and DG methods:

```bash
# Run all methods without plots
python run_batch.py

# Run all methods and show comparison
python run_batch.py --compare

# Compare results numerically and generate plot
python compare_results.py --all

# Save comparison plot to file
python compare_results.py --save
```

The comparison tools generate plots showing all methods overlaid:

![Method Comparison](data/method_comparison.png)

## Scripts

| Script | Description |
|--------|-------------|
| `run_simulation.sh` | Setup venv and run single simulation |
| `run_batch.py` | Run multiple methods sequentially |
| `run_all_methods.sh` | Run all 1D Sod methods with plots |
| `compare_results.py` | Compare and visualize results |

## References

- Fedkiw, R. P., Aslam, T., Merriman, B., & Osher, S. (1999). A non-oscillatory Eulerian approach to interfaces in multimaterial flows (the ghost fluid method). *Journal of Computational Physics*, 152(2), 457-492.

- Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43(2), 357-372.

- Cockburn, B., & Shu, C. W. (1998). The Runge-Kutta discontinuous Galerkin method for conservation laws V: multidimensional systems. *Journal of Computational Physics*, 141(2), 199-224.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This material is based upon work supported by the U.S. National Science Foundation
under Grant No. 2230162, *Collaborative Research: CIF: Small: Hypergraph Signal
Processing and Networks via t-Product Decompositions* (PI: Daniel L. Lau, University
of Kentucky). This software was developed by Grey Goodwin as part of his M.S. research
supported by this award.

Any opinions, findings, and conclusions or recommendations expressed in this material
are those of the author(s) and do not necessarily reflect the views of the National
Science Foundation.
