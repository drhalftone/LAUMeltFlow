# MeltFlow - Ghost Fluid Method Solver

A MATLAB framework for simulating multi-phase compressible and incompressible flows using the Ghost Fluid Method (GFM). This solver handles two-fluid systems with sharp interfaces in 1D and 2D domains.

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
- **Incompressible liquid** treatment option
- **Level set interface tracking** with WENO reinitialization
- **Real-time visualization** and animation
- **Multiple equation of state** support (perfect gas)

## Requirements

- MATLAB R2019b or later (uses `griddedInterpolant`, `scatteredInterpolant`)

## Quick Start

1. Open MATLAB and navigate to the project directory
2. Open `main.m`
3. Set the input file on line 18:
   ```matlab
   fl_in = "in_1Dsod1fl";  % Choose input file
   ```
4. Run `main.m`

## Project Structure

```
meltflow-matlab/
├── main.m                  # Main entry point
├── functions/              # Core solver functions
│   ├── roe_perfect.m       # Roe solver for compressible flow
│   ├── ghost_GFM.m         # Ghost fluid method (ghost domain)
│   ├── real_GFM.m          # Ghost fluid method (real domain)
│   ├── advc.m              # Level set advection
│   ├── reinit_fast.m       # Level set reinitialization (WENO)
│   ├── extrp.m             # Field extrapolation
│   ├── extrp_vel.m         # Velocity extrapolation
│   ├── timestep.m          # CFL-based time stepping
│   └── ...                 # Additional utilities
├── input/                  # Test case configurations
├── visualization/          # Plotting functions
├── exact/                  # Exact solutions for validation
├── data/                   # Output data files
└── info/                   # Documentation
```

## Test Cases

| Input File | Description |
|------------|-------------|
| `in_1Dsod1fl` | 1D Sod shock tube (single fluid, two regions) |
| `in_1Dsod2fl` | 1D Sod shock tube (two fluids) |
| `in_1Dcdrop` | 1D centered liquid droplet in gas |
| `in_1Dcdropimp` | 1D centered droplet (incompressible liquid) |
| `in_2Dcdrop` | 2D circular liquid droplet in gas |
| `in_2Dcdropimp` | 2D circular droplet (incompressible liquid) |
| `in_2Dsod1fl` | 2D Sod shock tube |
| `in_fedkiwEx2TestA` | Fedkiw Example 2 Test A validation case |

## Configuration

Input files in `input/` define simulation parameters:

```matlab
% Grid
n_dim = 1;                      % Spatial dimensions (1 or 2)
dx = 0.01;                      % Grid spacing [m]
x_min = 0; x_max = 1;           % Domain boundaries [m]

% Initial conditions per region
U_r(1,:) = [rho, u, p];         % Region 1: [density, velocity, pressure]
U_r(2,:) = [rho, u, p];         % Region 2: [density, velocity, pressure]

% Fluid properties
flg_fld = [0, 1];               % 0 = gas, 1 = liquid
EoS = ["perfect", "none"];      % Equation of state
c_EoS = {1.4, 1};               % EoS parameters (gamma for perfect gas)
slvr = ["roe_perfect", "none"]; % Solver selection

% Simulation
cfl = 0.9;                      % CFL number
t_f = 7.5e-4;                   % Final time [s]
flg_BCs = 1;                    % Boundary condition type

% Output
n_out = 51;                     % Output grid points
wrt_nm = "flow_1Dsod1fl";       % Output filename
opt_plt = 1;                    % Plot option

% Debug
flg_dbg = 0;                    % Debug mode (0 = off, 1 = on)
```

For 2D cases, arrays are used for grid parameters:
```matlab
dx = [0.02, 0.02];              % [dx, dy]
x_min = [0, 0];                 % [x_min, y_min]
x_max = [1, 1];                 % [x_max, y_max]
U_r(1,:) = [rho, u, v, p];      % Includes y-velocity
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
Separates the domain into real and ghost regions for each fluid:
1. Compute entropy in real regions
2. Extrapolate entropy to ghost regions
3. Copy velocity and pressure across interface
4. Compute ghost density from extrapolated entropy
5. Solve each fluid independently

### Flux Solver
- **Roe's approximate Riemann solver** for compressible flows
- Dimensional splitting for 2D (x-sweep, then y-sweep)
- Supports periodic, Dirichlet, and Neumann boundary conditions

### Level Set
- **Advection**: Godunov upwind scheme with artificial viscosity
- **Reinitialization**: 5th-order WENO finite differences

### Time Integration
- Forward Euler with CFL-based adaptive time stepping
- `dt = CFL * dx / (|V|_max + a_max)`

## Output

Results are written to the `data/` folder. Output includes:
- Density, velocity, and pressure fields
- Level set (interface position)
- Grid coordinates

## Adding New Test Cases

1. Create a new file in `input/` (e.g., `in_mycase.m`)
2. Define grid, initial conditions, and solver parameters
3. Include the grid setup at the end:
   ```matlab
   grid_setup;
   % Assign U and phi based on geometry
   ```
4. Set `fl_in = "in_mycase"` in `main.m`

## References

- Fedkiw, R. P., Aslam, T., Merriman, B., & Osher, S. (1999). A non-oscillatory Eulerian approach to interfaces in multimaterial flows (the ghost fluid method). *Journal of Computational Physics*, 152(2), 457-492.
- Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43(2), 357-372.

## License

See LICENSE file for details.
