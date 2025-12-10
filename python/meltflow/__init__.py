# MeltFlow - Ghost Fluid Method Solver (Python)
# A framework for simulating multi-phase compressible and incompressible flows

__version__ = "0.1.0"

from .main import run_simulation
from .input.configs import load_config, CONFIGS

__all__ = ['run_simulation', 'load_config', 'CONFIGS']
