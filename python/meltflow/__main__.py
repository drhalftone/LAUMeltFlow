"""
Entry point for running MeltFlow as a module.

Usage:
    python -m meltflow --config in_1Dsod1fl
    python -m meltflow --config in_2Dcdrop
    python -m meltflow --list-configs
"""

from .main import main

if __name__ == '__main__':
    main()
