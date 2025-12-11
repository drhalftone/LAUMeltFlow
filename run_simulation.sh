#!/bin/bash
#
# MeltFlow Simulation Runner
# Creates a virtual environment, installs dependencies, and runs simulations
#
# Usage:
#   ./run_simulation.sh                     # Run default FVM simulation
#   ./run_simulation.sh --config in_1Dsod1fl_dg  # Run DG simulation
#   ./run_simulation.sh --list-configs      # List available configs
#   ./run_simulation.sh --compare           # Run FVM vs DG comparison
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MeltFlow Simulation Runner ===${NC}"

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python 3 is required but not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Using Python: $($PYTHON --version)${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip --quiet

# Try full requirements first, fall back to core if numba fails
if pip install -r "$SCRIPT_DIR/requirements.txt" --quiet 2>/dev/null; then
    echo -e "${GREEN}Installed full dependencies (including Numba)${NC}"
else
    echo -e "${YELLOW}Numba installation failed (requires LLVM), using core dependencies...${NC}"
    pip install -r "$SCRIPT_DIR/requirements-core.txt" --quiet
fi

echo -e "${GREEN}Dependencies installed successfully${NC}"

# Handle special flags
if [ "$1" == "--compare" ]; then
    echo -e "\n${GREEN}=== Running FVM vs DG Comparison ===${NC}\n"

    echo -e "${YELLOW}--- Running FVM (in_1Dsod1fl) ---${NC}"
    python -m meltflow --config in_1Dsod1fl --no-plot

    echo -e "\n${YELLOW}--- Running DG p=1 (in_1Dsod1fl_dg) ---${NC}"
    python -m meltflow --config in_1Dsod1fl_dg --no-plot

    echo -e "\n${GREEN}=== Comparison Complete ===${NC}"
    echo "Output files written to data/ directory"

elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\n${GREEN}Usage:${NC}"
    echo "  ./run_simulation.sh                          Run default FVM simulation"
    echo "  ./run_simulation.sh --config <name>          Run specific config"
    echo "  ./run_simulation.sh --config <name> --no-plot  Run without plotting"
    echo "  ./run_simulation.sh --list-configs           List available configs"
    echo "  ./run_simulation.sh --compare                Run FVM vs DG comparison"
    echo ""
    echo -e "${GREEN}Available configurations:${NC}"
    echo "  in_1Dsod1fl       - 1D Sod shock tube (FVM - baseline)"
    echo "  in_1Dsod1fl_dg0   - 1D Sod shock tube (DG p=0, identical to FVM)"
    echo "  in_1Dsod1fl_dg    - 1D Sod shock tube (DG p=1, 2nd order)"
    echo "  in_1Dsod2fl       - 1D Two-fluid shock tube"
    echo "  in_1Dcdrop        - 1D Centered droplet"
    echo "  in_2Dcdrop        - 2D Circular droplet"
    echo "  in_2Dsod1fl       - 2D Sod shock tube"
    echo ""
    echo -e "${GREEN}DG Method Notes:${NC}"
    echo "  Modify dg_order in configs.py:"
    echo "    dg_order=0  ->  Equivalent to FVM (verified identical)"
    echo "    dg_order=1  ->  2nd order accuracy"
    echo "    dg_order=2  ->  3rd order accuracy"
    echo ""
    echo -e "${GREEN}Comparison Testing:${NC}"
    echo "  python compare_results.py                    Compare FVM vs DG results"
    echo "  python compare_results.py --plot             Show comparison plots"

else
    # Run simulation with any provided arguments
    echo -e "\n${GREEN}Running simulation...${NC}\n"
    python -m meltflow "$@"
fi

echo -e "\n${GREEN}Done!${NC}"
