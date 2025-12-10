#!/bin/bash
#
# Run all 1D Sod shock tube methods sequentially with animation
#
# Usage:
#   ./run_all_methods.sh           # Run all methods with plots
#   ./run_all_methods.sh --no-plot # Run all methods without plots
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Run ./run_simulation.sh first to set it up."
    exit 1
fi

NO_PLOT=""
if [ "$1" == "--no-plot" ]; then
    NO_PLOT="--no-plot"
fi

echo "========================================"
echo "Running all 1D Sod shock tube methods"
echo "========================================"

echo ""
echo "--- 1. FVM (baseline) ---"
python -m meltflow --config in_1Dsod1fl $NO_PLOT

echo ""
echo "--- 2. DG p=0 (should match FVM) ---"
python -m meltflow --config in_1Dsod1fl_dg0 $NO_PLOT

echo ""
echo "--- 3. DG p=1 (2nd order) ---"
python -m meltflow --config in_1Dsod1fl_dg $NO_PLOT

echo ""
echo "--- 4. DG p=2 (3rd order) ---"
python -m meltflow --config in_1Dsod1fl_dg2 $NO_PLOT

echo ""
echo "========================================"
echo "All simulations complete!"
echo "Output files in data/ directory"
echo "========================================"
echo ""
echo "Run 'python compare_results.py --all' to see comparison"
