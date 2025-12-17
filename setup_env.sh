#!/bin/bash
# Setup script for LAUMeltFlow with PyTorch support
# Requires Python 3.10-3.12 (PyTorch doesn't support 3.13+ yet)

set -e  # Exit on error

echo "=========================================="
echo "LAUMeltFlow Environment Setup"
echo "=========================================="

# Deactivate any active venv
deactivate 2>/dev/null || true

# Find a compatible Python version (3.10, 3.11, or 3.12)
PYTHON_CMD=""

for version in python3.12 python3.11 python3.10; do
    if command -v $version &> /dev/null; then
        # Check if this version actually works
        if $version --version &> /dev/null; then
            PYTHON_CMD=$version
            echo "Found compatible Python: $($PYTHON_CMD --version)"
            break
        fi
    fi
done

# Also check Homebrew paths on macOS
if [ -z "$PYTHON_CMD" ]; then
    for version in 3.12 3.11 3.10; do
        if [ -f "/usr/local/opt/python@$version/bin/python3" ]; then
            PYTHON_CMD="/usr/local/opt/python@$version/bin/python3"
            echo "Found Homebrew Python: $($PYTHON_CMD --version)"
            break
        fi
        if [ -f "/opt/homebrew/opt/python@$version/bin/python3" ]; then
            PYTHON_CMD="/opt/homebrew/opt/python@$version/bin/python3"
            echo "Found Homebrew Python (ARM): $($PYTHON_CMD --version)"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No compatible Python (3.10-3.12) found!"
    echo ""
    echo "Please install Python 3.11 or 3.12:"
    echo "  brew install python@3.11"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Remove old venv if it exists
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create new venv
echo ""
echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install numpy<2 first (PyTorch 2.2 requires numpy 1.x)
echo ""
echo "Installing numpy<2 (required for PyTorch compatibility)..."
pip install "numpy>=1.24,<2"

# Install other core requirements
echo ""
echo "Installing scipy and matplotlib..."
pip install "scipy>=1.11.0" "matplotlib>=3.7.0"

# Install PyTorch
echo ""
echo "Installing PyTorch..."
pip install torch

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import scipy; print(f'scipy: {scipy.__version__}')"
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""

# Run the multiphase test
echo "=========================================="
echo "Running multiphase test..."
echo "=========================================="
python meltflow_gnn/test_multiphase.py --config in_1Dsod2fl --output multiphase_comparison.png

echo ""
echo "Done! Check multiphase_comparison.png for results."
