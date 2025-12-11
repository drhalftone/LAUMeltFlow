#!/bin/bash
# Setup script for LAUMeltFlow - Linux/macOS
# Creates virtual environment and installs dependencies

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To run a simulation, use:"
echo "    python -m meltflow [options]"
