@echo off
REM Train and test MLP for multiphase (two-fluid) simulations

setlocal EnableDelayedExpansion

echo ==========================================
echo Multiphase MLP Training Pipeline
echo ==========================================

REM Check if venv exists, create if not
if not exist venv\Scripts\activate.bat (
    echo.
    echo Virtual environment not found. Creating venv...
    echo ==========================================

    REM Create venv
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create virtual environment
        echo Make sure Python 3.10-3.12 is installed and on PATH
        exit /b %ERRORLEVEL%
    )

    REM Activate venv
    call venv\Scripts\activate

    REM Upgrade pip
    echo.
    echo Upgrading pip...
    python -m pip install --upgrade pip

    REM Install numpy ^<2 first (PyTorch compatibility)
    echo.
    echo Installing numpy ^<2 ^(required for PyTorch compatibility^)...
    pip install "numpy>=1.24,<2"

    REM Install other dependencies
    echo.
    echo Installing scipy, matplotlib, and numba...
    pip install scipy matplotlib numba

    REM Install PyTorch with CUDA support
    echo.
    echo Installing PyTorch with CUDA 12.4...
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    REM Verify installation
    echo.
    echo ==========================================
    echo Verifying installation...
    echo ==========================================
    python -c "import numpy; print(f'numpy: {numpy.__version__}')"
    python -c "import torch; print(f'torch: {torch.__version__}')"
    python -c "import scipy; print(f'scipy: {scipy.__version__}')"
    python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

    echo.
    echo ==========================================
    echo Virtual environment setup complete!
    echo ==========================================
) else (
    REM Activate existing venv
    call venv\Scripts\activate
)

REM Step 1: Generate training data with importance sampling
echo.
echo Step 1: Generating training data (20M samples with importance sampling)...
echo   Core region (60%%): rho [0.05, 1.5], u [-150, 350], p [5k, 150k]
echo   Extended region (40%%): rho [0.05, 7.0], u [-200, 500], p [5k, 400k]
echo.

python meltflow_gnn\grid_sampler_multiphase.py ^
    --n-samples 20000000 ^
    --importance-sampling ^
    --core-fraction 0.6 ^
    --gamma-values 1.2 1.25 1.289 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.67 ^
    --output data\multiphase_flux_data.npz
if %ERRORLEVEL% neq 0 (
    echo Error in Step 1: Data generation failed
    exit /b %ERRORLEVEL%
)

REM Step 2: Train the model
echo.
echo Step 2: Training MLP (1000 epochs on GPU)...
python meltflow_gnn\train_uniform_cuda.py ^
    --data data\multiphase_flux_data.npz ^
    --epochs 1000 ^
    --hidden-dim 256 ^
    --n-layers 5 ^
    --activation gelu ^
    --batch-size 32768 ^
    --lr 1e-3 ^
    --output models\flux_mlp_multiphase.pt ^
    --output-npz models\flux_mlp_multiphase.npz ^
    --device cuda
if %ERRORLEVEL% neq 0 (
    echo Error in Step 2: Training failed
    exit /b %ERRORLEVEL%
)

REM Step 3: Test on two-fluid case
echo.
echo Step 3: Testing on in_1Dsod2fl...
python meltflow_gnn\test_multiphase.py ^
    --model models\flux_mlp_multiphase.pt ^
    --config in_1Dsod2fl ^
    --output multiphase_comparison_retrained.png
if %ERRORLEVEL% neq 0 (
    echo Error in Step 3: Testing failed
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Done! Check multiphase_comparison_retrained.png
echo ==========================================
