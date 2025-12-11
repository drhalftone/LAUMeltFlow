@echo off
REM
REM MeltFlow GNN Training Script
REM Creates virtual environment, installs dependencies with CUDA support, runs training
REM
REM Usage:
REM   run_gnn_train.bat           - Run GNN flux training
REM   run_gnn_train.bat --help    - Show help
REM

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"

echo ============================================
echo MeltFlow GNN Training Setup
echo ============================================

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not found
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo Using Python: %%i

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo.
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Check if PyTorch with CUDA is installed
python -c "import torch; assert torch.cuda.is_available()" >nul 2>nul
if %errorlevel% neq 0 goto :install_deps
python -c "import torch_geometric" >nul 2>nul
if %errorlevel% neq 0 goto :install_deps
python -c "import matplotlib" >nul 2>nul
if %errorlevel% neq 0 goto :install_deps
goto :deps_ready

:install_deps
echo.
echo Installing dependencies...
echo.

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing PyTorch with CUDA 12.4 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing PyTorch Geometric...
python -m pip install torch_geometric

echo Installing other dependencies...
python -m pip install numpy matplotlib scipy

echo.
echo Dependencies installed successfully

:deps_ready
echo.
echo Verifying CUDA setup...
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if "%1"=="--help" goto :showhelp
if "%1"=="-h" goto :showhelp

REM Run training
echo.
echo ============================================
echo Starting GNN Training
echo ============================================
echo.
python -m meltflow_gnn.train %*
goto :done

:showhelp
echo.
echo Usage:
echo   run_gnn_train.bat           Run GNN flux model training
echo   run_gnn_train.bat --help    Show this help
echo.
echo This script will:
echo   1. Create a virtual environment if needed
echo   2. Install PyTorch with CUDA support
echo   3. Install PyTorch Geometric and other dependencies
echo   4. Run the GNN training script
echo.
echo Output:
echo   - Training progress displayed in console
echo   - Model saved to flux_model.pt
echo   - Training plots displayed after completion

:done
echo.
echo Done!
endlocal
