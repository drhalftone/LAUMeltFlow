@echo off
REM
REM MeltFlow Simulation Runner
REM Creates a virtual environment, installs dependencies, and runs simulations
REM
REM Usage:
REM   run_simulation.bat                     - Run default FVM simulation
REM   run_simulation.bat --config in_1Dsod1fl_dg  - Run DG simulation
REM   run_simulation.bat --list-configs      - List available configs
REM   run_simulation.bat --compare           - Run FVM vs DG comparison
REM

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"

echo === MeltFlow Simulation Runner ===

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not found
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo Using Python: %%i

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip and install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip --quiet

REM Try full requirements first, fall back to core if numba fails
python -m pip install -r "%SCRIPT_DIR%requirements.txt" --quiet 2>nul
if %errorlevel% neq 0 goto :coredeps
echo Installed full dependencies including Numba
goto :depsdone

:coredeps
echo Numba installation failed, using core dependencies...
python -m pip install -r "%SCRIPT_DIR%requirements-core.txt" --quiet

:depsdone
echo Dependencies installed successfully

REM Handle special flags
if "%1"=="--compare" goto :compare
if "%1"=="--help" goto :showhelp
if "%1"=="-h" goto :showhelp

REM Run simulation with any provided arguments
echo.
echo Running simulation...
echo.
python -m meltflow %*
goto :done

:compare
echo.
echo === Running FVM vs DG Comparison ===
echo.
echo --- Running FVM in_1Dsod1fl ---
python -m meltflow --config in_1Dsod1fl --no-plot
echo.
echo --- Running DG p=1 in_1Dsod1fl_dg ---
python -m meltflow --config in_1Dsod1fl_dg --no-plot
echo.
echo === Comparison Complete ===
echo Output files written to data/ directory
goto :done

:showhelp
echo.
echo Usage:
echo   run_simulation.bat                             Run default FVM simulation
echo   run_simulation.bat --config [name]             Run specific config
echo   run_simulation.bat --config [name] --no-plot   Run without plotting
echo   run_simulation.bat --list-configs              List available configs
echo   run_simulation.bat --compare                   Run FVM vs DG comparison
echo.
echo Available configurations:
echo   in_1Dsod1fl       - 1D Sod shock tube, FVM baseline
echo   in_1Dsod1fl_dg0   - 1D Sod shock tube, DG p=0 identical to FVM
echo   in_1Dsod1fl_dg    - 1D Sod shock tube, DG p=1 2nd order
echo   in_1Dsod2fl       - 1D Two-fluid shock tube
echo   in_1Dcdrop        - 1D Centered droplet
echo   in_2Dcdrop        - 2D Circular droplet
echo   in_2Dsod1fl       - 2D Sod shock tube
echo.
echo DG Method Notes:
echo   Modify dg_order in configs.py:
echo     dg_order=0  Equivalent to FVM
echo     dg_order=1  2nd order accuracy
echo     dg_order=2  3rd order accuracy
echo.
echo Comparison Testing:
echo   python compare_results.py             Compare FVM vs DG results
echo   python compare_results.py --plot      Show comparison plots

:done
echo.
echo Done!
endlocal
