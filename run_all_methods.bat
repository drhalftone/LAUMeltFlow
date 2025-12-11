@echo off
REM
REM Run all 1D Sod shock tube methods sequentially with animation
REM
REM Usage:
REM   run_all_methods.bat           - Run all methods with plots
REM   run_all_methods.bat --no-plot - Run all methods without plots
REM

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check for virtual environment
if not exist ".venv" (
    echo Virtual environment not found. Run run_simulation.bat first to set it up.
    exit /b 1
)

REM Activate virtual environment
call ".venv\Scripts\activate.bat"

set "NO_PLOT="
if "%1"=="--no-plot" set "NO_PLOT=--no-plot"

echo ========================================
echo Running all 1D Sod shock tube methods
echo ========================================

echo.
echo --- 1. FVM (baseline) ---
python -m meltflow --config in_1Dsod1fl %NO_PLOT%

echo.
echo --- 2. DG p=0 (should match FVM) ---
python -m meltflow --config in_1Dsod1fl_dg0 %NO_PLOT%

echo.
echo --- 3. DG p=1 (2nd order) ---
python -m meltflow --config in_1Dsod1fl_dg %NO_PLOT%

echo.
echo --- 4. DG p=2 (3rd order) ---
python -m meltflow --config in_1Dsod1fl_dg2 %NO_PLOT%

echo.
echo ========================================
echo All simulations complete!
echo Output files in data/ directory
echo ========================================
echo.
echo Run 'python compare_results.py --all' to see comparison

endlocal
