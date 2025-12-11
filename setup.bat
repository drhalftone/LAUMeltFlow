@echo off
REM Setup script for LAUMeltFlow - Windows
REM Creates virtual environment and installs dependencies

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! To activate the environment, run:
echo     venv\Scripts\activate.bat
echo.
echo To run a simulation, use:
echo     python -m meltflow [options]
