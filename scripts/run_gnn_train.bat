@echo off
echo Activating virtual environment...
call "%~dp0..\venv\Scripts\activate.bat"

echo Changing to project directory...
cd /d "%~dp0.."

echo Running GNN training...
python -m meltflow_gnn.train

pause
