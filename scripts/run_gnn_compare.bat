@echo off
echo Activating virtual environment...
call "%~dp0..\venv\Scripts\activate.bat"

echo Changing to project directory...
cd /d "%~dp0.."

echo Running MeltFlow vs GNN comparison...
python -m meltflow_gnn.compare

pause
