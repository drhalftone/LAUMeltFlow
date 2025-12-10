@echo off
echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo Changing to python directory...
cd /d "%~dp0python"

echo Running MeltFlow vs GNN comparison...
python -m meltflow_gnn.compare

pause
