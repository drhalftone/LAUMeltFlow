@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
if not exist build mkdir build
cd build
C:\Qt\6.9.3\msvc2022_64\bin\qmake.exe ..\BeadChain.pro -spec win32-msvc
if %ERRORLEVEL% NEQ 0 (
    echo QMAKE FAILED
    exit /b 1
)
echo QMAKE SUCCESS
nmake
if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD SUCCESS
