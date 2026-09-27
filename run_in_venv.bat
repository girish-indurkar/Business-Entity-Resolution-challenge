@echo off
REM Launcher script to execute any Python file or command using the virtual environment (.venv)

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at .venv\Scripts\python.exe
    echo Please create the virtual environment using: python -m venv .venv
    exit /b 1
)

.venv\Scripts\python.exe %*
