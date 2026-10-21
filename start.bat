@echo off
cd /d "%~dp0"
if not exist .venv (
    echo Setting up for the first time...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -e .
) else (
    call .venv\Scripts\activate.bat
)
python -m ticket_classifier
pause
