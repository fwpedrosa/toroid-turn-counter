@echo off
REM Builds dist\ToroidCounter.exe on Windows. Needs Python 3.10+ from python.org (with tcl/tk).
cd /d "%~dp0\.."
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q tests || (echo Tests failed & exit /b 1)
pyinstaller packaging\ToroidCounter.spec --noconfirm --clean
echo.
echo Done: %cd%\dist\ToroidCounter.exe
