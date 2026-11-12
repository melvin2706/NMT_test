@echo off
REM KamerNMT - one-command setup for Windows.
REM Creates .venv in this folder, installs PyTorch (CPU) and all libraries,
REM and installs the `kamernmt` command.
REM Usage:  setup.bat          (CPU, recommended)
REM         setup.bat gpu      (PyTorch with NVIDIA CUDA, for faster training)

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org and tick "Add to PATH".
    exit /b 1
)

if not exist .venv (
    echo Creating the virtual environment .venv ...
    python -m venv .venv || exit /b 1
)

set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip || exit /b 1

if /i "%1"=="gpu" (
    echo Installing PyTorch with CUDA ...
    %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
) else (
    echo Installing PyTorch for CPU ...
    %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
)

echo Installing the other libraries ...
%PY% -m pip install -r requirements.txt || exit /b 1
echo Installing the kamernmt command ...
%PY% -m pip install -e . || exit /b 1

echo.
echo Done. Activate the environment with:   .venv\Scripts\activate
echo Then:
echo   python main.py --demo                     check everything works (a few minutes)
echo   python main.py                            train the model
echo   kamernmt                                  translate interactively
echo   kamernmt --tgt dua "Je vais au marche."   translate a phrase
endlocal
