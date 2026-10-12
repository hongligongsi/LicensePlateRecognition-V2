@echo off
rem ============================================================
rem  License Plate Recognition v2.3.3 - Windows launcher
rem
rem  Double-click       -> open the GUI
rem  With arguments     -> forwarded to the CLI, e.g.
rem      start.bat engines
rem      start.bat predict car.jpg --engine deep
rem
rem  Set LPR_PYTHON to force a specific interpreter, e.g.
rem      set LPR_PYTHON=C:\Python312\python.exe
rem
rem  NOTE: this file is intentionally ASCII-only. Batch files are
rem  parsed with the system ANSI codepage, so non-ASCII text here
rem  gets garbled and can even be executed as commands.
rem ============================================================
setlocal
cd /d "%~dp0"

echo ============================================
echo  License Plate Recognition v2.3.3
echo ============================================
echo.

rem ---------- 1. locate a Python interpreter ----------
set "PY="
if defined LPR_PYTHON set "PY=%LPR_PYTHON%"

if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not defined PY (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [ERROR] No Python found.
    echo.
    echo   Install Python 3.10 or newer ^(64-bit^):
    echo     https://www.python.org/downloads/windows/
    echo   Remember to tick "Add python.exe to PATH".
    echo.
    echo   Note: Windows 7 / 8 / 8.1 / XP cannot run this program.
    echo         See "Environment and OS requirements" in README.md.
    echo.
    pause
    exit /b 1
)

rem ---------- 2. check the interpreter version ----------
%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    %PY% -c "import sys; print('        current:', sys.version.split()[0])"
    echo.
    pause
    exit /b 1
)
echo [1/3] interpreter OK
%PY% -c "import sys; print('      ', sys.executable)"

rem ---------- 3. check the GUI dependency ----------
%PY% -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo [2/3] PySide6 missing, installing ^(needs network^)...
    %PY% -m pip install "PySide6>=6.5"
    if errorlevel 1 (
        echo.
        echo [ERROR] Install failed. Run manually:
        echo         %PY% -m pip install -e ".[gui,deep]"
        echo.
        pause
        exit /b 1
    )
) else (
    echo [2/3] GUI dependency OK
)

rem ---------- 4. launch ----------
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "ARGS=%*"
if not defined ARGS set "ARGS=gui"

echo [3/3] starting...
echo.
%PY% -m lpr %ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Please report the message above.
    pause
    exit /b 1
)

endlocal
