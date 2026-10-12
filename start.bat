@echo off
rem ============================================================
rem  License Plate Recognition v2.3.5 - Windows launcher
rem
rem  Double-click       -> open the GUI
rem  With arguments     -> forwarded to the CLI, e.g.
rem      start.bat engines
rem      start.bat predict car.jpg --engine deep
rem
rem  Environment overrides:
rem      LPR_PYTHON    force a specific interpreter, e.g.
rem                    set LPR_PYTHON=C:\Python312\python.exe
rem      LPR_NO_VENV   never create a project-local .venv; install
rem                    into whatever interpreter was found instead
rem
rem  Dependency policy (v2.3.5):
rem    1. probe numpy + OpenCV + Pillow + PySide6, not just PySide6.
rem       A clean CPython has none of the first three, so checking
rem       PySide6 alone used to "pass" and then crash on
rem       ModuleNotFoundError: numpy at the very last step;
rem    2. when something is missing, install into a project-local
rem       .venv - a user's global interpreter is never polluted;
rem    3. if that interpreter already has everything, it is reused
rem       as-is (no needless re-download).
rem
rem  NOTE: this file is intentionally ASCII-only. Batch files are
rem  parsed with the system ANSI codepage, so non-ASCII text here
rem  gets garbled and can even be executed as commands.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

rem An inherited PYTHONPATH must not leak into the installer process:
rem a foreign entry there can shadow packages and make pip fail in
rem confusing ways.  Save it, clear it, restore it for the launch.
set "KEEP_PYTHONPATH=%PYTHONPATH%"
set "PYTHONPATH="

echo ============================================
echo  License Plate Recognition v2.3.5
echo ============================================
echo.

rem ---------- 1. locate a candidate interpreter ----------
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
echo [1/4] interpreter OK
%PY% -c "import sys; print('      ', sys.executable)"

rem Keep the originally detected interpreter: it is what we fall back
rem to if .venv cannot be created.  (It is referenced inside the
rem parenthesised blocks below, where %PY% would be stale.)
set "BASE_PY=%PY%"

rem ---------- 3. check runtime + GUI dependencies ----------
rem  All four must import.  PySide6 alone is NOT enough: a clean
rem  CPython has neither numpy nor cv2, so the old PySide6-only check
rem  passed here and then died at the last step.
set "PROBE=import numpy, cv2, PIL, PySide6"

%PY% -c "%PROBE%" >nul 2>nul
if not errorlevel 1 goto :deps_ready

echo [2/4] dependencies missing - preparing an environment...
echo.

rem Never pollute an existing interpreter: prefer a project-local .venv.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    echo        using existing .venv
    goto :install_deps
)

if defined LPR_NO_VENV (
    echo        LPR_NO_VENV is set - installing into: %BASE_PY%
    goto :install_deps
)

echo        creating .venv ...
%PY% -m venv .venv
if errorlevel 1 (
    echo.
    echo [WARN] could not create .venv ^(venv module unavailable?^).
    echo        falling back to: %BASE_PY%
    echo        dependencies will be installed into that interpreter.
    echo.
    goto :install_deps
)
set "PY=.venv\Scripts\python.exe"
echo        created .venv

:install_deps
echo [3/4] installing into %PY% ^(first run only, needs network^)...
echo.
rem ".[deep,gui]" pulls from pyproject: the core stack (numpy / OpenCV /
rem Pillow), the recommended deep engine (rapidocr-onnxruntime, ~62 MB
rem installed) and the GUI toolkit - so pyproject stays the single
rem source of truth for dependencies.  The explicit list below is only
rem a fallback for when the editable install cannot run at all
rem (e.g. no build backend reachable).
%PY% -m pip install -e ".[deep,gui]"
if errorlevel 1 (
    echo.
    echo        editable install failed, retrying with explicit packages...
    %PY% -m pip install "numpy>=1.24" "opencv-python-headless>=4.10,<5" "Pillow>=10.0" "rapidocr-onnxruntime>=1.2" "PySide6-Essentials>=6.5"
    if errorlevel 1 (
        echo.
        echo [ERROR] Install failed. Run manually:
        echo         %PY% -m pip install -e ".[deep,gui]"
        echo.
        pause
        exit /b 1
    )
)

rem Re-probe afterwards: "install succeeded but into the wrong
rem interpreter / wrong name" must not pass silently.
%PY% -c "%PROBE%" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Dependencies still missing after install.
    echo         Try: %PY% -m pip install -e ".[deep,gui]"
    echo.
    pause
    exit /b 1
)
echo.
echo        done - CV engine + deep engine + GUI are ready.
echo        Everything lives in this folder; your global Python is untouched.
echo.
goto :launch

:deps_ready
echo [2/4] dependencies OK
echo [3/4] nothing to install

:launch
rem ---------- 4. launch ----------
if defined KEEP_PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%KEEP_PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
set "ARGS=%*"
if not defined ARGS set "ARGS=gui"

echo [4/4] starting...
echo.
%PY% -m lpr %ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Please report the message above.
    pause
    exit /b 1
)

endlocal
