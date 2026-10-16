@echo off
rem ============================================================
rem  License Plate Recognition - Windows launcher (ASCII shell)
rem
rem  Double-click   -> open the GUI
rem  With arguments -> forwarded to the CLI, e.g.
rem      start.bat engines
rem      start.bat predict car.jpg --engine deep
rem
rem  start.bat doctor   -> read-only environment check: interpreter,
rem                        runtime deps, model weights, per-engine
rem                        availability.  It never creates a venv,
rem                        never installs anything, never launches.
rem                        Aliases: --doctor, --check
rem
rem  No version number is written here on purpose: it used to say
rem  v2.3.6 while the project was already at 2.7.x, and a stale
rem  version in the file people double-click is worse than none.
rem  The real version is printed by tools/launcher.py at startup.
rem
rem  THIS FILE MUST STAY PURE ASCII - that is a hard requirement, not a
rem  style choice.  cmd.exe computes the offset of the next batch line
rem  from the previous line's CHARACTER count but applies it as a BYTE
rem  offset, so one multi-byte character makes the parser resume
rem  (bytes - chars) bytes into the following line and run the debris as
rem  a command.  Measured on this project: a single line holding 20 CJK
rem  characters swallowed the first 40 bytes of the next line and
rem  produced the classic "...is not recognized as an internal or
rem  external command".  Short files often mask this because the overrun
rem  lands inside the next line's leading spaces - our own 6-line probe
rem  passed for exactly that reason.
rem
rem  Therefore all user-facing text (banner, version check, .venv and
rem  pip work, model-weight report, Chinese errors) lives in
rem  tools/launcher.py, which Python prints through WriteConsoleW -
rem  independent of the console codepage (65001 or 936, same result).
rem  `chcp 65001` below only exists so the Chinese fallback message in
rem  tools/no-python.txt renders correctly when we could not find any
rem  Python at all.
rem
rem  Environment overrides (consumed by tools/launcher.py):
rem      LPR_PYTHON    force a specific interpreter
rem      LPR_NO_VENV   do not create/use a project-local .venv
rem ============================================================
setlocal EnableExtensions
set "HERE=%~dp0"
chcp 65001 >nul 2>nul
cd /d "%HERE%"

rem ---------- locate a candidate interpreter (first hit wins) ----------
set "PY="
if defined LPR_PYTHON set "PY=%LPR_PYTHON%"
if not defined PY if exist "%HERE%.venv\Scripts\python.exe" set "PY=%HERE%.venv\Scripts\python.exe"

rem Drop a candidate that cannot even start (stale LPR_PYTHON, broken venv)
if defined PY (
    %PY% -c "import sys" >nul 2>nul
    if errorlevel 1 set "PY="
)

if not defined PY (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY goto :no_python

rem ---------- hand over to tools/launcher.py ----------
%PY% "%HERE%tools\launcher.py" %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    pause
)

endlocal & exit /b %RC%

:no_python
rem No interpreter at all: print the bundled UTF-8 notice with `type`,
rem which copies bytes verbatim.  The batch parser never sees a
rem multi-byte character here, and `chcp 65001` above makes the console
rem render those bytes as the intended Chinese text.
type "%HERE%tools\no-python.txt"
pause
exit /b 1
