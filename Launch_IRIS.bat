@echo off
REM ============================================================
REM IRIS — the everyday launcher. This is the ONLY file you
REM should need to double-click on demo day.
REM
REM You told me you run IRIS with plain "python iris_gui.py" using
REM the system Python install (no venv, no conda env) — so this
REM just does exactly that, from the right folder, without needing
REM a terminal open or remembering the command.
REM ============================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found on PATH.
    echo Make sure Python is installed and that typing "python" in a
    echo normal terminal works, then try this again.
    echo.
    pause
    exit /b 1
)

if not exist "iris_gui.py" (
    echo.
    echo iris_gui.py was not found in this folder:
    echo   %cd%
    echo Make sure this launcher lives in the same folder as iris_gui.py.
    echo.
    pause
    exit /b 1
)

python iris_gui.py

if errorlevel 1 (
    echo.
    echo ============================================
    echo IRIS exited with an error — see the output above.
    echo ============================================
    pause
)