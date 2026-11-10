@echo off
rem Fire-VAD segmentation tool - Windows launcher.
rem
rem Double-click this file. It finds Python 3.9+ with Tk 8.6+ (and offers to
rem install Python 3.13 when there is none), installs numpy and Pillow into the
rem private .venv-segmenter folder next to this file (first run only, needs
rem internet) and starts the segmentation tool.
rem
rem   run_windows.bat --check-runtime   show the selected Python, change nothing
rem   run_windows.bat --selftest        headless check
rem
rem FIREVAD_PYTHON=C:\path\python.exe selects a specific Python.
rem FIREVAD_NONINTERACTIVE=1 never waits for a key, opens a browser or installs.

setlocal EnableExtensions DisableDelayedExpansion
title Fire-VAD segmentation tool
set "HERE=%~dp0"
cd /d "%HERE%"
set "VENV=%HERE%.venv-segmenter"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "PROBE_OUT=%TEMP%\firevad_probe_%RANDOM%%RANDOM%.txt"
set "PY="
set "STATUS=0"
rem Python check without percent signs or double quotes, so cmd passes it unchanged.
set "PROBE=import sys; sys.exit('Python 3.9 or newer is required.') if sys.version_info < (3, 9) else None; exec('try:\n import tkinter\nexcept ImportError:\n sys.exit(\'This Python has no tkinter support.\')'); import tkinter; sys.exit('Tk ' + str(tkinter.TkVersion) + ' is too old; Tk 8.6+ is required to avoid blank windows.') if tkinter.TkVersion < 8.6 else print('Python ' + sys.version.split()[0] + ' / Tk ' + str(tkinter.TkVersion))"

echo Fire-VAD segmentation tool
echo ========================

if exist "%HERE%segment_gui.py" goto :find_python
echo ERROR: segment_gui.py not found next to this script.
echo Extract the downloaded ZIP first (right-click it, Extract All...) and
echo start run_windows.bat from the extracted folder.
goto :fail

rem ---- select a supported Python/Tk runtime ---------------------------------
:find_python
if defined FIREVAD_PYTHON goto :use_override
call :try "%VENV_PY%"
call :try_py_launcher
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :try "%%~D\python.exe"
for /d %%D in ("%ProgramFiles%\Python3*") do call :try "%%~D\python.exe"
for /f "delims=" %%P in ('where python python3 2^>nul') do call :try "%%P"
if defined PY goto :found
goto :no_python

:use_override
rem An explicit override must fail clearly, not silently choose another Python.
set "CAND="
if exist "%FIREVAD_PYTHON%" set "CAND=%FIREVAD_PYTHON%"
if not defined CAND for /f "delims=" %%P in ('where "%FIREVAD_PYTHON%" 2^>nul') do if not defined CAND set "CAND=%%P"
if defined CAND goto :probe_override
echo ERROR: FIREVAD_PYTHON not found: "%FIREVAD_PYTHON%"
goto :fail
:probe_override
"%CAND%" -c "%PROBE%" > "%PROBE_OUT%" 2>&1
if errorlevel 1 goto :override_unusable
set "PY=%CAND%"
goto :found
:override_unusable
echo ERROR: "%CAND%" cannot be used:
type "%PROBE_OUT%"
goto :fail

:no_python
echo.
echo No Python 3.9+ with Tk 8.6+ was found on this computer.
if defined FIREVAD_NONINTERACTIVE goto :python_help
where winget >nul 2>nul
if errorlevel 1 goto :python_help
echo This script can install Python 3.13 for your Windows user now
echo (official python.org package via winget, no administrator rights needed).
choice /C YN /M "Install Python 3.13 now"
if errorlevel 2 goto :python_help
winget install --id Python.Python.3.13 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
rem The new Python is not on this window's PATH yet; look in its default folder.
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :try "%%~D\python.exe"
if defined PY goto :found
echo Python is still not available after the installation.
:python_help
echo.
echo Install Python 3.13 from https://www.python.org/downloads/windows/
echo (Windows installer 64-bit) and keep the option "tcl/tk and IDLE" selected.
echo Then double-click run_windows.bat again.
if not defined FIREVAD_NONINTERACTIVE start "" "https://www.python.org/downloads/windows/"
goto :fail

:found
echo Selected: "%PY%"
type "%PROBE_OUT%"
rem Diagnostic only: no GUI, config writes, venv creation, or pip installs.
if /i "%~1"=="--check-runtime" goto :success

rem ---- install packages into an isolated local environment -----------------
"%PY%" -c "import numpy; from PIL import Image, ImageTk" >nul 2>nul
if not errorlevel 1 goto :launch
if /i "%PY%"=="%VENV_PY%" goto :install
if exist "%VENV%" goto :reuse_venv
echo Creating the segmentation environment in "%VENV%" ...
"%PY%" -m venv "%VENV%"
if not errorlevel 1 goto :venv_created
rem Created by this run; leave nothing half-built behind.
if exist "%VENV%" rmdir /s /q "%VENV%"
echo ERROR: could not create a Python virtual environment.
goto :fail
:venv_created
set "PY=%VENV_PY%"
:install
echo Installing numpy and Pillow for this tool (first run only, needs internet)...
"%PY%" -m pip install --disable-pip-version-check --only-binary=:all: -r "%HERE%requirements.txt"
if not errorlevel 1 goto :verify_install
echo ERROR: installing numpy and Pillow failed. Check the internet connection.
echo If this Python is very new, install Python 3.13 instead and delete .venv-segmenter.
goto :fail
:verify_install
"%PY%" -c "import numpy; from PIL import Image, ImageTk"
if errorlevel 1 goto :fail
goto :launch

:reuse_venv
rem Reached with FIREVAD_PYTHON: reuse the environment created from it before.
if not exist "%VENV_PY%" goto :venv_unusable
"%VENV_PY%" -c "%PROBE%" > "%PROBE_OUT%" 2>&1
if errorlevel 1 goto :venv_unusable
set "PY=%VENV_PY%"
"%PY%" -c "import numpy; from PIL import Image, ImageTk" >nul 2>nul
if not errorlevel 1 goto :launch
goto :install

:venv_unusable
echo ERROR: "%VENV%" exists but is not a usable segmentation environment.
echo It has been left untouched. Delete the .venv-segmenter folder and start again.
goto :fail

rem ---- launch -----------------------------------------------------------------
:launch
echo.
echo Starting with: "%PY%"
echo Keep this window open until you close the segmentation tool.
echo.
"%PY%" "%HERE%segment_gui.py" %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="0" goto :success
goto :exit_error

:success
call :cleanup
exit /b 0

:fail
set "STATUS=1"
:exit_error
echo.
echo The tool could not start or exited with an error (code %STATUS%).
call :cleanup
if not defined FIREVAD_NONINTERACTIVE pause
exit /b %STATUS%

rem ---- subroutines -------------------------------------------------------------
:try
rem Select %1 when it is a supported Python; otherwise say why it was skipped.
if defined PY exit /b 0
if not exist "%~1" exit /b 1
"%~1" -c "%PROBE%" > "%PROBE_OUT%" 2>&1
if errorlevel 1 goto :try_skip
set "PY=%~1"
exit /b 0
:try_skip
echo Skipping "%~1":
type "%PROBE_OUT%"
exit /b 1

:try_py_launcher
if defined PY exit /b 0
set "PYL="
for /f "usebackq delims=" %%P in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do set "PYL=%%P"
if defined PYL call :try "%PYL%"
exit /b 0

:cleanup
if exist "%PROBE_OUT%" del "%PROBE_OUT%" >nul 2>nul
exit /b 0
