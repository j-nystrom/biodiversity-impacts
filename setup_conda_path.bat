@echo off
setlocal

if "%CONDA_PREFIX%"=="" (
  echo ERROR: CONDA_PREFIX is not set. Activate the conda env first.
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%core\" (
  set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
) else (
  for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
)

set "ACTIVATE_DIR=%CONDA_PREFIX%\etc\conda\activate.d"
set "DEACTIVATE_DIR=%CONDA_PREFIX%\etc\conda\deactivate.d"

if not exist "%ACTIVATE_DIR%" mkdir "%ACTIVATE_DIR%"
if not exist "%DEACTIVATE_DIR%" mkdir "%DEACTIVATE_DIR%"

> "%ACTIVATE_DIR%\set_path.bat" (
  echo @echo off
  echo set "_OLD_PYTHONPATH=%%PYTHONPATH%%"
  echo if not "%%PYTHONPATH%%"=="" ^(
  echo   set "PYTHONPATH=%PROJECT_ROOT%;%%PYTHONPATH%%"
  echo ^) else ^(
  echo   set "PYTHONPATH=%PROJECT_ROOT%"
  echo ^)
)

> "%DEACTIVATE_DIR%\unset_path.bat" (
  echo @echo off
  echo set "PYTHONPATH=%%_OLD_PYTHONPATH%%"
  echo set "_OLD_PYTHONPATH="
)

echo Wrote:
echo   %ACTIVATE_DIR%\set_path.bat
echo   %DEACTIVATE_DIR%\unset_path.bat

endlocal
