@echo off
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "LEROBOT_ROOT=%PROJECT_ROOT%\上游框架"
set "PROJECT_ENV=%PROJECT_ROOT%\运行环境\虚拟环境"
set "LOCAL_PYTHON_FILE=%PROJECT_ROOT%\运行环境\本机Python路径.txt"
set "PYTHON_EXE="
if defined LEROBOT_PYTHON set "PYTHON_EXE=%LEROBOT_PYTHON%"
if not defined PYTHON_EXE if exist "%LOCAL_PYTHON_FILE%" set /p "PYTHON_EXE=" < "%LOCAL_PYTHON_FILE%"
if not defined PYTHON_EXE if exist "%PROJECT_ENV%\Scripts\python.exe" (
  set "PYTHON_EXE=%PROJECT_ENV%\Scripts\python.exe"
)
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
  echo 未找到 Python。请配置运行环境\本机Python路径.txt 或激活项目环境。
  exit /b 1
)
if not exist "%PYTHON_EXE%" (
  echo Python 路径无效，请检查 LEROBOT_PYTHON 或运行环境\本机Python路径.txt。
  exit /b 1
)
for %%I in ("%PYTHON_EXE%") do set "PYTHON_ENV=%%~dpI"
for %%I in ("%PYTHON_ENV%.") do set "PYTHON_PREFIX=%%~fI"
set "PYTHONW_EXE=%PYTHON_ENV%pythonw.exe"
if not exist "%PYTHONW_EXE%" set "PYTHONW_EXE=%PYTHON_EXE%"
rem venv 与 Conda 均使用同一解释器；Conda 的 DLL 和辅助程序位于 Library 与 Scripts。
set "PATH=%PYTHON_ENV%;%PYTHON_PREFIX%\Library\mingw-w64\bin;%PYTHON_PREFIX%\Library\usr\bin;%PYTHON_PREFIX%\Library\bin;%PYTHON_PREFIX%\Scripts;%PATH%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%LEROBOT_ROOT%\src"
set "PYTHONPYCACHEPREFIX=%PROJECT_ROOT%\运行环境\缓存\字节码"
set "HF_HOME=%PROJECT_ROOT%\运行环境\缓存\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\运行环境\缓存\torch"
set "YOLO_CONFIG_DIR=%PROJECT_ROOT%\运行环境\缓存\ultralytics"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\标定备份"
set "TMP=%PROJECT_ROOT%\运行环境\临时文件"
set "TEMP=%PROJECT_ROOT%\运行环境\临时文件"
for %%D in ("%TMP%" "%YOLO_CONFIG_DIR%\Ultralytics" "%HF_HOME%" "%TORCH_HOME%") do if not exist "%%~D" mkdir "%%~D"
call "%PROJECT_ROOT%\脚本\环境配置\配置硬件端口.bat"
"%PYTHON_EXE%" -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if errorlevel 1 (
  echo Python 无法启动或版本低于 3.10，请检查项目环境。
  exit /b 1
)
exit /b 0
