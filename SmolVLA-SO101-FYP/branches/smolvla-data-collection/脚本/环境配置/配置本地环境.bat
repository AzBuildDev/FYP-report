@echo off
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "LEROBOT_ROOT=%PROJECT_ROOT%\上游框架"
set "PATH=%PROJECT_ROOT%\运行环境\虚拟环境\Scripts;%PATH%"
set "VIRTUAL_ENV=%PROJECT_ROOT%\运行环境\虚拟环境"
set "PYTHONPATH=%LEROBOT_ROOT%\src"
set "PYTHONPYCACHEPREFIX=%PROJECT_ROOT%\运行环境\缓存\字节码"
set "HF_HOME=%PROJECT_ROOT%\运行环境\缓存\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\运行环境\缓存\torch"
set "YOLO_CONFIG_DIR=%PROJECT_ROOT%\运行环境\缓存\ultralytics"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\标定备份"
set "TMP=%PROJECT_ROOT%\运行环境\临时文件"
set "TEMP=%PROJECT_ROOT%\运行环境\临时文件"
if not exist "%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\python.exe" exit /b 1
for %%D in ("%TMP%" "%YOLO_CONFIG_DIR%\Ultralytics" "%HF_HOME%" "%TORCH_HOME%") do if not exist "%%~D" mkdir "%%~D"
call "%PROJECT_ROOT%\脚本\环境配置\配置硬件端口.bat"
exit /b 0
