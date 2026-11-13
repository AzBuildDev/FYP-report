@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
echo 请先保存演示并关闭录制和预览窗口；本工具只检查摄像头。
"%PYTHON_EXE%" "脚本\程序模块\双路相机采集检查.py" %*
set "CAMERA_RESULT=%ERRORLEVEL%"
pause
exit /b %CAMERA_RESULT%
