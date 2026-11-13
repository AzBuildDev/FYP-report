@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
python "脚本\程序模块\微调预训练模型.py" --dataset-root "%PROJECT_ROOT%\..\训练数据采集\机械臂控制项目\任务演示\单相机演示_20261005_205636_902311" %*
set "VERIFY_RESULT=%ERRORLEVEL%"
pause
exit /b %VERIFY_RESULT%
