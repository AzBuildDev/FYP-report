@echo off
chcp 65001 >nul
title 机械臂正常同步测试
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "脚本\程序模块\机械臂同步测试.py" --mode=normal --wait-for-start --follower-port=%FOLLOWER_PORT% --leader-port=%LEADER_PORT% --duration=30 %*
set "TEST_RESULT=%ERRORLEVEL%"
pause
exit /b %TEST_RESULT%
