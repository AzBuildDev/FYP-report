@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
start "" "%PYTHONW_EXE%" "%~dp0..\程序模块\录制任务演示.py" %*
