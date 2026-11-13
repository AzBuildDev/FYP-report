@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" "%~dp0..\程序模块\四物体采集计划.py" %*
pause
