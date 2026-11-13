@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" "%~dp0..\程序模块\检查演示数据.py" %*
pause
