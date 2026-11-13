@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
start "" "%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\pythonw.exe" "%~dp0录制任务演示.py"
