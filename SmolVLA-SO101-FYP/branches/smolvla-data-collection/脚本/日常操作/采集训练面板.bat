@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
python "脚本\程序模块\采集训练面板.py"
if errorlevel 1 pause
