@echo off
chcp 65001 >nul
call "%~dp0配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
cmd /k
