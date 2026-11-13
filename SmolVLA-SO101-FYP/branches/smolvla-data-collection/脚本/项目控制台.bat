@echo off
chcp 65001 >nul
setlocal
call "%~dp0环境配置\配置本地环境.bat"
if errorlevel 1 (
  echo 项目环境未找到，请先查看本地部署说明。
  pause
  exit /b 1
)
start "" "%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\pythonw.exe" "%~dp0项目控制台.py"
