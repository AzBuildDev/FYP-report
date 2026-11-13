@echo off
chcp 65001 >nul
setlocal
call "%~dp0环境配置\配置本地环境.bat"
if errorlevel 1 (
  echo 项目环境未找到，请先查看本地部署说明。
  pause
  exit /b 1
)
start "" "%PYTHONW_EXE%" "%~dp0项目控制台.py"
