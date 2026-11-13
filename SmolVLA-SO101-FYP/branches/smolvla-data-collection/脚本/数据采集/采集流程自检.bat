@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
"%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\python.exe" "%~dp0采集数据保护自检.py"
if errorlevel 1 exit /b 1
"%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\python.exe" "%~dp0演示采集流程自检.py"
pause
