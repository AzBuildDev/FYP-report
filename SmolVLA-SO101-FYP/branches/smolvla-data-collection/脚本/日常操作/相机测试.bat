@echo off
chcp 65001 >nul
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 (
  echo 未找到本地 Python 环境，请先运行安装脚本。
  pause
  exit /b 1
)
cd /d "%PROJECT_ROOT%"
start "" "%PROJECT_ROOT%\运行环境\虚拟环境\Scripts\pythonw.exe" "%PROJECT_ROOT%\脚本\程序模块\相机测试窗口.py"
