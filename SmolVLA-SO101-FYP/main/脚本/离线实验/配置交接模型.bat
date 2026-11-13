@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "脚本\程序模块\配置交接模型.py" %*
set "CONFIGURE_RESULT=%ERRORLEVEL%"
pause
exit /b %CONFIGURE_RESULT%
