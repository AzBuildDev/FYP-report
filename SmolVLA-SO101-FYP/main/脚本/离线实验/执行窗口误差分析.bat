@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "脚本\程序模块\汇总执行窗口评估.py" %*
set "VERIFY_RESULT=%ERRORLEVEL%"
pause
exit /b %VERIFY_RESULT%
