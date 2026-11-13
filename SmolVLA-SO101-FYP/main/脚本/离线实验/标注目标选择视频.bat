@echo off
chcp 65001 >nul
setlocal
call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "脚本\程序模块\标注目标选择视频.py" %*
set "ANNOTATION_RESULT=%ERRORLEVEL%"
if not "%ANNOTATION_RESULT%"=="0" pause
exit /b %ANNOTATION_RESULT%
