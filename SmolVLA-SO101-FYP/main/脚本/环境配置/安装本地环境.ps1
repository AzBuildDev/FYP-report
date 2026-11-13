$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$env:TEMP = Join-Path (Get-Location) '运行环境/临时文件'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
if (-not (Test-Path 运行环境/虚拟环境/Scripts/python.exe)) {
    py -3.12 -m venv "运行环境/虚拟环境"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed' }
}
& "运行环境/虚拟环境/Scripts/python.exe" -m pip install --cache-dir "运行环境/缓存/pip" -r "文档/环境依赖.txt"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& "运行环境/虚拟环境/Scripts/python.exe" -m pip install --cache-dir 运行环境/缓存/pip torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA package installation failed' }
& "运行环境/虚拟环境/Scripts/python.exe" -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
