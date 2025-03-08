@echo off
setlocal enabledelayedexpansion

:: Check for Chocolatey
where choco > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
)

:: Install dependencies
echo Installing Go, Python, Rust...
choco install -y golang python rust
choco install visualstudio2022-build-tools -y --params="--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

:: Refresh environment
call refreshenv

:: Verify tools
where go || (echo Go install failed! && exit /b 1)
where python || (echo Python install failed! && exit /b 1)

:: Create and activate venv
python -m venv venv
call venv\Scripts\activate.bat  <-- Windows activation

:: Install Python packages
echo Installing Python dependencies...
pip install aiohttp aiolimiter bip-utils
if exist requirement.txt pip install -r requirement.txt

:: Install Go packages
echo Installing Go dependencies...
go mod init app
go get github.com/tyler-smith/go-bip39
go mod tidy

:: Run app
echo Starting application...
python app2.py
