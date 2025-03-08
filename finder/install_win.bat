@echo off
setlocal enabledelayedexpansion

:: Check for Chocolatey installation
where choco > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Chocolatey package manager...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
)

:: Install Go and Python
echo Installing Go and Python...
choco install -y golang python

:: Refresh environment variables
call refreshenv

:: Verify installations
where go > nul || (
    echo Go installation failed!
    exit /b 1
)
where python > nul || (
    echo Python installation failed!
    exit /b 1
)

:: creating a virtual envirenement
python -m venv venv
source venv/bin/activate 

:: Install Python dependencies
echo Installing Python packages...
pip install aiohttp aiolimiter bip-utils 

:: Install Go dependencies
echo Installing Go packages...
go mod init a 
go get github.com/tyler-smith/go-bip39
go mod tidy

:: Run application
echo Starting application...
python app2.py