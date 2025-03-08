@echo off
setlocal enabledelayedexpansion

:: Admin check
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process -Verb RunAs -FilePath '%COMSPEC%' -ArgumentList '/c cd /d ""%~dp0"" && %~0'"
    exit /b
)

:: Install Chocolatey once
where choco >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" || exit /b
    set "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
)

:: Install Git
choco install -y git || (
    echo Failed to install Git!
    exit /b
)

:: Clone repo
if not exist "wallet_finder" (
    git clone https://github.com/saoxdxd2/wallet_finder.git || (
        echo Failed to clone repository!
        exit /b
    )
)

:: Navigate to directory
cd wallet_finder\finder || (
    echo Could not find finder directory!
    exit /b
)

:: Run installer
call install_win.bat || (
    echo Failed to run installer!
    exit /b
)

pause
