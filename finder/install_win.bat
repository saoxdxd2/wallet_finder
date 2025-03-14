@echo off
setlocal enabledelayedexpansion

:: Install dependencies
echo Installing Go, Python, Rust...
choco install -y golang python rust
choco install visualstudio2022community -y
choco install microsoft-visual-cpp-build-tools
choco install visualstudio2022buildtools -y --params="--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

:: Refresh environment
call refreshenv

:: Verify tools
where go || (echo Go install failed! && exit /b 1)
where python || (echo Python install failed! && exit /b 1)

:: Create and activate venv
python -m venv venv
call venv\Scripts\activate.bat

:: Verify venv activation
where python | find "venv" >nul || (
    echo Virtual environment activation failed!
    exit /b 1
)

:: Install Python packages
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install aiohttp aiolimiter bip-utils
if exist requirements.txt python -m pip install -r requirements.txt

:: Initialize Go modules (only if missing)
if not exist go.mod (
    go mod init app
    go get github.com/tyler-smith/go-bip39
    go mod tidy
)

:: Run app
echo Starting application...
python app2.py
