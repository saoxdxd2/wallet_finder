@echo off
setlocal enabledelayedexpansion

:: Check for admin rights
NET FILE 1>NUL 2>NUL || (
    echo Error: Please run this script as Administrator!
    pause
    exit /b 1
)

:: Set paths
set "VS_INSTALL_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VC_TOOLS_PATH=%VS_INSTALL_PATH%\VC\Tools\MSVC"
set "WIN_SDK_PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"

:: Install Chocolatey if missing
if not exist "C:\ProgramData\chocolatey\choco.exe" (
    echo Installing Chocolatey...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" || (
        echo Chocolatey installation failed!
        exit /b 1
    )
)

:: Install core dependencies
echo Installing base requirements...
choco install -y golang python rust visualstudio2022community || (
    echo Base package installation failed!
    exit /b 1
)

:: Install Visual Studio Build Components
echo Installing C++ Build Tools...
choco install -y visualstudio2022buildtools --params="--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --norestart --passive --locale en-US" || (
    echo Build Tools installation failed!
    exit /b 1
)

:: Find latest MSVC version
set "MSVC_VER="
for /d %%i in ("%VC_TOOLS_PATH%\*") do set "MSVC_VER=%%~ni"
if "!MSVC_VER!"=="" (
    echo MSVC version not found!
    exit /b 1
)

:: Configure environment
set "BIN_PATH=%VC_TOOLS_PATH%\!MSVC_VER!\bin\Hostx64\x64"
setx PATH "%BIN_PATH%;%WIN_SDK_PATH%;%PATH%" /M || (
    echo PATH configuration failed!
    exit /b 1
)

:: Refresh environment
call refreshenv || (
    echo Environment refresh failed!
    exit /b 1
)

:: Verify installations
where go || (echo Go install failed! && exit /b 1)
where python || (echo Python install failed! && exit /b 1)
where cl || (echo C++ compiler not found! && exit /b 1)

:: Python environment setup
echo Configuring Python environment...
python -m venv venv || (
    echo Virtual environment creation failed!
    exit /b 1
)
call venv\Scripts\activate.bat || (
    echo Virtual environment activation failed!
    exit /b 1
)

:: Install Python requirements
echo Installing Python packages...
python -m pip install --upgrade pip || (
    echo Pip upgrade failed!
    exit /b 1
)
python -m pip install aiohttp aiolimiter bip-utils || (
    echo Package installation failed!
    exit /b 1
)
if exist requirements.txt (
    python -m pip install -r requirements.txt || (
        echo Requirements file installation failed!
        exit /b 1
    )
)

:: Go module initialization
echo Initializing Go modules...
if not exist go.mod (
    go mod init app || (
        echo Go mod init failed!
        exit /b 1
    )
)
go get github.com/tyler-smith/go-bip39 || (
    echo Go package installation failed!
    exit /b 1
)
go mod tidy || (
    echo Go mod tidy failed!
    exit /b 1
)

:: Final verification
echo Final system check...
cl || (
    echo C++ compiler verification failed!
    exit /b 1
)
go version || (
    echo Go verification failed!
    exit /b 1
)
python --version || (
    echo Python verification failed!
    exit /b 1
)

:: Run application
echo Starting application...
python app2.py || (
    echo Application failed to start!
    exit /b 1
)

endlocal
echo Installation and setup completed successfully!
pause
