# Crypto Wallet Scanner

![GUI Screenshot](https://via.placeholder.com/800x500.png?text=Crypto+Scanner+GUI+Preview)

A cross-platform tool for generating and checking cryptocurrency wallet balances with real-time monitoring GUI.

## Features

- 🚀 **High-Performance Generation**
  - Python-based multi-process BIP39 mnemonic generation using `multiprocessing`.
  - State preservation for mnemonic generation index.
- 🧠 **AI-Enhanced Operation (Experimental)**
  - **PPO (Proximal Policy Optimization) Agents:** Both API rate limits and the number of concurrent wallet processing workers are now dynamically adjusted by PPO agents for potentially more stable and efficient learning. (Replaced DQNs).
  - Heuristic-based performance tracking for individual API endpoints (scores logged).
  - (Experimental) Initial framework for ML classification of mnemonics/addresses based on on-chain history (includes feature extraction and model training scripts in `analysis/`).
- 📡 **Advanced Wallet Processing**
  - **Child Address Derivation:** Checks multiple derived child addresses (e.g., first 10) per mnemonic for each supported coin, significantly increasing coverage.
- 🔍 **Expanded Blockchain Scanning**
  - Supports **Bitcoin (BTC), Ethereum (ETH), USDT (ERC20), Litecoin (LTC), and Dogecoin (DOGE)**.
  - Each address checked for both balance and on-chain transaction history (existence).
  - Multiple API endpoint fallbacks with performance tracking.
  - AI-controlled request throttling.
- 🖥️ **Modernized GUI (PySide6 Foundation with React UI Conceptualized)**
  - Backend API (`finder/app.py`) now includes a WebSocket and HTTP server to stream data to a modern UI.
  - The primary GUI (`finder/app2.py`) is now built on PySide6, providing a native desktop experience.
  - Conceptual React components (`frontend/src/`) provided as a template for a potential future web-based UI, capable of displaying detailed child address info, logs, and stats, with basic pause/resume controls.
  - Ethical use disclaimer prompt at GUI startup.
- 🛡️ **Robustness & Security (Basic)**
  - Optional proxy support (SOCKS5/HTTP) for API requests, configurable in `finder/config.py`.
- 📊 **Real-Time GUI Monitoring** (via PySide6 GUI or future React UI)
  - Live updates of checked wallets, including details for multiple child addresses.
  - Display of operational statistics.
  - Log viewing.
- 🔒 **Safety Features**
  - Local data storage for models, logs, and state.
  - Clean shutdown handling for backend processes.

## Installation

There are two main ways to install/run the Crypto Wallet Scanner:

**1. Using PyInstaller Build (Recommended for Windows Users / Simplicity)**

This method uses a pre-built executable (if provided by developers) or allows you to build one using PyInstaller.

*   **Build the Executable (if not provided):**
    1.  Ensure Python 3.9+ and `pip` are installed.
    2.  Install PyInstaller: `pip install pyinstaller`
    3.  Install project dependencies: `pip install -r finder/requirements.txt`
    4.  Navigate to the project root directory in your terminal.
    5.  Run PyInstaller using the spec file: `pyinstaller CryptoWalletScanner.spec`
        *   This will create a `dist/CryptoWalletScanner` folder (one-dir build) or a `dist/CryptoWalletScanner.exe` (one-file build, if spec is modified for it).
*   **Windows Installation (`install.bat`):**
    1.  After building with PyInstaller (or if an executable is provided in `dist/`), run `install.bat` from the project root.
    2.  This script will:
        *   Ask for an installation location (defaults to user's AppData).
        *   Copy the built application to the chosen location.
        *   Create Desktop and Start Menu shortcuts.
    3.  You can then launch the application from these shortcuts.
*   **Linux/macOS (Manual for PyInstaller build):**
    1.  Build using PyInstaller as described above.
    2.  The output will be in the `dist/` folder. You can run the executable directly from `dist/CryptoWalletScanner/CryptoWalletScanner` (for one-dir) or `dist/CryptoWalletScanner` (for one-file).
    3.  You may need to `chmod +x` the executable.
    4.  Manually create shortcuts or move the `dist/CryptoWalletScanner` folder to a preferred location (e.g., `/opt` or `~/Applications`).

**2. Running from Source (Recommended for Linux Developers / Customization)**

This method involves setting up a Python virtual environment and running the application scripts directly.

*   **Prerequisites:**
    *   Python 3.9+
    *   `pip` (Python package installer)
    *   `python3-venv` (or equivalent for your Python distribution, e.g., `python3.9-venv`)
*   **Linux Installation (`finder/install_linux.sh`):**
    1.  Navigate to the `finder` directory: `cd finder`
    2.  Make the script executable: `chmod +x install_linux.sh`
    3.  Run the installer: `./install_linux.sh`
    4.  The script will:
        *   Check for dependencies.
        *   Ask for installation locations (defaults to `~/.local/share/CryptoWalletScanner` for app files and `~/.local/bin` for the launcher).
        *   Create a Python virtual environment.
        *   Install required Python packages into the venv.
        *   Create a launcher script (default: `cryptowalletscanner`) in your chosen binary directory.
        *   Optionally create a `.desktop` file for application menus.
    5.  If `~/.local/bin` (or your chosen bin directory) is in your PATH, you can then run `cryptowalletscanner` from any terminal. Otherwise, run the full path to the launcher.
*   **Windows/macOS (Manual Source Setup):**
    1.  Ensure Python 3.9+ and `pip` are installed.
    2.  Create a virtual environment: `python3 -m venv venv`
    3.  Activate the virtual environment:
        *   Windows: `venv\Scripts\activate`
        *   Linux/macOS: `source venv/bin/activate`
    4.  Install dependencies: `pip install -r finder/requirements.txt`
    5.  Run the application: `python finder/app2.py`

## Requirements
Component	Minimum Version
Python	3.9+ (compatible with PyTorch and PySide6)
RAM	4 GB (8GB+ recommended for AI features and smoother GUI)
Storage	100 MB (more if DQN models become large or many logs are kept)
CPU	Multi-core CPU recommended

## Dependencies
Key Python Packages (see `finder/requirements.txt` for full list and versions):
  - `PySide6`: For the native desktop GUI.
  - `torch`: Core deep learning framework (used by PPO agents).
  - `stable-baselines3`: For PPO reinforcement learning agents.
  - `gymnasium`: Dependency for `stable-baselines3`.
  - `aiohttp`: For asynchronous HTTP requests (API calls, WebSocket server).
  - `aiolimiter`: For rate limiting asynchronous operations.
  - `bip-utils`: For BIP39 mnemonic and BIP44 address derivation.
  - `mnemonic`: For BIP39 mnemonic generation.
  - `pandas`, `joblib`, `scikit-learn`: Used by offline analysis and ML model training scripts.

To install Python dependencies (ideally in a virtual environment):
```bash
pip install -r finder/requirements.txt
```

## Usage

*   **If installed via `install.bat` (Windows):** Use the Desktop or Start Menu shortcut.
*   **If installed via `finder/install_linux.sh` (Linux):** Run the command `cryptowalletscanner` in your terminal (if the launcher directory is in your PATH) or click the application menu icon if created.
*   **If running from source manually:**
    1.  Activate your Python virtual environment.
    2.  Navigate to the project root directory.
    3.  Run `python run_scanner.py`.

The backend logic is primarily in `finder/app.py`, orchestrated by `run_scanner.py`.
PPO models for API rate limiting and processing worker count adjustment are stored in `finder/models/`.
Logs and state files are stored in `finder/logs/` (e.g., `app_main.log`, `checked_wallets.jsonl`, `found_wallets.jsonl`, PPO agent logs).

## Main Features (Recap)

- Real-time generation and checking statistics (via WebSocket to UI).
- Dynamic API rate and worker adjustment using PPO agents.
- Detailed logging of checked and found wallets in JSONL format.
- Auto-rotating log files.
- Support for multiple cryptocurrencies with balance and transaction history checks.
- Child address derivation.
- Optional mnemonic classification scoring.

## Files Overview

- `run_scanner.py`: Main entry point for the application.
- `finder/app.py`: Core application logic: PPO agent control, balance checking, task management, WebSocket server.
- `finder/config.py`: Centralized configuration for all application parameters, including paths, API endpoints, PPO hyperparameters, and logging settings.
- `finder/logger_setup.py`: Module for configuring application-wide logging.
- `finder/api_handler.py`: Handles all external API interactions for balance/history checks, including endpoint scoring and fallbacks.
- `finder/mnemonic_generator.py`: Python-based multi-process BIP39 mnemonic phrase generator and address derivation logic.
- `finder/ppo_sb3_agent.py`: Implements the PPO agent using Stable Baselines3.
- `finder/features.py`: Feature extraction logic for mnemonic classification.
- `finder/task.py`: Defines the `Task` data structure for managing mnemonic processing.
- `finder/requirements.txt`: Python package dependencies.
- `finder/models/`: Directory where trained PPO models are saved/loaded.
- `finder/logs/`: Directory where all log files and state files are stored.
    - `app_main.log`: General application logs.
    - `checked_wallets.jsonl`: Log of all scanned wallets.
    - `found_wallets.jsonl`: Log of wallets with balance/history.
    - `debug_rl_agent.log`: Debug logs for PPO agents.
    - `mnemonic_generator_raw_state.txt`: State for raw mnemonic generation index (if used).
    - `application_processed_mnemonic_state.txt`: State for the application's processed mnemonic index.

## Directory Structure

```
crypto-wallet-scanner/
├── run_scanner.py             # Main application entry point
├── finder/
│   ├── app.py                 # Main application logic (PPO control, checking engine)
│   ├── config.py              # Centralized configuration
│   ├── logger_setup.py        # Logging setup
│   ├── api_handler.py         # API interaction handler
│   ├── mnemonic_generator.py  # Mnemonic generation & address derivation
│   ├── ppo_sb3_agent.py       # PPO agent class (Stable Baselines3)
│   ├── features.py            # Mnemonic feature extraction
│   ├── task.py                # Task data structure
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Stores trained PPO models (e.g., *.zip files)
│   │   └── *.zip
│   └── logs/                  # Stores all logs and state files
│       ├── app_main.log
│       ├── checked_wallets.jsonl
│       ├── found_wallets.jsonl
│       ├── debug_rl_agent.log
│       ├── *.txt              # State files
│       └── ...
│
├── README.md                  # This file
├── install.bat                # Example Windows installer (may need updates)
├── CryptoWalletScanner.spec   # Example PyInstaller spec file (may need updates)
└── ...                        # Other project files (e.g., .gitignore, frontend/)
```

Configuration

- **All major configurations are now centralized in `finder/config.py`**. This includes:
    - Paths for logs, models, and state files.
    - API endpoint URLs, keys (placeholders), and parsing hints.
    - PPO agent hyperparameters (learning rates, network sizes, cycle intervals, reward parameters, etc.).
    - Mnemonic generation settings (strength, language, worker counts).
    - Application performance settings (queue sizes, thread pool sizes).
    - Logging levels and log rotation settings.
- **PPO Agent Models**: Trained models are saved in `finder/models/` (e.g., `rate_limiter_ppo_agent.zip`). Deleting these will cause agents to start training from scratch.
- **State Files**:
    - `finder/logs/application_processed_mnemonic_state.txt`: Stores the index of the last mnemonic processed by the application, allowing resumption.
    - `finder/logs/mnemonic_generator_raw_state.txt`: (If used by specific generator features) stores raw generation index.
- **Logging**: All logs are now managed by `finder/logger_setup.py` and stored in `finder/logs/`.
    - `app_main.log` contains general application status, PPO decisions, and errors.
    - `checked_wallets.jsonl` and `found_wallets.jsonl` store detailed records of wallet checks.

Adjust API endpoints and other parameters directly in `finder/config.py`.

Legal Disclaimer

⚠️ Important: This software is intended for educational purposes only. The developers are not responsible for:

    Any legal consequences of using this software

    Loss of funds or incorrect balance reporting

    API rate limiting or service bans

    Any ethical implications of wallet scanning

Always comply with local regulations and blockchain service terms of use.
