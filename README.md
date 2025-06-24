# Crypto Wallet Scanner

![GUI Screenshot](https://via.placeholder.com/800x500.png?text=Crypto+Scanner+GUI+Preview)

A cross-platform tool for generating and checking cryptocurrency wallet balances with real-time monitoring GUI.

## Features

- 🚀 **High-Performance Generation**
  - Python-based multi-process BIP39 mnemonic generation using `multiprocessing`.
  - State preservation for mnemonic generation index.
- 🧠 **AI-Enhanced Operation (Experimental)**
  - Deep Q-Network (DQN) agent for dynamically adjusting API rate limits to optimize throughput and minimize errors.
  - DQN agent for dynamically adjusting the number of concurrent wallet processing workers based on system load and queue status.
  - Heuristic-based performance tracking for API endpoints.
- 🔍 **Blockchain Scanning**
  - Supports Bitcoin, Ethereum, and USDT.
  - Multiple API endpoint fallbacks with performance tracking.
  - DQN-controlled request throttling.
- 📊 **Real-Time GUI Monitoring**
  - Live updates of checked wallets (via `finder/app2.py`).
  - Balance statistics and history
  - Automatic log rotation
  - Cross-platform support (Windows/Linux)

- 🔒 **Safety Features**
  - No external server communication
  - Local storage only
  - Clean shutdown handling

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
Python Packages:
  - `PySide6` (for the graphical user interface)
  - `torch` (PyTorch for DQN agents)
  - `aiohttp` (for asynchronous HTTP requests)
  - `aiolimiter` (for rate limiting async operations)
  - `bip-utils` (for BIP39 mnemonic and BIP44 address derivation)
  - `mnemonic` (for BIP39 mnemonic generation)
  - (See `finder/requirements.txt` for specific versions and other minor dependencies)

To install Python dependencies manually (if not using installer scripts):
```bash
# (Activate your virtual environment first if using one)
pip install -r finder/requirements.txt
```

## Usage

*   **If installed via `install.bat` (Windows):** Use the Desktop or Start Menu shortcut.
*   **If installed via `finder/install_linux.sh` (Linux):** Run the command `cryptowalletscanner` in your terminal (if the launcher directory is in your PATH) or click the application menu icon if created.
*   **If running from source manually:**
    1.  Activate your Python virtual environment.
    2.  Navigate to the project root directory.
    3.  Run `python finder/app2.py`.

The application GUI (`finder/app2.py`) launches the backend logic (`finder/app.py`) as a separate process.
DQN models for rate limiting and worker count adjustment are stored in `finder/models/` relative to where `finder/app.py` is executed. Logs and state files are also typically stored in the `finder` directory.

## Main GUI Features

        Real-time generation statistics

        Balance history timeline

        Exportable results

        Auto-rotating log files

    Files Overview
        finder/app.py             # Main application engine: balance checking, DQN agent control.
        finder/app2.py            # GUI interface (starts app.py).
        finder/mnemonic_generator.py # Python-based mnemonic phrase generator.
        finder/dqn_agent.py       # Core DQN Agent class implementation.
        finder/models/            # Directory where trained DQN models are saved.
        finder/generator_state.txt # State file for the mnemonic generator (tracks index).
        finder/checked.txt        # Log of all scanned wallets and their balances (auto-rotates).
        finder/found.txt          # Log of wallets found with a positive balance.
        finder/requirements.txt   # Python package dependencies.

Directory Structure
Copy

crypto-finder/
├── finder/
│   ├── app.py                     # Main application logic (checker, DQN control)
│   ├── app2.py                    # GUI interface
│   ├── mnemonic_generator.py      # Mnemonic generation logic
│   ├── dqn_agent.py               # DQN agent class
│   ├── requirements.txt           # Python dependencies
│   ├── generator_state.txt        # Mnemonic generator state
│   ├── checked.txt                # Log of checked wallets (rotates)
│   ├── found.txt                  # Log of wallets with balance
│   └── models/                    # Stores trained DQN models
│       ├── rate_limiter_agent_local.pth
│       └── worker_count_agent_local.pth
│
├── install.bat                # Main installer script (may need updates)
├── install_linux.sh           # Linux installer (may need updates for Python-only env)
└── README.md

Configuration

- **API Endpoints**: Can be adjusted in `finder/app.py` within the `API_ENDPOINTS` dictionary.
- **DQN Parameters**: Hyperparameters for the DQN agents (e.g., learning rate, buffer size, network architecture) are defined in `finder/dqn_agent.py` and `finder/app.py` (for state/action sizes, cycle intervals). These may require tuning for optimal performance. Model checkpoints are saved in `finder/models/`.
- **Mnemonic Generator State**: The `finder/generator_state.txt` file stores the last processed mnemonic index, allowing generation to resume. Deleting this file will restart generation from the beginning.
- **Logging**: Application logs, including DQN decisions and API statistics, are output to standard output and can be observed via the GUI's log panel. Wallet check results are in `finder/checked.txt`.

Adjust API endpoints in app.py:
python
Copy

API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        "https://blockchain.info/balance?active={address}",
        # Add custom endpoints here
    ],
    # ... other coin configurations
}

Legal Disclaimer

⚠️ Important: This software is intended for educational purposes only. The developers are not responsible for:

    Any legal consequences of using this software

    Loss of funds or incorrect balance reporting

    API rate limiting or service bans

    Any ethical implications of wallet scanning

Always comply with local regulations and blockchain service terms of use.
