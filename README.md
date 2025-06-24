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

### Windows
1. download install.bat from the project (outside the folder)
2. Right-click `install.bat` and "Run as administrator"
3. Follow on-screen prompts
4. Application will launch automatically

### Linux
```bash
chmod +x install_linux.sh
./install_linux.sh

Requirements
Component	Minimum Version
Python	3.9+ (ensure it's a version compatible with PyTorch)
RAM	4 GB (8GB+ recommended for AI features)
Storage	100 MB (more if DQN models become large or many logs are kept)
CPU	Multi-core CPU recommended

Dependencies
Python Packages:
  - `torch` (PyTorch for DQN agents)
  - `aiohttp`
  - `aiolimiter`
  - `bip-utils`
  - `mnemonic` (for mnemonic generation)
  - (See `finder/requirements.txt` for specific versions)

To install Python dependencies:
```bash
pip install -r finder/requirements.txt
```

Usage

    Start the application (which includes the GUI):
    bash
    Copy

    python finder/app2.py

    The application now runs as a single, unified Python process.
    The core logic, including mnemonic generation and balance checking, is in `finder/app.py`.
    DQN models for rate limiting and worker count adjustment are stored in `finder/models/`.

    Main GUI Features

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
