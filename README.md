# Crypto Wallet Scanner

![GUI Screenshot](https://via.placeholder.com/800x500.png?text=Crypto+Scanner+GUI+Preview)

A cross-platform tool for generating and checking cryptocurrency wallet balances with real-time monitoring GUI.

## Features

- 🚀 **High-Performance Generation**
  - Multi-threaded BIP39 mnemonic generation
  - State preservation between sessions
  - Batch processing for efficiency

- 🔍 **Blockchain Scanning**
  - Supports Bitcoin, Ethereum, and USDT
  - Multiple API endpoint fallbacks
  - Smart balance checking with request throttling

- 📊 **Real-Time GUI Monitoring**
  - Live updates of checked wallets
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
Go	1.18+
Python	3.9+
RAM	4 GB
Storage	100 MB
Dependencies
Go Packages

    github.com/tyler-smith/go-bip39

bash
Copy

go get github.com/tyler-smith/go-bip39

Python Packages
bash
Copy

pip install aiohttp aiolimiter bip-utils

Usage

    Start the application
    bash
    Copy

    python app2.py

    Main GUI Features

        Real-time generation statistics

        Balance history timeline

        Exportable results

        Auto-rotating log files

    Files Overview

        generator - Core generation executable

        mnemonics.txt - Generated seed phrases

        checked.txt - Scanned wallet records

        found.txt - Wallets with balances

Directory Structure
Copy

crypto-finder/
├── finder/
│   ├── app.go             # Core generator logic
│   ├── app.py             # Balance checker
│   ├── app2.py            # GUI interface
│   ├── go.mod             # Go dependencies
│   ├── generator*         # Linux executable
│   ├── generator.exe      # Windows executable
│   ├── build_win.bat      # Windows build script
│   ├── build_linux.sh     # Linux build script
│   ├── install_win.bat    # Windows installer
│   └── install_linux.sh   # Linux installer

Configuration

Modify these constants in app.go:
go
Copy

const (
    stateFile    = "generator_state.txt"  // Change state file location
    mnemonicsFile = "mnemonics.txt"       // Change output file
)

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
