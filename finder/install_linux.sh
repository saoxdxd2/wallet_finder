#!/bin/bash

# Detect package manager
if command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
else
    echo "Unsupported package manager"
    exit 1
fi

# Install dependencies
echo "Installing system dependencies..."
sudo $PKG_MANAGER update -y
sudo $PKG_MANAGER install -y golang python3 python3-pip git

# Install Python Tkinter
if [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt install -y python3-tk
elif [ "$PKG_MANAGER" = "dnf" ]; then
    sudo dnf install -y python3-tkinter
fi

# Set up Go environment
export GOPATH=$HOME/go
export PATH=$PATH:/usr/local/go/bin:$GOPATH/bin
mkdir -p $GOPATH

# Install Python packages
echo "Installing Python dependencies..."
pip3 install aiohttp aiolimiter bip-utils

# Install Go packages
echo "Installing Go dependencies..."
go get github.com/tyler-smith/go-bip39

# Run application
echo "Starting application..."
python3 app2.py