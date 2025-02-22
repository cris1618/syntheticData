#!/bin/bash

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Debugging: Print Rust and Cargo versions
echo "Checking Rust version..."
rustc --version
cargo --version

# Install dependencies
pip install --no-cache-dir -r requirements.txt
