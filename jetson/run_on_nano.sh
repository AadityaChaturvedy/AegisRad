#!/bin/bash
set -e

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "=== AegisRad Jetson Nano Setup & Run Script ==="

# 1. Create the virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "[*] Virtual environment 'venv' already exists."
fi

# 2. Activate the virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# 3. Install/Update requirements
echo "[*] Installing required packages from requirements_nano.txt..."
pip install --upgrade "pip<21.3.2"
pip install -r requirements_nano.txt

# 4. Start the application
echo "[*] Starting the application (app.py)..."
python3 app.py
