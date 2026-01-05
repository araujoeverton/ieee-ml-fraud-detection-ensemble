#!/bin/bash

# =================================================================
# DVC Remote Configuration Script
# This script automates the environment setup and runs the
# Python configuration for GDrive storage.
# =================================================================

echo "----------------------------------------------------"
echo "🚀 Starting DVC Remote Configuration..."
echo "----------------------------------------------------"

# 1. Detect and activate Virtual Environment (venv)
# Check for both Windows (Scripts) and Linux/Mac (bin) folders
if [ -d "venv" ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "📂 Windows detected. Activating venv..."
        source venv/Scripts/activate
    else
        echo "📂 Linux/Mac detected. Activating venv..."
        source venv/bin/activate
    fi
else
    echo "⚠️ Warning: 'venv' folder not found. Running with global python..."
fi

# 2. Ensure dependencies are installed (optional but safe)
echo "📦 Checking for required dependencies (python-dotenv)..."
pip install -q python-dotenv dvc dvc-gdrive

# 3. Run the Python logic to handle secrets and DVC
echo "🐍 Executing Python setup script..."
python setup/setup_remote.py

# 4. Final check and push attempt
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------"
    echo "✅ Configuration finished successfully!"
    echo "💡 You can now run 'dvc push' to upload your data."
    echo "----------------------------------------------------"
else
    echo "----------------------------------------------------"
    echo "❌ ERROR: Configuration failed."
    echo "🔍 Please check if your .env file exists and has the correct ID."
    echo "----------------------------------------------------"
fi

# 5. Keep window open if running by double-click (Windows)
read -p "Press [Enter] to close this window..."