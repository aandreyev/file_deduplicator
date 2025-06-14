
# chmod +x setup_venv.sh
# Usage: ./setup_venv.sh
# This script sets up a Python virtual environment and installs dependencies from requirements.txt

#!/bin/bash

# Define the virtual environment directory name
VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "Creating virtual environment '$VENV_DIR'..."
    # Create the virtual environment using Python 3's venv module
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
fi

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt into $VENV_DIR..."
    # Upgrade pip first using the venv pip
    echo "Upgrading pip in $VENV_DIR..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    if [ $? -ne 0 ]; then
        echo "Error: Failed to upgrade pip in virtual environment."
        exit 1
    fi
    
    # Install requirements using the venv pip
    echo "Installing requirements..."
    "$VENV_DIR/bin/pip" install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        exit 1
    fi
    echo "Dependencies installed successfully."
else
    echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

echo ""
echo "Setup complete."
echo "To activate the virtual environment in your current shell, run:"
echo "source $VENV_DIR/bin/activate" 