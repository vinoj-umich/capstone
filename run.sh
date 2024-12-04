#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install SQLite3 and required libraries
echo "Installing SQLite3 and dependencies..."
sudo apt-get install -y sqlite3 libsqlite3-dev

# Verify the SQLite version (optional)
sqlite3 --version

# Install Python dependencies (if you haven't already done so)
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run the application (Streamlit app)
echo "Starting Streamlit app..."
python -m streamlit run chatbot.py --server.port 8000 --server.address 0.0.0.0