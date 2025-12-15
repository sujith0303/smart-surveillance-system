#!/bin/bash

echo " Smart Surveillance System - Quick Demo"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo " Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Start API server
echo " Starting API server..."
echo " API will be available at: http://localhost:8000"
echo " Interactive docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python src/api/main.py
