#!/bin/bash

# PhonemeIQ Development Setup Script
# Harvard-LUMS Speech Processing Laboratory

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PhonemeIQ v3.0 - Development Environment Setup            ║"
echo "║  Speech Processing Laboratory                              ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check Python version
echo -e "\n✓ Checking Python version..."
python --version

# Create virtual environment
echo -e "\n✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n✓ Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo -e "\n✓ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\n✓ Installing dependencies..."
pip install -r requirements.txt

# Verify model files
echo -e "\n✓ Checking model files..."
for file in cnn_model.h5 svm_model.pkl scaler.pkl label_encoder.pkl; do
    if [ -f "$file" ]; then
        echo "  ✓ $file found"
    else
        echo "  ✗ WARNING: $file not found"
    fi
done

echo -e "\n╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                             ║"
echo "║                                                              ║"
echo "║  To start the application, run:                             ║"
echo "║  $ streamlit run app_professional.py                        ║"
echo "║                                                              ║"
echo "║  Or use Docker:                                             ║"
echo "║  $ docker-compose up                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
