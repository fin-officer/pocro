#!/bin/bash

set -e

echo "Setting up European Invoice OCR environment..."

# Check if running on GPU-enabled system
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi
else
    echo "⚠ No NVIDIA GPU detected. CPU-only mode will be used."
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/models data/temp data/output logs

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-deu \
        tesseract-ocr-est \
        tesseract-ocr-eng
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download models
echo "Downloading models..."
python scripts/download_models.py

# Validate installation
echo "Validating installation..."
python scripts/validate_installation.py

echo "✓ Setup complete! You can now run:"
echo "  make run          # Run locally"
echo "  make docker-run   # Run with Docker"