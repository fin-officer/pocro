#!/bin/bash
set -e

echo "[pocro] Universal setup script - Linux distribution autodetect"

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    DISTRO="unknown"
fi

echo "Detected distribution: $DISTRO"

# Install system dependencies by distro
case "$DISTRO" in
    ubuntu|debian)
        echo "Installing dependencies for Debian/Ubuntu..."
        sudo apt-get update
        sudo apt-get install -y build-essential python3-dev python3-pip \
            tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng tesseract-ocr-est \
            poppler-utils ghostscript pkg-config libmupdf-dev
        ;;
    fedora)
        echo "Installing dependencies for Fedora..."
        sudo dnf install -y @development-tools python3-devel python3-pip \
            tesseract tesseract-langpack-deu tesseract-langpack-eng tesseract-langpack-est \
            poppler-utils ghostscript mupdf mupdf-tools
        ;;
    arch)
        echo "Installing dependencies for Arch Linux..."
        sudo pacman -Sy --noconfirm base-devel python-pip python \
            tesseract tesseract-data-deu tesseract-data-eng tesseract-data-est \
            poppler ghostscript mupdf
        ;;
    *)
        echo "[WARN] Unknown or unsupported distribution. Please install dependencies manually:"
        echo "  build tools, python3-dev, pip, tesseract-ocr (+deu/eng/est), poppler-utils, ghostscript, mupdf"
        ;;
esac

echo "Creating directories..."
mkdir -p data/models data/temp data/output logs

# Install Python dependencies
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    [ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true
else
    echo "requirements.txt not found!"
fi

# Download models
if [ -f scripts/download_models.py ]; then
    echo "Downloading models..."
    python3 scripts/download_models.py
fi

# Validate installation
if [ -f scripts/validate_installation.py ]; then
    echo "Validating installation..."
    python3 scripts/validate_installation.py || true
fi

echo "[pocro] Setup complete!"
