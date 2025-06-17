#!/bin/bash
set -e

echo "[procr] Universal setup script - Linux distribution autodetect"

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    DISTRO="unknown"
fi

echo "Detected distribution: $DISTRO"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Python 3.13 compatibility warning
if [[ $(echo "$PYTHON_VERSION >= 3.13" | bc) -eq 1 ]]; then
    echo "[WARNING] Python $PYTHON_VERSION detected. Some packages may have compatibility issues with Python 3.13+."
    echo "          Consider using Python 3.11 or 3.12 for better compatibility."
fi

# Ask for confirmation before installing system dependencies
install_system_deps() {
    echo "This script will install system dependencies using sudo."
    read -p "Do you want to continue with system dependency installation? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping system dependency installation."
        return
    fi

    # Install system dependencies by distro
    case "$DISTRO" in
        ubuntu|debian)
            echo "Installing dependencies for Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y build-essential python3-dev python3-pip \
                tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng tesseract-ocr-est \
                poppler-utils ghostscript pkg-config libmupdf-dev \
                swig libffi-dev libjpeg-dev zlib1g-dev
            ;;
        fedora)
            echo "Installing dependencies for Fedora..."
            sudo dnf install -y @development-tools python3-devel python3-pip \
                tesseract tesseract-langpack-deu tesseract-langpack-eng tesseract-langpack-est \
                poppler-utils ghostscript mupdf mupdf-tools mupdf-devel \
                swig libffi-devel libjpeg-turbo-devel zlib-devel
            ;;
        arch)
            echo "Installing dependencies for Arch Linux..."
            sudo pacman -Sy --noconfirm base-devel python-pip python \
                tesseract tesseract-data-deu tesseract-data-eng tesseract-data-est \
                poppler ghostscript mupdf \
                swig libffi libjpeg-turbo zlib
            ;;
        *)
            echo "[WARN] Unknown or unsupported distribution. Please install dependencies manually:"
            echo "  build tools, python3-dev, pip, tesseract-ocr (+deu/eng/est), poppler-utils, ghostscript, mupdf"
            ;;
    esac
}

echo "Creating directories..."
mkdir -p data/models data/temp data/output logs

# Handle problematic package installations separately
handle_problematic_packages() {
    echo "Handling potentially problematic package installations..."
    
    # Uninstall existing packages that might cause issues
    pip uninstall -y pymupdf fitz pillow || true
    
    # Install Pillow with specific version based on Python version
    echo "Installing Pillow..."
    if [[ $(echo "$PYTHON_VERSION >= 3.13" | bc) -eq 1 ]]; then
        # For Python 3.13+, try the latest version
        if ! pip install --no-cache-dir pillow; then
            echo "Trying alternative Pillow version..."
            pip install --no-cache-dir pillow==10.0.0
            # Update requirements.txt if successful
            if [ $? -eq 0 ]; then
                sed -i 's/Pillow==10.1.0/Pillow==10.0.0/' requirements.txt
            fi
        fi
    else
        # For older Python versions, use the specified version
        pip install --no-cache-dir pillow==10.1.0
    fi
    
    # Install PyMuPDF
    echo "Installing PyMuPDF..."
    if [[ $(echo "$PYTHON_VERSION >= 3.13" | bc) -eq 1 ]]; then
        # For Python 3.13+, try the latest version without version constraint
        if pip install --no-cache-dir pymupdf; then
            echo "Successfully installed latest PyMuPDF."
            # Update requirements.txt
            INSTALLED_VERSION=$(pip show pymupdf | grep Version | awk '{print $2}')
            sed -i "s/PyMuPDF==1.23.8/PyMuPDF==$INSTALLED_VERSION/" requirements.txt
        else
            echo "Trying alternative PyMuPDF version..."
            if pip install --no-cache-dir pymupdf==1.22.5; then
                echo "Successfully installed PyMuPDF 1.22.5"
                sed -i 's/PyMuPDF==1.23.8/PyMuPDF==1.22.5/' requirements.txt
            else
                echo "[ERROR] Failed to install PyMuPDF. Please try manually with:"
                echo "pip install --no-cache-dir pymupdf"
                echo "or consider using a different Python version (3.11 or 3.12 recommended)."
                return 1
            fi
        fi
    else
        # For older Python versions, use the specified version
        pip install --no-cache-dir pymupdf==1.23.8
    fi
    
    return 0
}

# Ask user if they want to install system dependencies
install_system_deps

# Install Python dependencies (excluding problematic packages)
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies (excluding problematic packages)..."
    pip install --upgrade pip
    
    # Create a temporary requirements file excluding problematic packages
    grep -v -E "PyMuPDF|Pillow" requirements.txt > /tmp/requirements_filtered.txt
    
    # Install the filtered dependencies
    pip install -r /tmp/requirements_filtered.txt || {
        echo "[WARNING] Some packages failed to install. Continuing with installation..."
    }
    
    # Handle problematic packages separately
    handle_problematic_packages
    
    # Install dev requirements if they exist
    if [ -f requirements-dev.txt ]; then
        echo "Installing development dependencies..."
        pip install -r requirements-dev.txt || echo "Some development dependencies failed to install."
    fi
else
    echo "requirements.txt not found!"
fi

# Download models
if [ -f scripts/download_models.py ]; then
    echo "Downloading models..."
    python3 scripts/download_models.py || echo "Model download failed, but continuing..."
fi

# Validate installation
if [ -f scripts/validate_installation.py ]; then
    echo "Validating installation..."
    python3 scripts/validate_installation.py || echo "Validation failed, but continuing..."
fi

echo "[procr] Setup completed!"
echo
echo "NOTE: If you encountered issues with PyMuPDF or Pillow, consider:"
echo "1. Creating a new virtual environment with Python 3.11 or 3.12"
echo "2. Installing packages manually one by one"
echo "3. Using pre-built wheels instead of building from source"
echo
