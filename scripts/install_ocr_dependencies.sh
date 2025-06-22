#!/usr/bin/env bash
# Universal OCR/LLM/Invoice Extraction Dependency Installer for Linux
# Installs system and Python dependencies for Tesseract, EasyOCR, PyMuPDF, pdf2image, etc.
# Supports Ubuntu/Debian, Fedora/RHEL/CentOS, Arch, openSUSE, and detects WSL.
set -e

# Languages to install for Tesseract/EasyOCR
OCR_LANGS=(eng deu pol est fra ita spa nld por ron hun swe fin dan nor ces slk slv ell lav lit)

# Helper to join array
function join_by { local IFS="$1"; shift; echo "$*"; }

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    DISTRO=$(uname -s)
fi

# Update and install system dependencies
case $DISTRO in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-pip python3-venv \
            tesseract-ocr tesseract-ocr-all \
            poppler-utils libpoppler-cpp-dev \
            libsm6 libxext6 libxrender-dev \
            libgl1-mesa-glx libglib2.0-0 \
            build-essential pkg-config \
            libleptonica-dev libtesseract-dev \
            imagemagick
        # Install specific language packs (in case tesseract-ocr-all is not available)
        for lang in "${OCR_LANGS[@]}"; do
            sudo apt-get install -y "tesseract-ocr-$lang" || true
        done
        ;;
    fedora|rhel|centos)
        sudo dnf install -y \
            python3 python3-pip python3-virtualenv \
            tesseract \
            poppler-utils leptonica leptonica-devel \
            mesa-libGL libSM libXext libXrender \
            gcc-c++ pkgconfig \
            ImageMagick
        # Fedora/RHEL: Install each language pack individually
        for lang in "${OCR_LANGS[@]}"; do
            sudo dnf install -y "tesseract-langpack-$lang" || true
        done
        ;;
    arch)
        sudo pacman -Sy --noconfirm \
            python python-pip python-virtualenv \
            tesseract tesseract-data \
            poppler leptonica \
            imagemagick
        ;;
    opensuse*|suse)
        sudo zypper refresh
        sudo zypper install -y \
            python3 python3-pip python3-virtualenv \
            tesseract-ocr tesseract-ocr-traineddata-all \
            poppler-tools leptonica-devel \
            ImageMagick
        ;;
    *)
        echo "[WARN] Unknown or unsupported Linux distribution: $DISTRO"
        echo "Please install Tesseract, poppler-utils, leptonica, and language packs manually."
        ;;
esac

# Verify and display installed Tesseract languages
echo "\nTesseract version:" && tesseract --version
if command -v tesseract &>/dev/null; then
    echo "\nInstalled Tesseract languages:"
    tesseract --list-langs || true
    echo "\nIf any required languages are missing, please install them manually."
    echo "For Fedora/RHEL/CentOS, example: sudo dnf install tesseract-langpack-deu tesseract-langpack-pol tesseract-langpack-est"
    echo "For Ubuntu/Debian, example: sudo apt-get install tesseract-ocr-deu tesseract-ocr-pol tesseract-ocr-est"
fi

# Python dependencies (in user venv or system-wide)
pip3 install --upgrade pip setuptools wheel
pip3 install --upgrade \
    fastapi uvicorn python-multipart python-dotenv \
    opencv-python-headless pytesseract pdf2image numpy Pillow \
    easyocr pymupdf pandas aiofiles psutil

# Print Tesseract version and installed languages
echo "\nTesseract version:" && tesseract --version
if command -v tesseract &>/dev/null; then
    echo "\nInstalled Tesseract languages:"
    tesseract --list-langs || true
fi

echo "\nAll OCR and invoice extraction dependencies installed successfully!"
