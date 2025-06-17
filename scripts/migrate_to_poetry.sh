#!/bin/bash
set -e

echo "[pocro] Migration to Poetry dependency management"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry installation
poetry --version

# Configure Poetry to create virtual environments in the project directory
poetry config virtualenvs.in-project true

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create a new Poetry environment
echo "Creating a new Poetry environment..."
poetry env use python3

# Install core dependencies first (excluding problematic packages)
echo "Installing core dependencies with Poetry..."
# Temporarily remove problematic packages from pyproject.toml
cp pyproject.toml pyproject.toml.bak
python3 -c "
import re
with open('pyproject.toml', 'r') as f:
    content = f.read()
# Comment out problematic packages
content = re.sub(r'^(pymupdf = .*)', '# \\1  # Installed separately', content, flags=re.MULTILINE)
content = re.sub(r'^(paddleocr = .*)', '# \\1  # Installed separately', content, flags=re.MULTILINE)
content = re.sub(r'^(pillow = .*)', '# \\1  # Installed separately', content, flags=re.MULTILINE)
with open('pyproject.toml', 'w') as f:
    f.write(content)
"

# Install dependencies with Poetry (excluding problematic packages)
poetry install --no-interaction

# Restore original pyproject.toml
mv pyproject.toml.bak pyproject.toml

# Install problematic packages separately
echo "Installing PyMuPDF, Pillow, and paddleocr with special handling..."
poetry run python3 -c "
import sys
import subprocess
print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}')

# Install PyMuPDF first
print('Installing PyMuPDF...')
try:
    # Try binary installation first
    subprocess.run(['pip', 'install', '--no-cache-dir', '--only-binary=:all:', 'pymupdf==1.26.1'], check=True)
except subprocess.CalledProcessError:
    # Fall back to source installation
    print('Binary installation failed, trying source installation...')
    subprocess.run(['pip', 'install', '--no-cache-dir', 'pymupdf==1.26.1'], check=True)

# Install Pillow
print('\\nInstalling Pillow...')
try:
    # Try binary installation first
    subprocess.run(['pip', 'install', '--no-cache-dir', '--only-binary=:all:', 'pillow==11.2.1'], check=True)
except subprocess.CalledProcessError:
    # Fall back to source installation
    print('Binary installation failed, trying source installation...')
    subprocess.run(['pip', 'install', '--no-cache-dir', 'pillow==11.2.1'], check=True)

# Install paddleocr with --no-deps to avoid PyMuPDF version conflict
print('\\nInstalling paddleocr...')
subprocess.run(['pip', 'install', '--no-deps', 'paddleocr==2.7.0'], check=True)

# Install paddleocr dependencies manually (excluding PyMuPDF)
print('\\nInstalling paddleocr dependencies...')
paddleocr_deps = [
    'numpy', 'Pillow', 'pyclipper', 'shapely', 'scikit-image',
    'imgaug', 'pyyaml', 'lanms-neo', 'tqdm', 'visualdl', 'rapidfuzz',
    'opencv-python', 'cython', 'lmdb', 'premailer', 'openpyxl', 'attrdict'
]
for dep in paddleocr_deps:
    if dep != 'Pillow':  # Skip Pillow as we've already installed it
        print(f'Installing {dep}...')
        try:
            subprocess.run(['pip', 'install', dep], check=True)
        except subprocess.CalledProcessError:
            print(f'Failed to install {dep}, continuing anyway...')
"

# Verify installation
echo "Verifying installation..."
poetry run python3 -c "
import sys
import importlib.util

def check_package(package_name):
    try:
        if package_name.lower() == 'pymupdf':
            import fitz
            return True
        spec = importlib.util.find_spec(package_name.lower())
        if spec is None:
            return False
        return True
    except ImportError:
        return False

packages = ['fastapi', 'pymupdf', 'pillow', 'torch', 'easyocr', 'paddleocr']
all_ok = True

print(f'Python version: {sys.version}')
for package in packages:
    if check_package(package):
        try:
            if package.lower() == 'pymupdf':
                import fitz
                version = getattr(fitz, '__version__', 'unknown')
                print(f'✅ {package}: {version}')
            else:
                module = __import__(package.lower())
                version = getattr(module, '__version__', 'unknown')
                print(f'✅ {package}: {version}')
        except ImportError as e:
            print(f'❌ {package}: Error importing - {e}')
            all_ok = False
    else:
        print(f'❌ {package}: Not found')
        all_ok = False

sys.exit(0 if all_ok else 1)
"

echo
echo "Migration to Poetry completed!"
echo
echo "To activate the Poetry environment, run:"
echo "  poetry shell"
echo
echo "To run commands with Poetry:"
echo "  poetry run python3 your_script.py"
echo
echo "To add new dependencies:"
echo "  poetry add package_name"
echo
echo "To add development dependencies:"
echo "  poetry add --group dev package_name"
echo
