#!/usr/bin/env python3
"""
Poetry Setup Script for Python 3.13 Compatibility

This script helps set up a Poetry environment for the project with special handling
for packages that have compatibility issues with Python 3.13.
"""

import os
import subprocess
import sys
import re
from pathlib import Path

# Check Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"Python version: {python_version}")

if sys.version_info < (3, 9):
    print("Error: Python 3.9 or higher is required")
    sys.exit(1)

# Define packages that need special handling with Python 3.13
SPECIAL_PACKAGES = {
    "pymupdf": "1.26.1",
    "pillow": "11.2.1",
    "numpy": "1.24.3",
    "pandas": "2.0.3",
    "pydantic": "1.10.13",
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "vllm": "0.2.2",
    "paddleocr": "2.7.0",
    "easyocr": "1.7.0",
    "transformers": "4.35.0",
    "bitsandbytes": "0.41.2.post2",
    "accelerate": "0.24.1",
}

def run_command(cmd, check=True, silent=False):
    """Run a shell command and return its output"""
    if not silent:
        print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            text=True, 
            capture_output=True
        )
        if not silent and result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"Error: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
        return False, e.stderr

def check_poetry_installed():
    """Check if Poetry is installed"""
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_poetry():
    """Install Poetry"""
    print("Installing Poetry...")
    run_command("curl -sSL https://install.python-poetry.org | python3 -")
    
    # Add Poetry to PATH
    home = Path.home()
    poetry_path = home / ".local" / "bin"
    os.environ["PATH"] = f"{poetry_path}:{os.environ['PATH']}"
    
    # Verify installation
    if check_poetry_installed():
        print("Poetry installed successfully")
        return True
    else:
        print("Failed to install Poetry")
        return False

def create_minimal_pyproject():
    """Create a minimal pyproject.toml file with only the essential dependencies"""
    print("Creating minimal pyproject.toml...")
    
    # Backup existing pyproject.toml if it exists
    if os.path.exists("pyproject.toml"):
        run_command("cp pyproject.toml pyproject.toml.bak")
    
    # Create minimal pyproject.toml
    with open("pyproject.toml", "w") as f:
        f.write("""[tool.poetry]
name = "pocro"
version = "0.1.3"
description = "OCR+LLM stack with GPU optimization, multilingual support, and EU compliance"
authors = ["Tom Sapletta <info@softreck.dev>"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = ">=3.9,<3.14"
# Core dependencies
fastapi = "0.104.1"
uvicorn = {extras = ["standard"], version = "0.24.0"}
python-multipart = "0.0.6"

# Database and Storage
sqlalchemy = "2.0.23"
alembic = "1.12.1"
redis = "5.0.1"

# Utilities
python-dotenv = "1.0.0"
click = "8.1.7"
tqdm = "4.66.1"
psutil = "5.9.6"

# Monitoring and Logging
prometheus-client = "0.19.0"
loguru = "0.7.2"

[tool.poetry.group.dev.dependencies]
# Testing
pytest = "7.4.3"
pytest-asyncio = "0.21.1"
pytest-cov = "4.1.0"
pytest-mock = "3.12.0"
httpx = "0.25.2"

# Code Quality
black = "23.11.0"
isort = "5.12.0"
flake8 = "6.1.0"
mypy = "1.7.1"
pre-commit = "3.5.0"

# Documentation
mkdocs = "1.5.3"
mkdocs-material = "9.4.8"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""")
    
    print("Minimal pyproject.toml created")

def setup_poetry_environment():
    """Set up Poetry environment"""
    print("Setting up Poetry environment...")
    
    # Configure Poetry to create virtual environments in the project directory
    run_command("poetry config virtualenvs.in-project true")
    
    # Remove existing virtual environment if it exists
    if os.path.exists(".venv"):
        print("Removing existing .venv directory...")
        run_command("rm -rf .venv")
    
    # Create a new Poetry environment
    print("Creating a new Poetry environment...")
    run_command("poetry env use python3")
    
    # Install dependencies with Poetry
    print("Installing core dependencies with Poetry...")
    run_command("poetry install --no-interaction")

def install_special_packages():
    """Install packages that need special handling"""
    print("\nInstalling packages with special handling for Python 3.13...")
    
    for package, version in SPECIAL_PACKAGES.items():
        print(f"\nInstalling {package} {version}...")
        
        # Try binary installation first
        success, _ = run_command(f"poetry run pip install --no-cache-dir --only-binary=:all: {package}=={version}")
        
        if not success:
            # Fall back to source installation
            print(f"Binary installation failed, trying source installation for {package}...")
            success, _ = run_command(f"poetry run pip install --no-cache-dir {package}=={version}")
        
        if not success:
            # Try without version constraint
            print(f"Installation with version constraint failed, trying latest version of {package}...")
            success, _ = run_command(f"poetry run pip install --no-cache-dir {package}")
        
        if not success:
            print(f"Failed to install {package}")

def verify_installation():
    """Verify installation of key packages"""
    print("\nVerifying installation...")
    
    # Create verification script
    verify_script = """
import sys
import importlib.util

def check_package(package_name):
    try:
        if package_name.lower() == 'pymupdf':
            import fitz
            return True, fitz.__version__ if hasattr(fitz, '__version__') else 'unknown'
        else:
            module = importlib.import_module(package_name.lower())
            version = getattr(module, '__version__', 'unknown')
            return True, version
    except ImportError:
        return False, None

# Check key packages
packages = ['fastapi', 'pymupdf', 'pillow', 'numpy', 'pandas', 'torch', 'pydantic']
all_ok = True

print(f'Python version: {sys.version}')
for package in packages:
    success, version = check_package(package)
    if success:
        print(f'✅ {package}: {version}')
    else:
        print(f'❌ {package}: Not found')
        all_ok = False

sys.exit(0 if all_ok else 1)
"""
    
    with open("verify_installation.py", "w") as f:
        f.write(verify_script)
    
    # Run verification script
    run_command("poetry run python verify_installation.py")
    
    # Clean up
    os.remove("verify_installation.py")

def update_requirements_txt():
    """Update requirements.txt to match installed packages"""
    print("\nUpdating requirements.txt...")
    
    # Get list of installed packages
    success, output = run_command("poetry run pip freeze", silent=True)
    if not success:
        print("Failed to get list of installed packages")
        return
    
    # Write to requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("# Generated by poetry_setup.py\n")
        f.write("# This file contains all packages installed in the Poetry environment\n\n")
        f.write(output)
    
    print("requirements.txt updated")

def main():
    """Main function"""
    print("Poetry Setup for Python 3.13 Compatibility")
    print("=========================================")
    
    # Check if Poetry is installed
    if not check_poetry_installed():
        if not install_poetry():
            print("Error: Failed to install Poetry")
            sys.exit(1)
    
    # Create minimal pyproject.toml
    create_minimal_pyproject()
    
    # Set up Poetry environment
    setup_poetry_environment()
    
    # Install special packages
    install_special_packages()
    
    # Verify installation
    verify_installation()
    
    # Update requirements.txt
    update_requirements_txt()
    
    print("\nPoetry setup completed!")
    print("\nTo activate the Poetry environment, run:")
    print("  poetry shell")
    print("\nTo run commands with Poetry:")
    print("  poetry run python your_script.py")

if __name__ == "__main__":
    main()
