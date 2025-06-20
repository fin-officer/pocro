#!/usr/bin/env python3
"""
Fix package installation issues with Python 3.13
This script specifically addresses PyMuPDF and Pillow installation issues
"""

import sys
import subprocess
import re
import os

def run_command(cmd, silent=False):
    """Run a shell command and return its output"""
    if not silent:
        print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               text=True, capture_output=True)
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

def update_requirements_file(package_name, new_version):
    """Update the version in requirements.txt"""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Warning: {req_file} not found")
        return
    
    with open(req_file, 'r') as f:
        content = f.read()
    
    # Create pattern to match package with any version
    pattern = rf"{package_name}==[\d\.]+"
    replacement = f"{package_name}=={new_version}"
    
    # Replace the version
    new_content = re.sub(pattern, replacement, content)
    
    # Write back to file
    with open(req_file, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {package_name} to version {new_version} in {req_file}")

def get_python_version():
    """Get the current Python version"""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"{major}.{minor}"

def install_pillow():
    """Install Pillow package"""
    print("\n=== Installing Pillow ===")
    
    # Uninstall existing Pillow
    run_command("pip uninstall -y pillow")
    
    # Try different installation methods
    python_version = get_python_version()
    
    if python_version >= "3.13":
        print("Python 3.13+ detected, trying alternative Pillow installation methods")
        
        # Try installing with binary only (no source builds)
        success, _ = run_command("pip install --no-cache-dir --only-binary=:all: pillow")
        if success:
            # Get installed version
            _, output = run_command("pip show pillow | grep Version", silent=True)
            version = output.strip().split(": ")[1] if output else "unknown"
            update_requirements_file("Pillow", version)
            return True
        
        # Try specific versions that might work with Python 3.13
        versions = ["10.0.0", "9.5.0", "9.0.0"]
        for version in versions:
            print(f"Trying Pillow version {version}...")
            success, _ = run_command(f"pip install --no-cache-dir --only-binary=:all: pillow=={version}")
            if success:
                update_requirements_file("Pillow", version)
                return True
            
            # If binary-only install failed, try with source
            print(f"Binary install failed, trying source install for Pillow {version}...")
            success, _ = run_command(f"pip install --no-cache-dir pillow=={version}")
            if success:
                update_requirements_file("Pillow", version)
                return True
    else:
        # For older Python versions, use the specified version from requirements
        success, _ = run_command("pip install --no-cache-dir pillow==10.1.0")
        if success:
            return True
    
    print("Failed to install Pillow")
    return False

def install_pymupdf():
    """Install PyMuPDF package"""
    print("\n=== Installing PyMuPDF ===")
    
    # Uninstall existing PyMuPDF/fitz
    run_command("pip uninstall -y pymupdf fitz")
    
    # Try different installation methods
    python_version = get_python_version()
    
    if python_version >= "3.13":
        print("Python 3.13+ detected, trying alternative PyMuPDF installation methods")
        
        # Try installing with binary only (no source builds)
        success, _ = run_command("pip install --no-cache-dir --only-binary=:all: pymupdf")
        if success:
            # Get installed version
            _, output = run_command("pip show pymupdf | grep Version", silent=True)
            version = output.strip().split(": ")[1] if output else "unknown"
            update_requirements_file("PyMuPDF", version)
            return True
        
        # Try specific versions that might work with Python 3.13
        versions = ["1.22.5", "1.22.3", "1.21.1"]
        for version in versions:
            print(f"Trying PyMuPDF version {version}...")
            success, _ = run_command(f"pip install --no-cache-dir --only-binary=:all: pymupdf=={version}")
            if success:
                update_requirements_file("PyMuPDF", version)
                return True
            
            # If binary-only install failed, try with source
            print(f"Binary install failed, trying source install for PyMuPDF {version}...")
            success, _ = run_command(f"pip install --no-cache-dir pymupdf=={version}")
            if success:
                update_requirements_file("PyMuPDF", version)
                return True
    else:
        # For older Python versions, use the specified version from requirements
        success, _ = run_command("pip install --no-cache-dir pymupdf==1.23.8")
        if success:
            return True
    
    print("Failed to install PyMuPDF")
    return False

def check_virtual_env():
    """Check if running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def main():
    """Main function"""
    print(f"Python version: {get_python_version()}")
    
    # Check if we're in a virtual environment
    if not check_virtual_env():
        print("\nWARNING: Not running in a virtual environment!")
        print("This script should be run from within a virtual environment.")
        print("Activate your virtual environment first with:")
        print("  source venv/bin/activate  # or the path to your virtual environment")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Install system dependencies if needed
    if sys.platform.startswith('linux'):
        print("\n=== System Dependencies ===")
        print("The following system packages may be needed for building Python packages:")
        
        if os.path.exists("/etc/debian_version"):
            print("For Debian/Ubuntu:")
            print("  sudo apt-get update")
            print("  sudo apt-get install -y build-essential python3-dev libjpeg-dev zlib1g-dev libmupdf-dev swig libffi-dev")
        elif os.path.exists("/etc/fedora-release"):
            print("For Fedora:")
            print("  sudo dnf install -y @development-tools python3-devel libjpeg-turbo-devel zlib-devel mupdf-devel swig libffi-devel")
        elif os.path.exists("/etc/arch-release"):
            print("For Arch Linux:")
            print("  sudo pacman -Sy --noconfirm base-devel libjpeg-turbo zlib mupdf swig libffi")
        else:
            print("For your unknown Linux distribution, please install equivalent packages.")
        
        print("\nPlease install these packages manually if needed.")
    
    # Install problematic packages
    run_command("pip install --upgrade pip")
    pillow_success = install_pillow()
    pymupdf_success = install_pymupdf()
    
    # Summary
    print("\n=== Installation Summary ===")
    print(f"Pillow: {'SUCCESS' if pillow_success else 'FAILED'}")
    print(f"PyMuPDF: {'SUCCESS' if pymupdf_success else 'FAILED'}")
    
    if not (pillow_success and pymupdf_success):
        print("\nSome packages failed to install. Consider the following options:")
        print("1. Try installing packages manually one by one")
        print("2. Use a different Python version (3.11 or 3.12 recommended)")
        print("3. Create a new virtual environment with a compatible Python version")
        print("4. Check for additional system dependencies required for building these packages")
    else:
        print("\nAll packages installed successfully!")

if __name__ == "__main__":
    main()
