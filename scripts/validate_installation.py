#!/usr/bin/env python3
"""
Validate that all components are properly installed
"""
import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_import(module_name: str, package_name: str = None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({memory:.1f}GB)")
            return True
        else:
            print("⚠ GPU: CUDA not available, using CPU")
            return False
    except ImportError:
        print("✗ GPU: PyTorch not installed")
        return False

def check_system_dependencies():
    """Check system dependencies"""
    import subprocess
    
    commands = {
        "tesseract": ["tesseract", "--version"],
        "poppler": ["pdfinfo", "-v"],
    }
    
    for name, cmd in commands.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {name}")
            else:
                print(f"✗ {name}: not found")
        except FileNotFoundError:
            print(f"✗ {name}: not installed")

def main():
    """Main validation function"""
    print("Validating European Invoice OCR installation...")
    print("-" * 50)
    
    # Check Python packages
    print("Python packages:")
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("easyocr", "EasyOCR"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("pdf2image", "pdf2image"),
        ("pydantic", "Pydantic"),
    ]
    
    all_packages_ok = True
    for module, name in packages:
        if not check_import(module, name):
            all_packages_ok = False
    
    print("\nSystem dependencies:")
    check_system_dependencies()
    
    print("\nGPU availability:")
    gpu_ok = check_gpu()
    
    print("\nModel cache directory:")
    model_dir = Path("./data/models")
    if model_dir.exists():
        print(f"✓ Model cache directory exists: {model_dir}")
    else:
        print(f"⚠ Model cache directory not found: {model_dir}")
    
    print("\n" + "-" * 50)
    if all_packages_ok:
        print("✓ Installation validation complete!")
        if not gpu_ok:
            print("⚠ Note: Running without GPU acceleration")
    else:
        print("✗ Installation has issues. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()