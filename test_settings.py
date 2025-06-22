"""Test script to verify settings import and usage"""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.config.settings import AppSettings

def main():
    try:
        print("Testing settings import...")
        settings = AppSettings()
        print("Successfully created AppSettings instance!")
        print(f"Environment: {settings.model_config}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
