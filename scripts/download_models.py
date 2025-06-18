#!/usr/bin/env python3
"""
Download and cache required models
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",  # Open-access base model
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, open model for testing
}

def download_model(model_name: str, model_id: str, cache_dir: str):
    """Download and cache a model"""
    print(f"Downloading {model_name}...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer downloaded for {model_name}")
        
        # Download model (quantized version if available)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print(f"✓ Model downloaded for {model_name}")
        
        # Clear memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

def main():
    """Main download function"""
    cache_dir = os.getenv("MODEL_CACHE_DIR", "./data/models")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print("Downloading models...")
    print(f"Cache directory: {cache_dir}")
    
    model_name = os.getenv("MODEL_NAME", "mistral-7b-instruct")
    
    if model_name in MODELS:
        download_model(model_name, MODELS[model_name], cache_dir)
    else:
        print(f"Available models: {list(MODELS.keys())}")
        for name, model_id in MODELS.items():
            download_model(name, model_id, cache_dir)
    
    print("✓ Model download complete!")

if __name__ == "__main__":
    main()