"""
Application configuration settings
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings"""

    # Basic settings
    app_name: str = "European Invoice OCR"
    debug: bool = False
    environment: str = "production"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # GPU settings
    cuda_visible_devices: str = "0"
    gpu_memory_utilization: float = 0.9
    max_model_length: int = 4096

    # Model settings
    model_name: str = "mistral-7b-instruct"
    quantization: str = "awq"
    model_cache_dir: str = "./data/models"

    # OCR settings
    ocr_engine: str = "easyocr"  # easyocr, paddleocr
    ocr_languages: List[str] = ["en", "de", "et"]

    # Processing settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_batch_size: int = 50
    temp_dir: str = "./data/temp"
    output_dir: str = "./data/output"

    # Database settings
    redis_url: str = "redis://localhost:6379/0"

    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("model_cache_dir", "temp_dir", "output_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @validator("ocr_languages")
    def validate_languages(cls, v):
        """Validate OCR languages"""
        supported = ["en", "de", "et", "fr", "es", "it", "nl"]
        for lang in v:
            if lang not in supported:
                raise ValueError(f"Unsupported language: {lang}")
        return v