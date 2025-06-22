"""
Application configuration settings using pydantic.BaseSettings
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# Load environment variables from .env file
load_dotenv()


class AppSettings(BaseSettings):
    """Application settings with environment variable support"""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            # This ensures environment variables take precedence over .env file
            return env_settings, init_settings, file_secret_settings

    # Basic settings
    app_name: str = "European Invoice OCR"
    debug: bool = False
    environment: str = "production"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8005
    workers: int = 1

    # GPU settings
    cuda_visible_devices: str = "0"
    gpu_memory_utilization: float = 0.9
    max_model_length: int = 4096

    # Model settings - this will be read from MODEL_NAME environment variable
    model_name: str = Field("facebook/opt-125m", env="MODEL_NAME")

    # Other model settings
    quantization: str = "awq"
    model_cache_dir: str = "./data/models"

    # OCR settings
    ocr_engine: str = Field(default="easyocr", description="OCR engine to use (easyocr, paddleocr)")
    ocr_languages: List[str] = Field(default=["en", "de", "et"], description="Languages to support for OCR")

    # Processing settings
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Maximum file size in bytes (50MB)")
    max_batch_size: int = Field(default=50, description="Maximum batch size for processing")
    temp_dir: str = Field(default="./data/temp", description="Temporary directory")
    output_dir: str = Field(default="./data/output", description="Output directory")

    # Database settings
    redis_url: str = Field(default="redis://localhost:6380/0", description="Redis connection URL")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")

    @validator("gpu_memory_utilization", pre=True)
    def parse_float(cls, v):
        """Parse float values from strings"""
        if isinstance(v, str):
            return float(v)
        return v

    @validator("model_cache_dir", "temp_dir", "output_dir")
    def create_directories(cls, v):
        """Ensure output directories exist"""
        if v:
            os.makedirs(v, exist_ok=True)
        return v
