"""
Application configuration settings using pydantic.BaseSettings
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic import BaseSettings

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
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "ocr_languages":
                # Handle OCR_LANGUAGES specially
                if not raw_val:
                    return []
                if raw_val.startswith('[') and raw_val.endswith(']'):
                    # If it's a JSON array, parse it
                    import json
                    try:
                        return json.loads(raw_val)
                    except json.JSONDecodeError:
                        pass
                # Handle as comma-separated string
                return [lang.strip('"\' ') for lang in raw_val.split(',') if lang.strip()]
            return raw_val
    # No duplicate parse_env_var method needed here - it's defined in the Config class

    def __init__(self, **data):
        print("\n=== Environment Variables ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Environment file: {os.path.abspath('.env')}")
        print(f"Environment file exists: {os.path.exists('.env')}")
        print(f"OCR_LANGUAGES env var: {os.environ.get('OCR_LANGUAGES')}")
        print("===========================\n")
        super().__init__(**data)

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
    ocr_engine: str = Field(
        default="easyocr",
        env="OCR_ENGINE",
        description="OCR engine to use (easyocr, paddleocr)"
    )
    ocr_languages: List[str] = Field(
        default=["en", "de", "et"],
        env="OCR_LANGUAGES",
        description="Languages to support for OCR. Can be a JSON array (e.g., [\"en\", \"de\"]) or comma-separated (e.g., en,de)."
    )
    
    @validator("ocr_languages", pre=True)
    def parse_languages(cls, v):
        """Parse languages from JSON array, comma-separated string, or use default."""
        if v is None:
            return ["en", "de", "et"]
            
        if isinstance(v, list):
            return v
            
        # Handle string values
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["en", "de", "et"]
            
            # Try to parse as JSON first
            if (v.startswith('[') and v.endswith(']')) or (v.startswith('"') and v.endswith('"')):
                import json
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, str):
                        v = parsed  # Continue processing as string
                except json.JSONDecodeError:
                    pass
            
            # Handle comma-separated string
            return [lang.strip('"\' ') for lang in v.split(',') if lang.strip()]
            
        # Default fallback
        return ["en", "de", "et"]

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
