"""
Application configuration settings
"""
import os
from typing import List, Dict, Any, Type, TypeVar, get_type_hints, Optional
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field, validator

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseModel):
    """Base settings class that handles environment variables"""
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert settings to a dictionary"""
        return super().dict(**kwargs)


class AppSettings(Settings):
    """Application settings with environment variable defaults"""
    
    # Basic settings
    app_name: str = Field(default="European Invoice OCR", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="production", description="Runtime environment")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8005, description="Port to run on")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # GPU settings
    cuda_visible_devices: str = Field(default="0", description="CUDA devices to make visible")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization (0-1)")
    max_model_length: int = Field(default=4096, description="Maximum model sequence length")
    
    # Model settings
    model_name: str = Field(default="mistral-7b-instruct", description="Model name")
    quantization: str = Field(default="awq", description="Quantization method")
    model_cache_dir: str = Field(default="./data/models", description="Directory to cache models")
    
    # OCR settings
    ocr_engine: str = Field(default="easyocr", description="OCR engine to use (easyocr, paddleocr)")
    ocr_languages: List[str] = Field(
        default=["en", "de", "et"],
        description="Languages to support for OCR"
    )
    
    # Processing settings
    max_file_size: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum file size in bytes (50MB)"
    )
    max_batch_size: int = Field(default=50, description="Maximum batch size for processing")
    temp_dir: str = Field(default="./data/temp", description="Temporary directory")
    output_dir: str = Field(default="./data/output", description="Output directory")
    
    # Database settings
    redis_url: str = Field(
        default="redis://localhost:6380/0",
        description="Redis connection URL"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @validator('gpu_memory_utilization', pre=True)
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
