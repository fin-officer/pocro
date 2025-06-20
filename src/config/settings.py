"""
Application configuration settings
"""
import os
from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict, Field
from typing import List, Optional, Dict, Any, Type, TypeVar, get_type_hints
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Base settings class that handles environment variables"""
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    
    def __init__(self, **data: Any):
        super().__init__(**data)
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert settings to a dictionary"""
        return self.__dict__.copy()


class AppSettings(Settings):
    """Application settings with environment variable defaults"""
    
    # Basic settings
    app_name: str = Field(default="European Invoice OCR", description="Name of the application")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="production", description="Runtime environment (e.g., development, production, staging)")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8005, description="Port to run the server on")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # GPU settings
    cuda_visible_devices: str = Field(default="0", description="CUDA devices to make visible")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization fraction")
    max_model_length: int = Field(default=4096, description="Maximum model context length")
    
    # Model settings
    model_name: str = Field(default="mistral-7b-instruct", description="Name of the model to use")
    quantization: str = Field(default="awq", description="Quantization method (awq, nf4, etc.)")
    model_cache_dir: str = Field(default="./data/models", description="Directory to cache models")
    
    # OCR settings
    ocr_engine: str = Field(default="easyocr", description="OCR engine to use (easyocr, paddleocr)")
    ocr_languages: List[str] = Field(
        default_factory=lambda: ["en", "de", "et"],
        description="Languages to support for OCR"
    )
    
    # Processing settings
    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum file size in bytes"
    )
    max_batch_size: int = Field(default=50, description="Maximum batch size for processing")
    temp_dir: str = Field(default="./data/temp", description="Directory for temporary files")
    output_dir: str = Field(default="./data/output", description="Directory for output files")
    
    # Database settings
    redis_url: str = Field(
        default="redis://localhost:6380/0",
        description="Redis connection URL"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('ocr_languages', mode='before')
    @classmethod
    def parse_ocr_languages(cls, v):
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",") if lang.strip()]
        return v
    
    @field_validator('debug', 'enable_metrics', mode='before')
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 't')
        return bool(v)
    
    @field_validator('port', 'workers', 'max_model_length', 'max_batch_size', mode='before')
    @classmethod
    def parse_int(cls, v):
        if isinstance(v, str):
            return int(v)
        return v
    
    @field_validator('gpu_memory_utilization', mode='before')
    @classmethod
    def parse_float(cls, v):
        if isinstance(v, str):
            return float(v)
        return v
    
    @field_validator("model_cache_dir", "temp_dir", "output_dir")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist"""
        if v:
            os.makedirs(v, exist_ok=True)
        return v
    
    @validator("ocr_languages", pre=True)
    def validate_languages(cls, v):
        """Validate OCR languages"""
        if isinstance(v, str):
            # Handle case where value comes from environment variable as comma-separated string
            v = [lang.strip() for lang in v.split(",") if lang.strip()]
        
        supported = ["en", "de", "et", "fr", "es", "it", "nl"]
        for lang in v:
            if lang not in supported:
                raise ValueError(f"Unsupported language: {lang}")
        return v
        
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
