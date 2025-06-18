"""
File utilities for invoice processing
"""
import os
import tempfile
import aiofiles
from pathlib import Path
from typing import Optional
import logging
import hashlib

from fastapi import UploadFile

logger = logging.getLogger(__name__)


async def save_temp_file(file: UploadFile, temp_dir: str) -> str:
    """Save uploaded file to temporary location"""

    # Create temp directory if it doesn't exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
    file_ext = Path(file.filename).suffix
    temp_filename = f"invoice_{file_hash}{file_ext}"
    temp_path = Path(temp_dir) / temp_filename

    # Save file
    async with aiofiles.open(temp_path, 'wb') as temp_file:
        content = await file.read()
        await temp_file.write(content)

    logger.debug(f"Saved temp file: {temp_path}")
    return str(temp_path)


def cleanup_temp_file(file_path: str):
    """Remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def ensure_directory(directory: str) -> Path:
    """Ensure directory exists and return Path object"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)


def validate_file_type(filename: str, allowed_types: set) -> bool:
    """Validate file type by extension"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_types