"""
File handling utilities for invoice processing
"""

import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

import aiofiles
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for invoice processing"""

    def __init__(self, temp_dir: str = "./data/temp", output_dir: str = "./data/output"):
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)

        # Create directories if they don't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported file types
        self.supported_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        self.max_file_size = 50 * 1024 * 1024  # 50MB

    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file

        Args:
            filename: Name of the file
            file_size: Size of the file in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename is required"

        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.supported_extensions:
            return False, f"Unsupported file type. Supported: {', '.join(self.supported_extensions)}"

        # Check file size
        if file_size > self.max_file_size:
            return False, f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"

        return True, None

    async def save_upload_file(self, upload_file: UploadFile, custom_name: Optional[str] = None) -> str:
        """
        Save uploaded file to temporary directory

        Args:
            upload_file: FastAPI UploadFile object
            custom_name: Optional custom filename

        Returns:
            Path to saved file
        """
        # Validate file
        file_size = 0
        if hasattr(upload_file, "size"):
            file_size = upload_file.size

        is_valid, error_msg = self.validate_file(upload_file.filename, file_size)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate unique filename
        if custom_name:
            filename = custom_name
        else:
            file_ext = Path(upload_file.filename).suffix
            unique_id = str(uuid.uuid4())
            filename = f"temp_{unique_id}{file_ext}"

        file_path = self.temp_dir / filename

        try:
            # Save file
            async with aiofiles.open(file_path, "wb") as f:
                content = await upload_file.read()
                await f.write(content)

            logger.info(f"File saved: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save file {upload_file.filename}: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            raise

    def save_processed_result(self, result_data: dict, filename: str) -> str:
        """
        Save processing result to output directory

        Args:
            result_data: Processing result data
            filename: Output filename

        Returns:
            Path to saved result file
        """
        import json

        output_path = self.output_dir / f"{filename}.json"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Result saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save result to {output_path}: {e}")
            raise

    def cleanup_temp_file(self, file_path: str):
        """
        Remove temporary file

        Args:
            file_path: Path to file to remove
        """
        try:
            path = Path(file_path)
            if path.exists() and path.parent == self.temp_dir:
                path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def cleanup_old_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files

        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old temporary files")

        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")

    def get_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of file

        Args:
            file_path: Path to file

        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            return hash_md5.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            raise

    def get_file_info(self, file_path: str) -> dict:
        """
        Get file information

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()

            return {
                "filename": path.name,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": path.suffix.lower(),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "hash": self.get_file_hash(file_path),
            }

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise


# Global file handler instance
file_handler = FileHandler()


async def save_temp_file(upload_file: UploadFile, temp_dir: str) -> str:
    """
    Convenience function to save temporary file

    Args:
        upload_file: FastAPI UploadFile object
        temp_dir: Temporary directory path

    Returns:
        Path to saved file
    """
    handler = FileHandler(temp_dir=temp_dir)
    return await handler.save_upload_file(upload_file)


async def cleanup_temp_file(file_path: str):
    """
    Convenience function to cleanup temporary file

    Args:
        file_path: Path to file to remove
    """
    file_handler.cleanup_temp_file(file_path)


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        directory_path: Path to directory

    Returns:
        Path object
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_unique_filename(base_name: str, extension: str, directory: str) -> str:
    """
    Generate unique filename in directory

    Args:
        base_name: Base filename without extension
        extension: File extension (with dot)
        directory: Target directory

    Returns:
        Unique filename
    """
    directory_path = Path(directory)
    counter = 1

    # Try base name first
    filename = f"{base_name}{extension}"
    if not (directory_path / filename).exists():
        return filename

    # Add counter if file exists
    while True:
        filename = f"{base_name}_{counter}{extension}"
        if not (directory_path / filename).exists():
            return filename
        counter += 1


def copy_file_safe(source: str, destination: str) -> bool:
    """
    Safely copy file with error handling

    Args:
        source: Source file path
        destination: Destination file path

    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(source_path, dest_path)

        logger.debug(f"File copied: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy file {source} to {destination}: {e}")
        return False


def move_file_safe(source: str, destination: str) -> bool:
    """
    Safely move file with error handling

    Args:
        source: Source file path
        destination: Destination file path

    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        shutil.move(str(source_path), str(dest_path))

        logger.debug(f"File moved: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to move file {source} to {destination}: {e}")
        return False


def delete_file_safe(file_path: str) -> bool:
    """
    Safely delete file with error handling

    Args:
        file_path: Path to file to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"File deleted: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def list_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """
    List files in directory with specific extensions

    Args:
        directory: Directory to search
        extensions: List of extensions (with dots)

    Returns:
        List of file paths
    """
    try:
        directory_path = Path(directory)
        files = []

        for ext in extensions:
            files.extend([str(f) for f in directory_path.glob(f"*{ext}") if f.is_file()])

        return sorted(files)

    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        return []


def get_directory_size(directory: str) -> int:
    """
    Get total size of directory in bytes

    Args:
        directory: Directory path

    Returns:
        Total size in bytes
    """
    try:
        directory_path = Path(directory)
        total_size = 0

        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    except Exception as e:
        logger.error(f"Failed to calculate directory size for {directory}: {e}")
        return 0


def archive_old_files(directory: str, archive_dir: str, max_age_days: int = 30) -> int:
    """
    Archive old files to archive directory

    Args:
        directory: Source directory
        archive_dir: Archive directory
        max_age_days: Maximum age of files to keep

    Returns:
        Number of archived files
    """
    import time
    from datetime import datetime

    try:
        source_path = Path(directory)
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        archived_count = 0

        # Create dated archive subdirectory
        date_str = datetime.now().strftime("%Y-%m-%d")
        dated_archive = archive_path / date_str
        dated_archive.mkdir(exist_ok=True)

        for file_path in source_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    # Move to archive
                    archive_file_path = dated_archive / file_path.name
                    shutil.move(str(file_path), str(archive_file_path))
                    archived_count += 1

        logger.info(f"Archived {archived_count} files from {directory}")
        return archived_count

    except Exception as e:
        logger.error(f"Failed to archive files from {directory}: {e}")
        return 0
