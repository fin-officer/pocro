"""
FastAPI dependencies for European Invoice OCR
"""

import logging
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import AppSettings as Settings
from core.pipeline import EuropeanInvoiceProcessor

logger = logging.getLogger(__name__)

# Global instances
_processor_instance: Optional[EuropeanInvoiceProcessor] = None
_settings_instance: Optional[Settings] = None

# Rate limiting storage
rate_limit_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

# Authentication
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached)

    Returns:
        Settings instance
    """
    global _settings_instance

    if _settings_instance is None:
        _settings_instance = Settings()
        logger.info("Settings loaded")

    return _settings_instance


async def get_processor() -> EuropeanInvoiceProcessor:
    """
    Get the invoice processor instance

    Returns:
        EuropeanInvoiceProcessor instance

    Raises:
        HTTPException: If processor is not initialized
    """
    global _processor_instance

    if _processor_instance is None:
        try:
            settings = get_settings()
            _processor_instance = EuropeanInvoiceProcessor(settings)
            await _processor_instance.initialize()
            logger.info("Invoice processor initialized")

        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise HTTPException(status_code=503, detail="Invoice processor is not available")

    return _processor_instance


async def cleanup_processor():
    """
    Cleanup the processor instance
    """
    global _processor_instance

    if _processor_instance:
        try:
            await _processor_instance.cleanup()
            _processor_instance = None
            logger.info("Invoice processor cleaned up")

        except Exception as e:
            logger.error(f"Failed to cleanup processor: {e}")


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    # Check for forwarded headers first (proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, use the first one
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, max_requests: int = 60, window_seconds: int = 60) -> None:
    """
    Simple rate limiting dependency

    Args:
        request: FastAPI request object
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds

    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = get_client_ip(request)
    current_time = time.time()
    window_start = current_time - window_seconds

    # Get or create client's request history
    client_requests = rate_limit_storage[client_ip]

    # Remove old requests outside the window
    while client_requests and client_requests[0] < window_start:
        client_requests.popleft()

    # Check if limit exceeded
    if len(client_requests) >= max_requests:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "max_requests": max_requests,
                "window_seconds": window_seconds,
                "retry_after": int(client_requests[0] + window_seconds - current_time),
            },
        )

    # Add current request
    client_requests.append(current_time)


def rate_limit_strict(request: Request) -> None:
    """
    Strict rate limiting for processing endpoints
    """
    return rate_limit(request, max_requests=10, window_seconds=60)


def rate_limit_relaxed(request: Request) -> None:
    """
    Relaxed rate limiting for read-only endpoints
    """
    return rate_limit(request, max_requests=100, window_seconds=60)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """
    Verify API key if authentication is enabled

    Args:
        credentials: HTTP Bearer credentials
        settings: Application settings

    Returns:
        API key if valid, None if authentication disabled

    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication if not configured
    if not hasattr(settings, "api_key") or not settings.api_key:
        return None

    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")

    if credentials.credentials != settings.api_key:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return credentials.credentials


def check_file_upload_permissions(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """
    Check if file upload is allowed

    Args:
        request: FastAPI request object
        settings: Application settings

    Raises:
        HTTPException: If uploads not allowed
    """
    # Check if uploads are disabled
    if hasattr(settings, "allow_file_upload") and not settings.allow_file_upload:
        raise HTTPException(status_code=403, detail="File uploads are disabled")

    # Check content length
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_file_size:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
        )


def validate_content_type(request: Request, allowed_types: Optional[list] = None) -> None:
    """
    Validate request content type

    Args:
        request: FastAPI request object
        allowed_types: List of allowed content types

    Raises:
        HTTPException: If content type not allowed
    """
    if allowed_types is None:
        allowed_types = ["multipart/form-data", "application/json", "application/octet-stream"]

    content_type = request.headers.get("content-type", "").split(";")[0]

    if content_type and not any(content_type.startswith(allowed) for allowed in allowed_types):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")


class DependencyManager:
    """Manage application dependencies and their lifecycle"""

    def __init__(self):
        self.initialized = False
        self.cleanup_handlers = []

    async def initialize(self):
        """Initialize all dependencies"""
        if self.initialized:
            return

        try:
            # Initialize settings
            settings = get_settings()
            logger.info("Settings initialized")

            # Initialize processor
            processor = await get_processor()
            logger.info("Processor initialized")

            self.initialized = True
            logger.info("All dependencies initialized")

        except Exception as e:
            logger.error(f"Dependency initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup all dependencies"""
        try:
            # Run cleanup handlers
            for handler in self.cleanup_handlers:
                try:
                    await handler()
                except Exception as e:
                    logger.error(f"Cleanup handler failed: {e}")

            # Cleanup processor
            await cleanup_processor()

            # Clear rate limiting storage
            rate_limit_storage.clear()

            self.initialized = False
            logger.info("All dependencies cleaned up")

        except Exception as e:
            logger.error(f"Dependency cleanup failed: {e}")

    def add_cleanup_handler(self, handler):
        """Add cleanup handler"""
        self.cleanup_handlers.append(handler)


# Global dependency manager
dependency_manager = DependencyManager()


# Health check dependency
async def health_check_dependency() -> Dict[str, Any]:
    """
    Dependency for health checks

    Returns:
        Basic health status
    """
    try:
        # Check if processor is available
        processor_status = _processor_instance is not None

        # Basic system check
        import psutil

        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        return {
            "processor_ready": processor_status,
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "status": "healthy" if processor_status and cpu_percent < 90 and memory_percent < 90 else "degraded",
        }

    except Exception as e:
        logger.error(f"Health check dependency failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# Request logging dependency
def log_request(request: Request) -> None:
    """
    Log incoming requests

    Args:
        request: FastAPI request object
    """
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(f"Request: {request.method} {request.url.path} " f"from {client_ip} ({user_agent})")


# Performance monitoring dependency
def monitor_request_performance(request: Request):
    """
    Monitor request performance

    Args:
        request: FastAPI request object

    Returns:
        Context manager for performance monitoring
    """

    class PerformanceMonitor:
        def __init__(self, request: Request):
            self.request = request
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                endpoint = self.request.url.path
                method = self.request.method

                # Log slow requests
                if duration > 5.0:  # 5 seconds threshold
                    logger.warning(f"Slow request: {method} {endpoint} took {duration:.2f}s")

                # Store metrics
                from utils.monitoring import performance_tracker

                performance_tracker.record_processing_time(f"{method}_{endpoint}", duration, exc_type is None)

    return PerformanceMonitor(request)


# Cache management
_cache_storage: Dict[str, Dict[str, Any]] = {}


def get_cached_data(cache_key: str, ttl: int = 300) -> Optional[Any]:
    """
    Get cached data

    Args:
        cache_key: Cache key
        ttl: Time to live in seconds

    Returns:
        Cached data or None if expired/not found
    """
    if cache_key not in _cache_storage:
        return None

    cache_entry = _cache_storage[cache_key]

    if time.time() - cache_entry["timestamp"] > ttl:
        del _cache_storage[cache_key]
        return None

    return cache_entry["data"]


def set_cached_data(cache_key: str, data: Any) -> None:
    """
    Set cached data

    Args:
        cache_key: Cache key
        data: Data to cache
    """
    _cache_storage[cache_key] = {"data": data, "timestamp": time.time()}


def clear_cache() -> None:
    """Clear all cached data"""
    _cache_storage.clear()


# Dependency for cached responses
def cache_response(ttl: int = 300):
    """
    Dependency factory for response caching

    Args:
        ttl: Time to live in seconds

    Returns:
        Dependency function
    """

    def dependency(request: Request):
        cache_key = f"{request.method}_{request.url.path}_{request.url.query}"

        # Try to get cached response
        cached_data = get_cached_data(cache_key, ttl)
        if cached_data:
            return cached_data

        # Return cache setter function
        def set_cache(data):
            set_cached_data(cache_key, data)
            return data

        return set_cache

    return dependency
