"""
FastAPI middleware for European Invoice OCR
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

import psutil
from fastapi import HTTPException, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from utils.monitoring import performance_tracker, system_monitor

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""

    def __init__(self, app, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        if self.log_requests:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")

            logger.info(f"[{request_id}] {request.method} {request.url.path} " f"from {client_ip} - {user_agent}")

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)
            processing_time = time.time() - start_time

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

            # Log response
            if self.log_responses:
                logger.info(f"[{request_id}] Response: {response.status_code} " f"({processing_time:.3f}s)")

            return response

        except Exception as e:
            processing_time = time.time() - start_time

            logger.error(f"[{request_id}] Error: {str(e)} ({processing_time:.3f}s)")

            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics collection"""

    def __init__(self, app, enable_detailed_metrics: bool = True):
        super().__init__(app)
        self.enable_detailed_metrics = enable_detailed_metrics

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Start timing
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Track request
        endpoint = f"{request.method}_{request.url.path}"

        try:
            # Process request
            response = await call_next(request)

            # Calculate metrics
            processing_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory

            # Record metrics
            performance_tracker.record_processing_time(endpoint, processing_time, True)

            # Add detailed metrics if enabled
            if self.enable_detailed_metrics:
                response.headers["X-Memory-Delta"] = f"{memory_delta:.2f}MB"

            # Log slow requests
            if processing_time > 5.0:
                logger.warning(
                    f"Slow request: {endpoint} took {processing_time:.3f}s " f"(memory: +{memory_delta:.2f}MB)"
                )

            return response

        except Exception as e:
            # Record error
            processing_time = time.time() - start_time
            performance_tracker.record_processing_time(endpoint, processing_time, False)

            logger.error(f"Request failed: {endpoint} - {str(e)}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""

    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware"""

    def __init__(
        self, app, calls: int = 100, period: int = 60, per_endpoint: bool = True, exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.per_endpoint = per_endpoint
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

        # Storage for rate limiting
        self.requests = {}

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Create key for rate limiting
        if self.per_endpoint:
            key = f"{client_id}:{request.method}:{request.url.path}"
        else:
            key = client_id

        # Check rate limit
        current_time = time.time()

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if current_time - req_time < self.period]

        # Check limit
        if len(self.requests[key]) >= self.calls:
            oldest_request = min(self.requests[key])
            retry_after = int(self.period - (current_time - oldest_request))

            logger.warning(f"Rate limit exceeded for {client_id}")

            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        # Add current request
        self.requests[key].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.calls - len(self.requests[key]))
        reset_time = int(current_time + self.period)

        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get from X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Try X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    async def _cleanup_task(self):
        """Background task to cleanup old rate limit data"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                current_time = time.time()
                keys_to_remove = []

                for key, timestamps in self.requests.items():
                    # Remove old timestamps
                    valid_timestamps = [ts for ts in timestamps if current_time - ts < self.period]

                    if valid_timestamps:
                        self.requests[key] = valid_timestamps
                    else:
                        keys_to_remove.append(key)

                # Remove empty entries
                for key in keys_to_remove:
                    del self.requests[key]

                logger.debug(f"Rate limit cleanup: removed {len(keys_to_remove)} entries")

            except Exception as e:
                logger.error(f"Rate limit cleanup failed: {e}")


class SystemMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for system resource monitoring"""

    def __init__(self, app, alert_thresholds: Optional[Dict[str, float]] = None):
        super().__init__(app)
        self.alert_thresholds = alert_thresholds or {"cpu_percent": 90.0, "memory_percent": 90.0, "disk_percent": 90.0}
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check system resources before processing
        await self._check_system_resources()

        # Process request
        response = await call_next(request)

        # Add system status to response headers (for monitoring)
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            response.headers["X-System-CPU"] = f"{cpu_percent:.1f}"
            response.headers["X-System-Memory"] = f"{memory_percent:.1f}"

        except Exception:
            pass  # Don't fail request if monitoring fails

        return response

    async def _check_system_resources(self):
        """Check system resources and alert if thresholds exceeded"""
        try:
            current_time = time.time()

            # Check CPU
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                await self._maybe_alert("cpu_high", cpu_percent, current_time)

            # Check Memory
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.alert_thresholds["memory_percent"]:
                await self._maybe_alert("memory_high", memory_percent, current_time)

            # Check Disk
            disk_percent = psutil.disk_usage("/").percent
            if disk_percent > self.alert_thresholds["disk_percent"]:
                await self._maybe_alert("disk_high", disk_percent, current_time)

        except Exception as e:
            logger.error(f"System resource check failed: {e}")

    async def _maybe_alert(self, alert_type: str, value: float, current_time: float):
        """Send alert if cooldown period has passed"""
        last_alert = self.last_alert_time.get(alert_type, 0)

        if current_time - last_alert > self.alert_cooldown:
            logger.warning(f"System alert: {alert_type} = {value:.1f}%")
            self.last_alert_time[alert_type] = current_time


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling and reporting"""

    def __init__(self, app, include_traceback: bool = False):
        super().__init__(app)
        self.include_traceback = include_traceback

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)

        except HTTPException:
            # Let HTTPExceptions pass through
            raise

        except Exception as e:
            # Log unexpected errors
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(f"[{request_id}] Unhandled exception: {str(e)}", exc_info=True)

            # Prepare error response
            error_data = {"error": "Internal server error", "request_id": request_id, "type": type(e).__name__}

            # Include traceback in development
            if self.include_traceback:
                import traceback

                error_data["traceback"] = traceback.format_exc()

            return JSONResponse(status_code=500, content=error_data, headers={"X-Request-ID": request_id})


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for setting cache control headers"""

    def __init__(self, app, default_max_age: int = 0):
        super().__init__(app)
        self.default_max_age = default_max_age

        # Cache rules for different endpoints
        self.cache_rules = {
            "/health": 30,
            "/status": 60,
            "/config": 300,
            "/languages": 3600,
            "/formats": 3600,
        }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Determine cache max-age
        max_age = self.default_max_age
        for path, cache_time in self.cache_rules.items():
            if request.url.path.startswith(path):
                max_age = cache_time
                break

        # Set cache control headers
        if max_age > 0:
            response.headers["Cache-Control"] = f"public, max-age={max_age}"
        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response


def create_middleware_stack(app, config: Optional[Dict[str, Any]] = None):
    """
    Create and configure the complete middleware stack

    Args:
        app: FastAPI application
        config: Middleware configuration
    """
    if config is None:
        config = {}

    # Add middleware in reverse order (they wrap around each other)

    # Error handling (outermost)
    app.add_middleware(ErrorHandlingMiddleware, include_traceback=config.get("include_traceback", False))

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    if config.get("enable_rate_limiting", True):
        app.add_middleware(
            RateLimitMiddleware,
            calls=config.get("rate_limit_calls", 100),
            period=config.get("rate_limit_period", 60),
            per_endpoint=config.get("rate_limit_per_endpoint", True),
        )

    # System monitoring
    if config.get("enable_system_monitoring", True):
        app.add_middleware(SystemMonitoringMiddleware)

    # Performance monitoring
    if config.get("enable_performance_monitoring", True):
        app.add_middleware(
            PerformanceMonitoringMiddleware, enable_detailed_metrics=config.get("detailed_metrics", True)
        )

    # Cache control
    app.add_middleware(CacheControlMiddleware, default_max_age=config.get("default_cache_max_age", 0))

    # Request logging (innermost)
    if config.get("enable_request_logging", True):
        app.add_middleware(
            RequestLoggingMiddleware,
            log_requests=config.get("log_requests", True),
            log_responses=config.get("log_responses", False),
        )

    logger.info("Middleware stack configured")
