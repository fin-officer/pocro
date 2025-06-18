"""
Monitoring utilities for tracking resource usage and performance metrics.
"""
import os
import time
import psutil
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from pydantic import BaseModel

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

class MemoryUsage(BaseModel):
    """Memory usage metrics"""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float   # Virtual Memory Size in MB
    percent: float  # Percentage of memory used


def monitor_memory_usage(func: F) -> F:
    """
    Decorator to monitor memory usage of a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function that logs memory usage
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        process = psutil.Process()
        
        # Memory before
        mem_before = process.memory_info()
        start_time = time.time()
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            return result
            
        finally:
            # Memory after
            mem_after = process.memory_info()
            end_time = time.time()
            
            # Calculate memory usage
            rss_mb = (mem_after.rss - mem_before.rss) / (1024 * 1024)
            vms_mb = (mem_after.vms - mem_before.vms) / (1024 * 1024)
            
            # Log the memory usage
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Function {func.__name__} executed in {end_time - start_time:.2f}s. "
                f"Memory used: {rss_mb:.2f}MB RSS, {vms_mb:.2f}MB VMS"
            )
    
    return cast(F, wrapper)


def get_memory_usage() -> MemoryUsage:
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()
    
    return MemoryUsage(
        rss_mb=mem_info.rss / (1024 * 1024),
        vms_mb=mem_info.vms / (1024 * 1024),
        percent=mem_percent
    )


def log_system_resources() -> Dict[str, Any]:
    """Log system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    resources = {
        'cpu_percent': cpu_percent,
        'memory_total_gb': memory.total / (1024 ** 3),
        'memory_available_gb': memory.available / (1024 ** 3),
        'memory_percent': memory.percent,
        'disk_total_gb': disk.total / (1024 ** 3),
        'disk_used_gb': disk.used / (1024 ** 3),
        'disk_percent': disk.percent,
    }
    
    logger = logging.getLogger(__name__)
    logger.debug(f"System resources: {resources}")
    
    return resources

