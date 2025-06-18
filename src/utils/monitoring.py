"""
System monitoring utilities
"""
import psutil
import time
from typing import Dict, Any
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def monitor_memory_usage() -> Dict[str, Any]:
    """Monitor system memory usage"""

    # CPU memory
    cpu_memory = psutil.virtual_memory()

    memory_info = {
        "cpu_memory": {
            "total_gb": round(cpu_memory.total / (1024**3), 2),
            "used_gb": round(cpu_memory.used / (1024**3), 2),
            "available_gb": round(cpu_memory.available / (1024**3), 2),
            "percentage": cpu_memory.percent
        }
    }

    # GPU memory if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)

                gpu_memory[f"gpu_{i}"] = {
                    "total_gb": round(total / (1024**3), 2),
                    "allocated_gb": round(allocated / (1024**3), 2),
                    "cached_gb": round(cached / (1024**3), 2),
                    "free_gb": round((total - allocated) / (1024**3), 2),
                    "utilization": round((allocated / total) * 100, 2)
                }

            memory_info["gpu_memory"] = gpu_memory

        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")

    return memory_info


def monitor_cpu_usage() -> Dict[str, float]:
    """Monitor CPU usage"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
    }


class PerformanceMonitor:
    """Performance monitoring context manager"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = monitor_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = monitor_memory_usage()

        duration = end_time - self.start_time

        logger.info(f"Performance - {self.operation_name}: {duration:.2f}s")

        # Log memory changes if significant
        start_used = self.start_memory["cpu_memory"]["used_gb"]
        end_used = end_memory["cpu_memory"]["used_gb"]
        memory_delta = end_used - start_used

        if abs(memory_delta) > 0.1:  # If change > 100MB
            logger.info(f"Memory change - {self.operation_name}: {memory_delta:+.2f}GB")