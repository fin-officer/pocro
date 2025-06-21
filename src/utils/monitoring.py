"""
System monitoring and metrics utilities
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System resource monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))

        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.request_counter = Counter("invoice_requests_total", "Total requests", ["method", "endpoint", "status"])
            self.request_duration = Histogram("invoice_request_duration_seconds", "Request duration")
            self.gpu_memory_gauge = Gauge("gpu_memory_usage_bytes", "GPU memory usage")
            self.cpu_gauge = Gauge("cpu_usage_percent", "CPU usage percentage")
            self.memory_gauge = Gauge("memory_usage_bytes", "Memory usage in bytes")

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            info = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "cpu": self._get_cpu_info(),
                "memory": self._get_memory_info(),
                "disk": self._get_disk_info(),
                "network": self._get_network_info(),
                "processes": self._get_process_info(),
            }

            # Add GPU info if available
            if TORCH_AVAILABLE:
                info["gpu"] = self._get_gpu_info()

            return info

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            cpu_info = {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "physical_count": psutil.cpu_count(logical=False),
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "per_cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
            }

            # Update Prometheus metric
            if PROMETHEUS_AVAILABLE:
                self.cpu_gauge.set(cpu_percent)

            # Store in history
            self.metrics_history["cpu_usage"].append({"timestamp": time.time(), "value": cpu_percent})

            return cpu_info

        except Exception as e:
            logger.error(f"Failed to get CPU info: {e}")
            return {"error": str(e)}

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            memory_info = {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "free_bytes": memory.free,
                "usage_percent": memory.percent,
                "swap_total_bytes": swap.total,
                "swap_used_bytes": swap.used,
                "swap_free_bytes": swap.free,
                "swap_usage_percent": swap.percent,
            }

            # Update Prometheus metric
            if PROMETHEUS_AVAILABLE:
                self.memory_gauge.set(memory.used)

            # Store in history
            self.metrics_history["memory_usage"].append({"timestamp": time.time(), "value": memory.percent})

            return memory_info

        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {"error": str(e)}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information"""
        try:
            disk_usage = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            disk_info = {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
            }

            return disk_info

        except Exception as e:
            logger.error(f"Failed to get disk info: {e}")
            return {"error": str(e)}

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())

            network_info = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout,
                "connections_count": net_connections,
            }

            return network_info

        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return {"error": str(e)}

    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information"""
        try:
            process = psutil.Process()

            process_info = {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info_bytes": process.memory_info().rss,
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "cmdline": " ".join(process.cmdline()),
            }

            return process_info

        except Exception as e:
            logger.error(f"Failed to get process info: {e}")
            return {"error": str(e)}

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return {"available": False}

            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": [],
            }

            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = device_props.total_memory

                device_info = {
                    "index": i,
                    "name": device_props.name,
                    "total_memory_bytes": memory_total,
                    "allocated_memory_bytes": memory_allocated,
                    "reserved_memory_bytes": memory_reserved,
                    "free_memory_bytes": memory_total - memory_reserved,
                    "memory_usage_percent": (memory_reserved / memory_total) * 100,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "multiprocessor_count": device_props.multi_processor_count,
                }

                gpu_info["devices"].append(device_info)

                # Update Prometheus metric for first GPU
                if i == 0 and PROMETHEUS_AVAILABLE:
                    self.gpu_memory_gauge.set(memory_reserved)

            return gpu_info

        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {"available": False, "error": str(e)}

    def get_metrics_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            history = self.metrics_history.get(metric_name, [])

            return [entry for entry in history if entry["timestamp"] > cutoff_time]

        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            if not PROMETHEUS_AVAILABLE:
                return "# Prometheus not available\n"

            return generate_latest().decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error: {e}\n"


class PerformanceTracker:
    """Track performance metrics for invoice processing"""

    def __init__(self):
        self.processing_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.total_processed = 0
        self.lock = threading.Lock()

    def record_processing_time(self, operation: str, duration: float, success: bool = True):
        """Record processing time for an operation"""
        try:
            with self.lock:
                timestamp = time.time()

                self.processing_times.append(
                    {"operation": operation, "duration": duration, "timestamp": timestamp, "success": success}
                )

                if success:
                    self.success_counts[operation] += 1
                else:
                    self.error_counts[operation] += 1

                self.total_processed += 1

                logger.debug(f"Recorded {operation}: {duration:.3f}s ({'success' if success else 'error'})")

        except Exception as e:
            logger.error(f"Failed to record processing time: {e}")

    def get_performance_stats(self, operation: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            with self.lock:
                cutoff_time = time.time() - (hours * 3600)

                # Filter data
                if operation:
                    recent_data = [
                        entry
                        for entry in self.processing_times
                        if entry["timestamp"] > cutoff_time and entry["operation"] == operation
                    ]
                else:
                    recent_data = [entry for entry in self.processing_times if entry["timestamp"] > cutoff_time]

                if not recent_data:
                    return {"error": "No data available for the specified period"}

                # Calculate statistics
                durations = [entry["duration"] for entry in recent_data]
                successful = [entry for entry in recent_data if entry["success"]]
                failed = [entry for entry in recent_data if not entry["success"]]

                stats = {
                    "period_hours": hours,
                    "total_operations": len(recent_data),
                    "successful_operations": len(successful),
                    "failed_operations": len(failed),
                    "success_rate": len(successful) / len(recent_data) if recent_data else 0,
                    "average_duration": sum(durations) / len(durations) if durations else 0,
                    "min_duration": min(durations) if durations else 0,
                    "max_duration": max(durations) if durations else 0,
                    "median_duration": sorted(durations)[len(durations) // 2] if durations else 0,
                    "operations_per_hour": len(recent_data) / hours if hours > 0 else 0,
                }

                # Add percentiles
                if durations:
                    sorted_durations = sorted(durations)
                    stats["p50_duration"] = sorted_durations[int(len(sorted_durations) * 0.5)]
                    stats["p90_duration"] = sorted_durations[int(len(sorted_durations) * 0.9)]
                    stats["p95_duration"] = sorted_durations[int(len(sorted_durations) * 0.95)]
                    stats["p99_duration"] = sorted_durations[int(len(sorted_durations) * 0.99)]

                return stats

        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}


# Global instances
system_monitor = SystemMonitor()
performance_tracker = PerformanceTracker()


def track_processing_time(operation: str):
    """Decorator to track processing time of functions"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                performance_tracker.record_processing_time(operation, duration, success)

                # Update Prometheus histogram if available
                if PROMETHEUS_AVAILABLE and hasattr(system_monitor, "request_duration"):
                    system_monitor.request_duration.observe(duration)

        return wrapper

    return decorator


def monitor_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information"""
    try:
        memory_info = system_monitor._get_memory_info()
        gpu_info = system_monitor._get_gpu_info() if TORCH_AVAILABLE else {"available": False}

        return {
            "system_memory": {
                "used_gb": memory_info.get("used_bytes", 0) / (1024**3),
                "total_gb": memory_info.get("total_bytes", 0) / (1024**3),
                "usage_percent": memory_info.get("usage_percent", 0),
            },
            "gpu_memory": gpu_info,
        }

    except Exception as e:
        logger.error(f"Failed to monitor memory usage: {e}")
        return {"error": str(e)}


def log_system_stats():
    """Log current system statistics"""
    try:
        stats = system_monitor.get_system_info()

        logger.info(
            f"System Stats - CPU: {stats['cpu']['usage_percent']:.1f}%, "
            f"Memory: {stats['memory']['usage_percent']:.1f}%, "
            f"Disk: {stats['disk']['usage_percent']:.1f}%"
        )

        if stats.get("gpu", {}).get("available"):
            gpu_usage = stats["gpu"]["devices"][0]["memory_usage_percent"]
            logger.info(f"GPU Memory: {gpu_usage:.1f}%")

    except Exception as e:
        logger.error(f"Failed to log system stats: {e}")


def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check"""
    try:
        health = {"status": "healthy", "timestamp": datetime.now().isoformat(), "checks": {}}

        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        health["checks"]["cpu"] = {"status": "ok" if cpu_usage < 90 else "warning", "usage_percent": cpu_usage}

        # Check memory usage
        memory = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "status": "ok" if memory.percent < 90 else "warning",
            "usage_percent": memory.percent,
        }

        # Check disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        health["checks"]["disk"] = {"status": "ok" if disk_percent < 90 else "warning", "usage_percent": disk_percent}

        # Check GPU if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory * 100
                health["checks"]["gpu"] = {
                    "status": "ok" if gpu_memory < 90 else "warning",
                    "memory_usage_percent": gpu_memory,
                }
            except Exception:
                health["checks"]["gpu"] = {"status": "error", "error": "GPU check failed"}

        # Overall status
        warning_checks = [check for check in health["checks"].values() if check["status"] == "warning"]
        error_checks = [check for check in health["checks"].values() if check["status"] == "error"]

        if error_checks:
            health["status"] = "unhealthy"
        elif warning_checks:
            health["status"] = "degraded"

        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "timestamp": datetime.now().isoformat(), "error": str(e)}


def cleanup_old_metrics(days: int = 7):
    """Cleanup old metrics data"""
    try:
        cutoff_time = time.time() - (days * 24 * 3600)

        # Clean performance tracker data
        with performance_tracker.lock:
            performance_tracker.processing_times = deque(
                [entry for entry in performance_tracker.processing_times if entry["timestamp"] > cutoff_time],
                maxlen=1000,
            )

        # Clean system monitor history
        for metric_name in list(system_monitor.metrics_history.keys()):
            system_monitor.metrics_history[metric_name] = deque(
                [entry for entry in system_monitor.metrics_history[metric_name] if entry["timestamp"] > cutoff_time],
                maxlen=100,
            )

        logger.info(f"Cleaned up metrics older than {days} days")

    except Exception as e:
        logger.error(f"Failed to cleanup old metrics: {e}")


class AlertManager:
    """Simple alerting for system issues"""

    def __init__(self):
        self.alert_thresholds = {
            "cpu_usage": 90,
            "memory_usage": 90,
            "disk_usage": 90,
            "gpu_memory_usage": 90,
            "error_rate": 0.1,
        }
        self.alert_cooldown = 300  # 5 minutes
        self.last_alerts = {}

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        current_time = time.time()

        try:
            # Get current system stats
            stats = system_monitor.get_system_info()

            # Check CPU usage
            if stats["cpu"]["usage_percent"] > self.alert_thresholds["cpu_usage"]:
                alert_key = "cpu_high"
                if self._should_alert(alert_key, current_time):
                    alerts.append(
                        {
                            "type": "cpu_high",
                            "severity": "warning",
                            "message": f"High CPU usage: {stats['cpu']['usage_percent']:.1f}%",
                            "timestamp": current_time,
                        }
                    )

            # Check memory usage
            if stats["memory"]["usage_percent"] > self.alert_thresholds["memory_usage"]:
                alert_key = "memory_high"
                if self._should_alert(alert_key, current_time):
                    alerts.append(
                        {
                            "type": "memory_high",
                            "severity": "warning",
                            "message": f"High memory usage: {stats['memory']['usage_percent']:.1f}%",
                            "timestamp": current_time,
                        }
                    )

            # Check error rate
            perf_stats = performance_tracker.get_performance_stats(hours=1)
            if perf_stats.get("total_operations", 0) > 0:
                error_rate = 1 - perf_stats.get("success_rate", 1)
                if error_rate > self.alert_thresholds["error_rate"]:
                    alert_key = "error_rate_high"
                    if self._should_alert(alert_key, current_time):
                        alerts.append(
                            {
                                "type": "error_rate_high",
                                "severity": "critical",
                                "message": f"High error rate: {error_rate:.1%}",
                                "timestamp": current_time,
                            }
                        )

            return alerts

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
            return []

    def _should_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if enough time has passed since last alert"""
        last_alert_time = self.last_alerts.get(alert_key, 0)

        if current_time - last_alert_time > self.alert_cooldown:
            self.last_alerts[alert_key] = current_time
            return True

        return False


# Global alert manager
alert_manager = AlertManager()
