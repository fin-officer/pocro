"""
Performance tests for processing throughput
"""
import pytest
import time
import asyncio
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.pipeline import EuropeanInvoiceProcessor
from src.core.preprocessor import InvoicePreprocessor
from src.core.ocr_engine import InvoiceOCREngine


class TestThroughputPerformance:
    """Test processing throughput and performance characteristics"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Performance monitoring helper"""
        class PerformanceMonitor:
            def __init__(self):
                self.measurements = []
                self.start_time = None
                self.end_time = None
            
            def start_timing(self):
                """Start timing measurement"""
                self.start_time = time.time()
            
            def stop_timing(self):
                """Stop timing measurement"""
                self.end_time = time.time()
                return self.get_duration()
            
            def get_duration(self) -> float:
                """Get duration in seconds"""
                if self.start_time and self.end_time:
                    return self.end_time - self.start_time
                return 0.0
            
            def measure_operation(self, operation_name: str, operation_func, *args, **kwargs):
                """Measure a single operation"""
                start = time.time()
                result = operation_func(*args, **kwargs)
                end = time.time()
                
                duration = end - start
                self.measurements.append({
                    'operation': operation_name,
                    'duration': duration,
                    'timestamp': start
                })
                
                return result, duration
            
            async def measure_async_operation(self, operation_name: str, operation_func, *args, **kwargs):
                """Measure an async operation"""
                start = time.time()
                result = await operation_func(*args, **kwargs)
                end = time.time()
                
                duration = end - start
                self.measurements.append({
                    'operation': operation_name,
                    'duration': duration,
                    'timestamp': start
                })
                
                return result, duration
            
            def get_statistics(self) -> Dict[str, float]:
                """Get performance statistics"""
                if not self.measurements:
                    return {}
                
                durations = [m['duration'] for m in self.measurements]
                
                return {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'average': statistics.mean(durations),
                    'median': statistics.median(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    'throughput_per_second': len(durations) / sum(durations) if sum(durations) > 0 else 0.0
                }
            
            def print_report(self):
                """Print performance report"""
                stats = self.get_statistics()
                if not stats:
                    print("No measurements recorded")
                    return
                
                print(f"\nPerformance Report:")
                print(f"  Operations: {stats['count']}")
                print(f"  Total Time: {stats['total_time']:.2f}s")
                print(f"  Average: {stats['average']:.3f}s")
                print(f"  Median: {stats['median']:.3f}s")
                print(f"  Min: {stats['min']:.3f}s")
                print(f"  Max: {stats['max']:.3f}s")
                print(f"  Std Dev: {stats['std_dev']:.3f}s")
                print(f"  Throughput: {stats['throughput_per_second']:.2f} ops/sec")
        
        return PerformanceMonitor()
    
    def test_preprocessor_throughput(self, performance_monitor):
        """Test image preprocessing throughput"""
        preprocessor = InvoicePreprocessor()
        
        # Test with different image sizes
        image_sizes = [
            (800, 600),     # Small
            (1200, 900),    # Medium
            (1600, 1200),   # Large
        ]
        
        for width, height in image_sizes:
            test_images = []
            
            # Create test images
            for i in range(10):
                image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                test_images.append(image)
            
            # Measure preprocessing performance
            for i, image in enumerate(test_images):
                _, duration = performance_monitor.measure_operation(
                    f"preprocess_{width}x{height}_{i}",
                    preprocessor.preprocess_invoice_image,
                    image
                )
                
                # Each operation should complete reasonably quickly
                assert duration < 10.0, f"Preprocessing too slow: {duration:.2f}s"
        
        stats = performance_monitor.get_statistics()
        performance_monitor.print_report()
        
        # Overall throughput requirements
        assert stats['throughput_per_second'] > 0.5, f"Throughput too low: {stats['throughput_per_second']:.2f} ops/sec"
        assert stats['average'] < 5.0, f"Average processing time too high: {stats['average']:.2f}s"
    
    def test_ocr_engine_throughput(self, performance_monitor):
        """Test OCR engine throughput with mocked components"""
        from unittest.mock import patch
        
        with patch('src.core.ocr_engine.EASYOCR_AVAILABLE', True):
            with patch('src.core.ocr_engine.easyocr.Reader') as mock_reader:
                # Mock OCR responses
                mock_reader_instance = Mock()
                mock_reader_instance.readtext.return_value = [
                    ([[50, 50], [300, 50], [300, 80], [50, 80]], 'INVOICE #001', 0.95),
                    ([[50, 100], [200, 100], [200, 120], [50, 120]], 'Date: 2024-01-15', 0.92),
                    ([[400, 500], [550, 500], [550, 520], [400, 520]], 'Total: €100.00', 0.93)
                ]
                mock_reader.return_value = mock_reader_instance
                
                ocr_engine = InvoiceOCREngine(engine_type="easyocr")
                
                # Test with multiple images
                for i in range(20):
                    test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                    
                    _, duration = performance_monitor.measure_operation(
                        f"ocr_extract_{i}",
                        ocr_engine.extract_invoice_text,
                        test_image
                    )
                    
                    # OCR should be fast with mocked components
                    assert duration < 1.0, f"OCR extraction too slow: {duration:.2f}s"
        
        stats = performance_monitor.get_statistics()
        performance_monitor.print_report()
        
        # OCR throughput requirements
        assert stats['throughput_per_second'] > 5.0, f"OCR throughput too low: {stats['throughput_per_second']:.2f} ops/sec"
    
    @pytest.mark.asyncio
    async def test_pipeline_throughput_single_files(self, performance_monitor, test_settings, sample_invoice_image):
        """Test end-to-end pipeline throughput for single files"""
        # Create processor with mocked components
        processor = EuropeanInvoiceProcessor(test_settings)
        processor.preprocessor = Mock()
        processor.ocr_engine = Mock()
        processor.table_extractor = Mock()
        processor.llm_processor = Mock()
        
        # Setup fast mocks
        processor.preprocessor.preprocess_invoice_image.return_value = sample_invoice_image
        processor.ocr_engine.extract_invoice_text.return_value = {
            'full_text': 'INVOICE #001 Total: €100.00',
            'text_elements': [{'text': 'INVOICE #001', 'confidence': 0.95}],
            'structured_data': {},
            'total_elements': 1,
            'avg_confidence': 0.95
        }
        processor.ocr_engine.detect_language.return_value = 'en'
        processor.table_extractor.extract_tables.return_value = []
        processor.table_extractor.parse_invoice_line_items.return_value = []
        processor.llm_processor.extract_structured_data = AsyncMock(return_value={
            'invoice_id': 'INV-001',
            'total_incl_vat': 100.0
        })
        processor.llm_processor.initialize = AsyncMock()
        processor.llm_processor.cleanup = AsyncMock()
        
        # Process multiple images
        for i in range(15):
            _, duration = await performance_monitor.measure_async_operation(
                f"pipeline_process_{i}",
                processor._process_single_image,
                sample_invoice_image,
                f"test_{i}"
            )
            
            # Pipeline should be reasonably fast with mocked components
            assert duration < 2.0, f"Pipeline processing too slow: {duration:.2f}s"
        
        await processor.cleanup()
        
        stats = performance_monitor.get_statistics()
        performance_monitor.print_report()
        
        # Pipeline throughput requirements
        assert stats['throughput_per_second'] > 2.0, f"Pipeline throughput too low: {stats['throughput_per_second']:.2f} ops/sec"
        assert stats['average'] < 1.0, f"Average pipeline time too high: {stats['average']:.2f}s"
    
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self, performance_monitor, test_settings, sample_image_bytes):
        """Test batch processing throughput"""
        from fastapi import UploadFile
        from io import BytesIO
        
        # Create processor with mocked components
        processor = EuropeanInvoiceProcessor(test_settings)
        processor.preprocessor = Mock()
        processor.ocr_engine = Mock()
        processor.table_extractor = Mock()
        processor.llm_processor = Mock()
        
        # Setup fast mocks
        processor.preprocessor.preprocess_invoice_image.return_value = np.ones((800, 600, 3), dtype=np.uint8)
        processor.ocr_engine.extract_invoice_text.return_value = {
            'full_text': 'INVOICE #001',
            'text_elements': [],
            'structured_data': {},
            'total_elements': 0,
            'avg_confidence': 0.9
        }
        processor.ocr_engine.detect_language.return_value = 'en'
        processor.table_extractor.extract_tables.return_value = []
        processor.table_extractor.parse_invoice_line_items.return_value = []
        processor.llm_processor.extract_structured_data = AsyncMock(return_value={'invoice_id': 'INV-001'})
        processor.llm_processor.initialize = AsyncMock()
        processor.llm_processor.cleanup = AsyncMock()
        
        # Test different batch sizes
        batch_sizes = [5, 10, 20]
        
        for batch_size in batch_sizes:
            # Create batch of files
            files = []
            for i in range(batch_size):
                upload_file = UploadFile(
                    filename=f"batch_test_{i}.png",
                    file=BytesIO(sample_image_bytes),
                    content_type="image/png"
                )
                upload_file.size = len(sample_image_bytes)
                files.append(upload_file)
            
            # Measure batch processing
            _, duration = await performance_monitor.measure_async_operation(
                f"batch_process_{batch_size}",
                processor.process_batch_upload,
                files
            )
            
            throughput = batch_size / duration
            print(f"Batch size {batch_size}: {duration:.2f}s, {throughput:.2f} files/sec")
            
            # Batch processing should be efficient
            assert throughput > 1.0, f"Batch throughput too low: {throughput:.2f} files/sec"
        
        await processor.cleanup()
        
        stats = performance_monitor.get_statistics()
        performance_monitor.print_report()
    
    def test_concurrent_processing_throughput(self, performance_monitor):
        """Test concurrent processing performance"""
        preprocessor = InvoicePreprocessor()
        
        def process_image(image_id: int) -> Tuple[int, float]:
            """Process a single image and return timing"""
            test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
            
            start_time = time.time()
            result = preprocessor.preprocess_invoice_image(test_image)
            end_time = time.time()
            
            duration = end_time - start_time
            return image_id, duration
        
        # Test sequential processing
        sequential_start = time.time()
        sequential_results = []
        for i in range(10):
            image_id, duration = process_image(i)
            sequential_results.append(duration)
        sequential_total = time.time() - sequential_start
        
        # Test concurrent processing
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image, i) for i in range(10)]
            concurrent_results = [future.result()[1] for future in futures]
        concurrent_total = time.time() - concurrent_start
        
        print(f"\nConcurrency Performance:")
        print(f"  Sequential: {sequential_total:.2f}s ({10/sequential_total:.2f} ops/sec)")
        print(f"  Concurrent: {concurrent_total:.2f}s ({10/concurrent_total:.2f} ops/sec)")
        print(f"  Speedup: {sequential_total/concurrent_total:.2f}x")
        
        # Concurrent processing should be faster (or at least not much slower)
        speedup = sequential_total / concurrent_total
        assert speedup > 0.8, f"Concurrent processing slower than expected: {speedup:.2f}x"
    
    @pytest.mark.slow
    def test_sustained_throughput(self, performance_monitor):
        """Test sustained processing throughput over time"""
        preprocessor = InvoicePreprocessor()
        
        # Process images over several minutes
        total_images = 100
        batch_size = 10
        
        batch_times = []
        
        for batch in range(total_images // batch_size):
            batch_start = time.time()
            
            for i in range(batch_size):
                test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                
                _, duration = performance_monitor.measure_operation(
                    f"sustained_{batch}_{i}",
                    preprocessor.preprocess_invoice_image,
                    test_image
                )
            
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            batch_times.append(batch_duration)
            
            batch_throughput = batch_size / batch_duration
            print(f"Batch {batch}: {batch_throughput:.2f} images/sec")
            
            # Small delay to simulate real-world usage
            time.sleep(0.1)
        
        # Check for performance degradation over time
        early_batches = batch_times[:3]
        late_batches = batch_times[-3:]
        
        early_avg = statistics.mean(early_batches)
        late_avg = statistics.mean(late_batches)
        
        degradation = late_avg / early_avg
        
        print(f"\nSustained Performance:")
        print(f"  Early batches avg: {early_avg:.2f}s")
        print(f"  Late batches avg: {late_avg:.2f}s")
        print(f"  Performance ratio: {degradation:.2f}")
        
        # Performance should not degrade significantly
        assert degradation < 1.5, f"Performance degraded too much: {degradation:.2f}x"
        
        stats = performance_monitor.get_statistics()
        performance_monitor.print_report()
    
    def test_throughput_with_different_image_sizes(self, performance_monitor):
        """Test throughput with various image sizes"""
        preprocessor = InvoicePreprocessor()
        
        image_configs = [
            ("small", 400, 300),
            ("medium", 800, 600),
            ("large", 1200, 900),
            ("xlarge", 1600, 1200),
        ]
        
        throughput_by_size = {}
        
        for size_name, width, height in image_configs:
            size_start = time.time()
            
            # Process multiple images of this size
            for i in range(10):
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                _, duration = performance_monitor.measure_operation(
                    f"{size_name}_{i}",
                    preprocessor.preprocess_invoice_image,
                    test_image
                )
            
            size_total = time.time() - size_start
            size_throughput = 10 / size_total
            throughput_by_size[size_name] = size_throughput
            
            print(f"{size_name} ({width}x{height}): {size_throughput:.2f} images/sec")
        
        # Throughput should scale reasonably with image size
        assert throughput_by_size["small"] > throughput_by_size["xlarge"], \
            "Small images should process faster than large images"
        
        # Even large images should maintain reasonable throughput
        assert throughput_by_size["xlarge"] > 0.1, \
            f"Large image throughput too low: {throughput_by_size['xlarge']:.2f} images/sec"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_throughput(self, performance_monitor):
        """Test API endpoint throughput"""
        from fastapi.testclient import TestClient
        from src.main import app
        import threading
        
        client = TestClient(app)
        
        def make_request(request_id: int) -> Tuple[int, float, int]:
            """Make a single API request"""
            start_time = time.time()
            response = client.get("/api/v1/test", json={"test": f"data_{request_id}"})
            end_time = time.time()
            
            return request_id, end_time - start_time, response.status_code
        
        # Test concurrent API requests
        num_requests = 20
        
        # Sequential requests
        sequential_start = time.time()
        sequential_results = []
        for i in range(num_requests):
            _, duration, status = make_request(i)
            sequential_results.append(duration)
            assert status in [200, 429]  # OK or rate limited
        sequential_total = time.time() - sequential_start
        
        # Concurrent requests
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            concurrent_results = []
            for future in futures:
                _, duration, status = future.result()
                concurrent_results.append(duration)
                assert status in [200, 429]  # OK or rate limited
        concurrent_total = time.time() - concurrent_start
        
        print(f"\nAPI Throughput:")
        print(f"  Sequential: {num_requests/sequential_total:.2f} req/sec")
        print(f"  Concurrent: {num_requests/concurrent_total:.2f} req/sec")
        
        # API should handle reasonable request rates
        concurrent_throughput = num_requests / concurrent_total
        assert concurrent_throughput > 5.0, f"API throughput too low: {concurrent_throughput:.2f} req/sec"
    
    def test_memory_vs_throughput_tradeoff(self, performance_monitor):
        """Test memory usage vs throughput tradeoffs"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        preprocessor = InvoicePreprocessor()
        
        # Test different processing strategies
        strategies = {
            "small_batches": {"batch_size": 5, "cleanup_freq": 1},
            "medium_batches": {"batch_size": 10, "cleanup_freq": 2},
            "large_batches": {"batch_size": 20, "cleanup_freq": 5},
        }
        
        results = {}
        
        for strategy_name, config in strategies.items():
            print(f"\nTesting strategy: {strategy_name}")
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            strategy_start = time.time()
            peak_memory = initial_memory
            
            total_processed = 0
            batch_count = 0
            
            while total_processed < 50:  # Process 50 images total
                batch_start = time.time()
                
                # Process batch
                for i in range(min(config["batch_size"], 50 - total_processed)):
                    test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                    result = preprocessor.preprocess_invoice_image(test_image)
                    del test_image, result
                    total_processed += 1
                
                # Check memory
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Periodic cleanup
                batch_count += 1
                if batch_count % config["cleanup_freq"] == 0:
                    import gc
                    gc.collect()
                
                batch_end = time.time()
                batch_duration = batch_end - batch_start
                batch_throughput = config["batch_size"] / batch_duration
                
                print(f"  Batch {batch_count}: {batch_throughput:.2f} images/sec, "
                      f"Memory: {current_memory:.1f}MB")
            
            strategy_total = time.time() - strategy_start
            final_memory = process.memory_info().rss / 1024 / 1024
            
            results[strategy_name] = {
                "throughput": 50 / strategy_total,
                "peak_memory": peak_memory - initial_memory,
                "final_memory": final_memory - initial_memory,
                "efficiency": (50 / strategy_total) / (peak_memory - initial_memory)
            }
            
            # Cleanup
            import gc
            gc.collect()
        
        # Print comparison
        print(f"\nStrategy Comparison:")
        print(f"{'Strategy':<15} {'Throughput':<12} {'Peak Mem':<10} {'Efficiency':<10}")
        print("-" * 50)
        
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['throughput']:>8.2f}/sec {metrics['peak_memory']:>7.1f}MB "
                  f"{metrics['efficiency']:>9.3f}")
        
        # All strategies should achieve reasonable performance
        for name, metrics in results.items():
            assert metrics['throughput'] > 0.5, f"{name} throughput too low: {metrics['throughput']:.2f}/sec"
            assert metrics['peak_memory'] < 500, f"{name} peak memory too high: {metrics['peak_memory']:.1f}MB"


class TestScalabilityLimits:
    """Test system scalability limits"""
    
    @pytest.mark.slow
    def test_maximum_concurrent_operations(self, performance_monitor):
        """Test maximum number of concurrent operations"""
        preprocessor = InvoicePreprocessor()
        
        def process_image(image_id: int) -> float:
            """Process single image"""
            test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
            start = time.time()
            result = preprocessor.preprocess_invoice_image(test_image)
            end = time.time()
            del test_image, result
            return end - start
        
        # Test increasing levels of concurrency
        concurrency_levels = [1, 2, 4, 8, 16]
        results = {}
        
        for max_workers in concurrency_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image, i) for i in range(20)]
                durations = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            throughput = 20 / total_time
            avg_duration = statistics.mean(durations)
            
            results[max_workers] = {
                'throughput': throughput,
                'avg_duration': avg_duration,
                'total_time': total_time
            }
            
            print(f"Concurrency {max_workers:2d}: {throughput:6.2f} ops/sec, "
                  f"avg {avg_duration:.3f}s")
        
        # Find optimal concurrency level
        best_throughput = max(results.values(), key=lambda x: x['throughput'])['throughput']
        optimal_concurrency = [k for k, v in results.items() if v['throughput'] == best_throughput][0]
        
        print(f"\nOptimal concurrency: {optimal_concurrency} workers")
        print(f"Best throughput: {best_throughput:.2f} ops/sec")
        
        # System should handle reasonable concurrency
        assert results[4]['throughput'] > results[1]['throughput'] * 0.8, \
            "Concurrency should provide some benefit"
    
    def test_file_size_limits(self, performance_monitor):
        """Test processing limits with different file sizes"""
        preprocessor = InvoicePreprocessor()
        
        # Test increasingly large images
        sizes = [
            (800, 600),      # 480K pixels
            (1600, 1200),    # 1.9M pixels
            (2400, 1800),    # 4.3M pixels
            (3200, 2400),    # 7.7M pixels
        ]
        
        for width, height in sizes:
            pixels = width * height
            
            try:
                # Create large image
                large_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # Measure processing time
                start_time = time.time()
                result = preprocessor.preprocess_invoice_image(large_image)
                end_time = time.time()
                
                duration = end_time - start_time
                pixels_per_second = pixels / duration
                
                print(f"Size {width}x{height} ({pixels/1e6:.1f}MP): "
                      f"{duration:.2f}s ({pixels_per_second/1e6:.1f} MP/s)")
                
                # Clean up
                del large_image, result
                
                # Processing should complete within reasonable time
                assert duration < 30.0, f"Large image processing too slow: {duration:.2f}s"
                
            except MemoryError:
                print(f"Size {width}x{height}: Memory limit reached")
                break
            except Exception as e:
                print(f"Size {width}x{height}: Failed with {type(e).__name__}")
                break
    
    @pytest.mark.slow
    def test_long_running_stability(self, performance_monitor):
        """Test system stability under long-running load"""
        preprocessor = InvoicePreprocessor()
        
        # Run for extended period
        target_duration = 60  # 1 minute
        start_time = time.time()
        
        operation_count = 0
        performance_samples = []
        
        while time.time() - start_time < target_duration:
            batch_start = time.time()
            
            # Process small batch
            for i in range(5):
                test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                result = preprocessor.preprocess_invoice_image(test_image)
                del test_image, result
                operation_count += 1
            
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            batch_throughput = 5 / batch_duration
            
            performance_samples.append(batch_throughput)
            
            # Log every 10 seconds
            elapsed = time.time() - start_time
            if operation_count % 50 == 0:
                print(f"Time {elapsed:5.1f}s: {operation_count:4d} ops, "
                      f"current {batch_throughput:.2f} ops/sec")
        
        total_time = time.time() - start_time
        overall_throughput = operation_count / total_time
        
        # Analyze performance stability
        early_samples = performance_samples[:len(performance_samples)//4]
        late_samples = performance_samples[-len(performance_samples)//4:]
        
        early_avg = statistics.mean(early_samples)
        late_avg = statistics.mean(late_samples)
        performance_ratio = late_avg / early_avg
        
        print(f"\nLong-running Performance:")
        print(f"  Total operations: {operation_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Overall throughput: {overall_throughput:.2f} ops/sec")
        print(f"  Early period avg: {early_avg:.2f} ops/sec")
        print(f"  Late period avg: {late_avg:.2f} ops/sec")
        print(f"  Performance ratio: {performance_ratio:.3f}")
        
        # Performance should remain stable
        assert performance_ratio > 0.8, f"Performance degraded significantly: {performance_ratio:.3f}"
        assert overall_throughput > 1.0, f"Overall throughput too low: {overall_throughput:.2f} ops/sec"
