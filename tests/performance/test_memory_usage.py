"""
Performance tests for memory usage
"""
import pytest
import psutil
import os
import time
import gc
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.core.pipeline import EuropeanInvoiceProcessor
from src.core.preprocessor import InvoicePreprocessor
from src.core.ocr_engine import InvoiceOCREngine


class TestMemoryUsage:
    """Test memory usage patterns"""
    
    @pytest.fixture
    def memory_monitor(self):
        """Memory monitoring helper"""
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.initial_memory = self.get_memory_mb()
                self.peak_memory = self.initial_memory
                self.measurements = []
            
            def get_memory_mb(self) -> float:
                """Get current memory usage in MB"""
                return self.process.memory_info().rss / 1024 / 1024
            
            def measure(self, label: str = ""):
                """Take a memory measurement"""
                current = self.get_memory_mb()
                self.peak_memory = max(self.peak_memory, current)
                self.measurements.append({
                    'label': label,
                    'memory_mb': current,
                    'increase_mb': current - self.initial_memory,
                    'timestamp': time.time()
                })
                return current
            
            def get_peak_increase(self) -> float:
                """Get peak memory increase from initial"""
                return self.peak_memory - self.initial_memory
            
            def get_final_increase(self) -> float:
                """Get final memory increase from initial"""
                return self.get_memory_mb() - self.initial_memory
            
            def force_gc(self):
                """Force garbage collection"""
                gc.collect()
                time.sleep(0.1)  # Allow time for cleanup
        
        return MemoryMonitor()
    
    def test_preprocessor_memory_usage(self, memory_monitor):
        """Test memory usage of image preprocessor"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        memory_monitor.measure("preprocessor_created")
        
        # Test with various image sizes
        image_sizes = [(800, 600), (1600, 1200), (3200, 2400)]
        
        for width, height in image_sizes:
            # Create test image
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            memory_monitor.measure(f"image_created_{width}x{height}")
            
            # Process image
            processed = preprocessor.preprocess_invoice_image(test_image)
            memory_monitor.measure(f"image_processed_{width}x{height}")
            
            # Clean up
            del test_image, processed
            memory_monitor.force_gc()
            memory_monitor.measure(f"cleaned_up_{width}x{height}")
        
        final_increase = memory_monitor.get_final_increase()
        peak_increase = memory_monitor.get_peak_increase()
        
        # Assert reasonable memory usage
        assert final_increase < 50, f"Final memory increase too high: {final_increase:.1f}MB"
        assert peak_increase < 500, f"Peak memory increase too high: {peak_increase:.1f}MB"
        
        # Print memory measurements for analysis
        for measurement in memory_monitor.measurements:
            print(f"{measurement['label']}: {measurement['memory_mb']:.1f}MB (+{measurement['increase_mb']:.1f}MB)")
    
    @patch('src.core.ocr_engine.EASYOCR_AVAILABLE', True)
    @patch('src.core.ocr_engine.easyocr.Reader')
    def test_ocr_engine_memory_usage(self, mock_reader, memory_monitor):
        """Test memory usage of OCR engine"""
        memory_monitor.measure("start")
        
        # Mock OCR to avoid loading real models
        mock_reader_instance = Mock()
        mock_reader_instance.readtext.return_value = [
            ([[50, 50], [300, 50], [300, 80], [50, 80]], 'TEST TEXT', 0.95)
        ]
        mock_reader.return_value = mock_reader_instance
        
        # Create OCR engine
        ocr_engine = InvoiceOCREngine(engine_type="easyocr", languages=['en'])
        memory_monitor.measure("ocr_engine_created")
        
        # Process multiple images
        for i in range(10):
            test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
            memory_monitor.measure(f"image_{i}_created")
            
            result = ocr_engine.extract_invoice_text(test_image)
            memory_monitor.measure(f"image_{i}_processed")
            
            assert isinstance(result, dict)
            
            # Clean up
            del test_image, result
            if i % 3 == 0:  # Periodic cleanup
                memory_monitor.force_gc()
        
        memory_monitor.force_gc()
        memory_monitor.measure("final_cleanup")
        
        final_increase = memory_monitor.get_final_increase()
        peak_increase = memory_monitor.get_peak_increase()
        
        # Assert reasonable memory usage
        assert final_increase < 100, f"Final memory increase too high: {final_increase:.1f}MB"
        assert peak_increase < 200, f"Peak memory increase too high: {peak_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_processor_memory_usage_single_file(self, memory_monitor, test_settings, sample_invoice_image):
        """Test memory usage when processing single file"""
        memory_monitor.measure("start")
        
        # Create processor with mocked components
        processor = EuropeanInvoiceProcessor(test_settings)
        processor.preprocessor = Mock()
        processor.ocr_engine = Mock()
        processor.table_extractor = Mock()
        processor.llm_processor = Mock()
        
        # Setup mocks
        processor.preprocessor.preprocess_invoice_image.return_value = sample_invoice_image
        processor.ocr_engine.extract_invoice_text.return_value = {
            'full_text': 'Test invoice',
            'text_elements': [],
            'structured_data': {},
            'total_elements': 0,
            'avg_confidence': 0.9
        }
        processor.ocr_engine.detect_language.return_value = 'en'
        processor.table_extractor.extract_tables.return_value = []
        processor.table_extractor.parse_invoice_line_items.return_value = []
        
        from unittest.mock import AsyncMock
        processor.llm_processor.extract_structured_data = AsyncMock(return_value={
            'invoice_id': 'TEST-001',
            'total_incl_vat': 100.0
        })
        processor.llm_processor.initialize = AsyncMock()
        processor.llm_processor.cleanup = AsyncMock()
        
        memory_monitor.measure("processor_created")
        
        # Process single image multiple times
        for i in range(5):
            result = await processor._process_single_image(sample_invoice_image, f"test_{i}")
            memory_monitor.measure(f"processed_{i}")
            
            assert isinstance(result, dict)
            
            # Clean up result
            del result
            if i % 2 == 0:
                memory_monitor.force_gc()
        
        await processor.cleanup()
        memory_monitor.force_gc()
        memory_monitor.measure("final_cleanup")
        
        final_increase = memory_monitor.get_final_increase()
        peak_increase = memory_monitor.get_peak_increase()
        
        # Assert reasonable memory usage
        assert final_increase < 100, f"Final memory increase too high: {final_increase:.1f}MB"
        assert peak_increase < 300, f"Peak memory increase too high: {peak_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_processor_memory_usage_batch(self, memory_monitor, test_settings, sample_image_bytes):
        """Test memory usage during batch processing"""
        memory_monitor.measure("start")
        
        # Create processor with mocked components
        processor = EuropeanInvoiceProcessor(test_settings)
        processor.preprocessor = Mock()
        processor.ocr_engine = Mock()
        processor.table_extractor = Mock()
        processor.llm_processor = Mock()
        
        # Setup mocks (same as above)
        processor.preprocessor.preprocess_invoice_image.return_value = np.ones((800, 600, 3), dtype=np.uint8)
        processor.ocr_engine.extract_invoice_text.return_value = {
            'full_text': 'Test invoice',
            'text_elements': [],
            'structured_data': {},
            'total_elements': 0,
            'avg_confidence': 0.9
        }
        processor.ocr_engine.detect_language.return_value = 'en'
        processor.table_extractor.extract_tables.return_value = []
        processor.table_extractor.parse_invoice_line_items.return_value = []
        
        from unittest.mock import AsyncMock
        processor.llm_processor.extract_structured_data = AsyncMock(return_value={
            'invoice_id': 'TEST-001',
            'total_incl_vat': 100.0
        })
        processor.llm_processor.initialize = AsyncMock()
        processor.llm_processor.cleanup = AsyncMock()
        
        memory_monitor.measure("processor_created")
        
        # Create batch of files
        from fastapi import UploadFile
        from io import BytesIO
        
        batch_sizes = [5, 10, 20]
        
        for batch_size in batch_sizes:
            memory_monitor.measure(f"batch_{batch_size}_start")
            
            files = []
            for i in range(batch_size):
                upload_file = UploadFile(
                    filename=f"test_{i}.png",
                    file=BytesIO(sample_image_bytes),
                    content_type="image/png"
                )
                upload_file.size = len(sample_image_bytes)
                files.append(upload_file)
            
            memory_monitor.measure(f"batch_{batch_size}_files_created")
            
            # Process batch
            results = await processor.process_batch_upload(files)
            memory_monitor.measure(f"batch_{batch_size}_processed")
            
            assert len(results) == batch_size
            
            # Clean up
            del files, results
            memory_monitor.force_gc()
            memory_monitor.measure(f"batch_{batch_size}_cleaned")
        
        await processor.cleanup()
        memory_monitor.force_gc()
        memory_monitor.measure("final_cleanup")
        
        final_increase = memory_monitor.get_final_increase()
        peak_increase = memory_monitor.get_peak_increase()
        
        # Assert reasonable memory usage
        assert final_increase < 150, f"Final memory increase too high: {final_increase:.1f}MB"
        assert peak_increase < 500, f"Peak memory increase too high: {peak_increase:.1f}MB"
    
    def test_large_image_memory_usage(self, memory_monitor):
        """Test memory usage with large images"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        memory_monitor.measure("preprocessor_created")
        
        # Test with increasingly large images
        sizes = [
            (1920, 1080),   # Full HD
            (3840, 2160),   # 4K
            (7680, 4320)    # 8K
        ]
        
        for width, height in sizes:
            memory_monitor.measure(f"creating_{width}x{height}")
            
            # Create large image
            large_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            memory_monitor.measure(f"created_{width}x{height}")
            
            # Process image
            try:
                processed = preprocessor.preprocess_invoice_image(large_image)
                memory_monitor.measure(f"processed_{width}x{height}")
                
                assert processed is not None
                
                # Clean up immediately
                del processed
            except MemoryError:
                pytest.skip(f"Insufficient memory for {width}x{height} image")
            
            del large_image
            memory_monitor.force_gc()
            memory_monitor.measure(f"cleaned_{width}x{height}")
        
        final_increase = memory_monitor.get_final_increase()
        peak_increase = memory_monitor.get_peak_increase()
        
        # Large images will use more memory, but should clean up
        assert final_increase < 200, f"Final memory increase too high: {final_increase:.1f}MB"
        assert peak_increase < 2000, f"Peak memory increase too high: {peak_increase:.1f}MB"
    
    def test_memory_leak_detection(self, memory_monitor):
        """Test for memory leaks in repeated operations"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        memory_monitor.measure("preprocessor_created")
        
        # Baseline measurement
        baseline_measurements = []
        
        # Perform operations in cycles to detect leaks
        for cycle in range(5):
            cycle_start = memory_monitor.get_memory_mb()
            
            # Perform multiple operations
            for i in range(10):
                test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                processed = preprocessor.preprocess_invoice_image(test_image)
                del test_image, processed
            
            # Force cleanup
            memory_monitor.force_gc()
            cycle_end = memory_monitor.get_memory_mb()
            
            cycle_increase = cycle_end - cycle_start
            baseline_measurements.append(cycle_increase)
            
            memory_monitor.measure(f"cycle_{cycle}_completed")
        
        # Check for memory leak pattern
        # Memory usage should not consistently increase across cycles
        if len(baseline_measurements) >= 3:
            # Check if memory consistently increases
            increases = [baseline_measurements[i+1] - baseline_measurements[i] 
                        for i in range(len(baseline_measurements)-1)]
            
            # Most differences should be small
            large_increases = sum(1 for inc in increases if inc > 10)  # > 10MB
            
            assert large_increases <= 1, f"Potential memory leak detected: {increases}"
    
    def test_memory_usage_with_errors(self, memory_monitor):
        """Test memory usage when errors occur"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        memory_monitor.measure("preprocessor_created")
        
        # Test with various error conditions
        error_conditions = [
            None,  # None input
            np.array([]),  # Empty array
            np.random.randint(0, 255, (10, 10), dtype=np.uint8),  # Too small
            np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)  # Very large
        ]
        
        for i, condition in enumerate(error_conditions):
            try:
                memory_monitor.measure(f"error_test_{i}_start")
                
                if condition is not None:
                    result = preprocessor.preprocess_invoice_image(condition)
                else:
                    result = preprocessor.preprocess_invoice_image(None)
                
                memory_monitor.measure(f"error_test_{i}_completed")
                
                # Clean up
                if result is not None:
                    del result
                if condition is not None and condition.size > 0:
                    del condition
                
            except Exception:
                # Expected for some error conditions
                pass
            
            memory_monitor.force_gc()
            memory_monitor.measure(f"error_test_{i}_cleaned")
        
        final_increase = memory_monitor.get_final_increase()
        
        # Memory should not increase significantly even with errors
        assert final_increase < 100, f"Memory increase too high after errors: {final_increase:.1f}MB"
    
    @pytest.mark.slow
    def test_long_running_memory_stability(self, memory_monitor):
        """Test memory stability over long-running operations"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        memory_monitor.measure("preprocessor_created")
        
        # Simulate long-running processing
        for batch in range(10):  # 10 batches
            batch_start = memory_monitor.get_memory_mb()
            
            for i in range(20):  # 20 images per batch
                test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
                processed = preprocessor.preprocess_invoice_image(test_image)
                
                # Simulate some processing
                time.sleep(0.01)
                
                del test_image, processed
            
            # Periodic cleanup
            memory_monitor.force_gc()
            
            batch_end = memory_monitor.get_memory_mb()
            batch_increase = batch_end - batch_start
            
            memory_monitor.measure(f"batch_{batch}_completed")
            
            # Each batch should not increase memory significantly
            assert batch_increase < 50, f"Batch {batch} memory increase too high: {batch_increase:.1f}MB"
        
        final_increase = memory_monitor.get_final_increase()
        
        # Final memory should be reasonable after long operation
        assert final_increase < 100, f"Long-running memory increase too high: {final_increase:.1f}MB"
    
    def test_memory_usage_profiling(self, memory_monitor):
        """Detailed memory profiling of components"""
        memory_monitor.measure("profiling_start")
        
        # Test individual components
        components = {
            'preprocessor': InvoicePreprocessor(),
        }
        
        for name, component in components.items():
            memory_monitor.measure(f"{name}_created")
            
            # Test different operations
            test_image = np.random.randint(0, 255, (1200, 800, 3), dtype=np.uint8)
            memory_monitor.measure(f"{name}_image_created")
            
            if hasattr(component, 'preprocess_invoice_image'):
                result = component.preprocess_invoice_image(test_image)
                memory_monitor.measure(f"{name}_processed")
                del result
            
            del test_image
            memory_monitor.force_gc()
            memory_monitor.measure(f"{name}_cleaned")
        
        # Generate memory profile report
        print("\nMemory Profile Report:")
        print("-" * 50)
        
        for i, measurement in enumerate(memory_monitor.measurements):
            if i == 0:
                continue
            
            prev_memory = memory_monitor.measurements[i-1]['memory_mb']
            current_memory = measurement['memory_mb']
            delta = current_memory - prev_memory
            
            print(f"{measurement['label']:<25} {current_memory:>8.1f}MB {delta:>+7.1f}MB")
        
        print("-" * 50)
        print(f"{'Total Increase':<25} {memory_monitor.get_final_increase():>+8.1f}MB")
        print(f"{'Peak Increase':<25} {memory_monitor.get_peak_increase():>+8.1f}MB")


class TestMemoryOptimizations:
    """Test memory optimization techniques"""
    
    def test_image_size_optimization(self, memory_monitor):
        """Test memory usage with different image processing strategies"""
        memory_monitor.measure("start")
        
        preprocessor = InvoicePreprocessor()
        large_image = np.random.randint(0, 255, (4000, 3000, 3), dtype=np.uint8)
        memory_monitor.measure("large_image_created")
        
        # Strategy 1: Direct processing
        result1 = preprocessor.preprocess_invoice_image(large_image)
        memory_monitor.measure("direct_processing")
        del result1
        
        # Strategy 2: Resize first, then process
        resized = preprocessor._enhance_dpi(large_image, target_dpi=200)  # Smaller DPI
        memory_monitor.measure("resized_first")
        result2 = preprocessor.preprocess_invoice_image(resized)
        memory_monitor.measure("resized_processed")
        del resized, result2
        
        del large_image
        memory_monitor.force_gc()
        memory_monitor.measure("cleanup_complete")
        
        # Resizing first should use less peak memory
        peak_increase = memory_monitor.get_peak_increase()
        assert peak_increase < 1000, f"Peak memory too high: {peak_increase:.1f}MB"
    
    def test_batch_size_optimization(self, memory_monitor, test_settings, sample_image_bytes):
        """Test optimal batch sizes for memory usage"""
        memory_monitor.measure("start")
        
        from fastapi import UploadFile
        from io import BytesIO
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            memory_monitor.force_gc()
            batch_start = memory_monitor.get_memory_mb()
            
            # Create batch
            files = []
            for i in range(batch_size):
                upload_file = UploadFile(
                    filename=f"test_{i}.png",
                    file=BytesIO(sample_image_bytes),
                    content_type="image/png"
                )
                files.append(upload_file)
            
            batch_peak = memory_monitor.get_memory_mb()
            batch_usage = batch_peak - batch_start
            memory_usage[batch_size] = batch_usage
            
            # Clean up
            del files
            memory_monitor.force_gc()
            
            memory_monitor.measure(f"batch_size_{batch_size}")
        
        # Memory usage should scale reasonably with batch size
        print(f"\nBatch Size Memory Usage:")
        for size, usage in memory_usage.items():
            print(f"  {size:2d} files: {usage:6.1f}MB")
        
        # Find optimal batch size (good memory/throughput trade-off)
        # Memory per file should not increase dramatically
        per_file_usage = {size: usage/size for size, usage in memory_usage.items()}
        
        # Variance in per-file usage should be reasonable
        usage_values = list(per_file_usage.values())
        max_usage = max(usage_values)
        min_usage = min(usage_values)
        
        assert max_usage / min_usage < 3, "Memory usage per file varies too much with batch size"


class TestMemoryLeakDetection:
    """Specific tests for memory leak detection"""
    
    def test_cyclic_reference_cleanup(self, memory_monitor):
        """Test that cyclic references are properly cleaned up"""
        memory_monitor.measure("start")
        
        # Create objects with potential cyclic references
        for cycle in range(10):
            # Simulate processor creation and cleanup
            objects = []
            
            for i in range(100):
                # Create mock objects that might have cyclic references
                obj = Mock()
                obj.parent = Mock()
                obj.parent.child = obj  # Cyclic reference
                objects.append(obj)
            
            memory_monitor.measure(f"cycle_{cycle}_objects_created")
            
            # Clear references
            for obj in objects:
                obj.parent = None
            del objects
            
            # Force cleanup
            memory_monitor.force_gc()
            memory_monitor.measure(f"cycle_{cycle}_cleaned")
        
        final_increase = memory_monitor.get_final_increase()
        
        # Should not accumulate memory across cycles
        assert final_increase < 50, f"Potential memory leak detected: {final_increase:.1f}MB"
    
    def test_file_handle_cleanup(self, memory_monitor, temp_dir):
        """Test that file handles are properly closed"""
        memory_monitor.measure("start")
        
        # Create and process many temporary files
        for i in range(100):
            temp_file = temp_dir / f"test_{i}.txt"
            temp_file.write_text(f"Test content {i}")
            
            # Read file (simulating file processing)
            with open(temp_file, 'r') as f:
                content = f.read()
            
            # Clean up
            temp_file.unlink()
            
            if i % 10 == 0:
                memory_monitor.measure(f"files_processed_{i}")
        
        memory_monitor.force_gc()
        memory_monitor.measure("files_cleanup_complete")
        
        final_increase = memory_monitor.get_final_increase()
        
        # File operations should not leak memory
        assert final_increase < 30, f"File handling memory leak: {final_increase:.1f}MB"
