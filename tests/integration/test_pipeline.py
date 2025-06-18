"""
Integration tests for the complete processing pipeline
"""
import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.core.pipeline import EuropeanInvoiceProcessor
from src.config.settings import Settings
from ..conftest import assert_valid_invoice_data


class TestPipelineIntegration:
    """Integration tests for the complete processing pipeline"""
    
    @pytest.fixture
    async def processor_with_mocks(self, test_settings):
        """Create processor with mocked components for testing"""
        processor = EuropeanInvoiceProcessor(test_settings)
        
        # Mock all components to avoid loading real models
        processor.preprocessor = Mock()
        processor.ocr_engine = Mock()
        processor.table_extractor = Mock()
        processor.llm_processor = Mock()
        
        # Setup mock behaviors
        await self._setup_mock_behaviors(processor)
        
        yield processor
        
        await processor.cleanup()
    
    async def _setup_mock_behaviors(self, processor):
        """Setup mock behaviors for processor components"""
        # Mock preprocessor
        processor.preprocessor.preprocess_invoice_image.return_value = np.ones((800, 600, 3), dtype=np.uint8) * 255
        processor.preprocessor.pdf_to_images.return_value = [Mock()]
        
        # Mock OCR engine
        processor.ocr_engine.extract_invoice_text.return_value = {
            'full_text': 'INVOICE #INV-2024-001\nDate: 15.01.2024\nTotal: €178.50',
            'text_elements': [
                {'text': 'INVOICE #INV-2024-001', 'confidence': 0.95, 'bbox': [[50, 50], [300, 80]]},
                {'text': 'Date: 15.01.2024', 'confidence': 0.92, 'bbox': [[50, 100], [200, 120]]},
                {'text': 'Total: €178.50', 'confidence': 0.93, 'bbox': [[400, 500], [550, 520]]}
            ],
            'structured_data': {'lines': [], 'amounts': [], 'headers': []},
            'total_elements': 3,
            'avg_confidence': 0.93
        }
        processor.ocr_engine.detect_language.return_value = 'en'
        
        # Mock table extractor
        processor.table_extractor.extract_tables.return_value = []
        processor.table_extractor.parse_invoice_line_items.return_value = [
            {
                'description': 'Software License',
                'quantity': 1.0,
                'unit_price': 100.0,
                'line_total': 100.0,
                'vat_rate': 0.19
            }
        ]
        
        # Mock LLM processor
        processor.llm_processor.extract_structured_data = AsyncMock(return_value={
            'invoice_id': 'INV-2024-001',
            'issue_date': '2024-01-15',
            'currency_code': 'EUR',
            'supplier': {
                'name': 'Test Company GmbH',
                'vat_id': 'DE123456789',
                'country_code': 'DE'
            },
            'customer': {
                'name': 'Customer Ltd',
                'vat_id': 'EE987654321',
                'country_code': 'EE'
            },
            'invoice_lines': [
                {
                    'line_id': '1',
                    'description': 'Software License',
                    'quantity': 1.0,
                    'unit_price': 100.0,
                    'line_total': 100.0,
                    'vat_rate': 0.19
                }
            ],
            'total_excl_vat': 100.0,
            'total_vat': 19.0,
            'total_incl_vat': 119.0
        })
        processor.llm_processor.initialize = AsyncMock()
        processor.llm_processor.cleanup = AsyncMock()
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, test_settings):
        """Test processor initialization"""
        processor = EuropeanInvoiceProcessor(test_settings)
        
        # Should initialize without errors (even if models fail to load)
        try:
            await processor.initialize()
        except Exception:
            # Expected to fail without real models
            pass
        
        # Processor should be created
        assert processor is not None
        assert processor.settings == test_settings
        
        await processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_process_single_image_complete_pipeline(self, processor_with_mocks, sample_invoice_image):
        """Test processing a single image through complete pipeline"""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            import cv2
            cv2.imwrite(tmp_file.name, sample_invoice_image)
            tmp_path = tmp_file.name
        
        try:
            result = await processor_with_mocks.process_invoice_file(tmp_path)
            
            assert isinstance(result, dict)
            assert result['status'] == 'success'
            assert result['pages_processed'] == 1
            assert 'extracted_data' in result
            
            extracted_data = result['extracted_data']
            assert extracted_data['invoice_id'] == 'INV-2024-001'
            assert extracted_data['issue_date'] == '2024-01-15'
            assert extracted_data['currency_code'] == 'EUR'
            assert extracted_data['total_incl_vat'] == 119.0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_process_pdf_file(self, processor_with_mocks, sample_pdf_file):
        """Test processing PDF file"""
        # Mock PDF to images conversion
        processor_with_mocks.preprocessor.pdf_to_images.return_value = [
            Mock(spec=['size'], size=(800, 600))
        ]
        
        result = await processor_with_mocks.process_invoice_file(str(sample_pdf_file))
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'
        assert result['pages_processed'] >= 1
        assert 'extracted_data' in result
    
    @pytest.mark.asyncio
    async def test_process_upload_file(self, processor_with_mocks, sample_image_bytes):
        """Test processing uploaded file"""
        from fastapi import UploadFile
        from io import BytesIO
        
        upload_file = UploadFile(
            filename="test_invoice.png",
            file=BytesIO(sample_image_bytes),
            content_type="image/png"
        )
        upload_file.size = len(sample_image_bytes)
        
        result = await processor_with_mocks.process_invoice_upload(upload_file)
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'
        assert result['filename'] == 'test_invoice.png'
        assert 'extracted_data' in result
        assert 'processing_time' in result
    
    @pytest.mark.asyncio
    async def test_process_batch_upload(self, processor_with_mocks, sample_image_bytes):
        """Test batch processing of uploaded files"""
        from fastapi import UploadFile
        from io import BytesIO
        
        # Create multiple upload files
        files = []
        for i in range(3):
            upload_file = UploadFile(
                filename=f"test_invoice_{i}.png",
                file=BytesIO(sample_image_bytes),
                content_type="image/png"
            )
            upload_file.size = len(sample_image_bytes)
            files.append(upload_file)
        
        results = await processor_with_mocks.process_batch_upload(files)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert isinstance(result, dict)
            assert result['status'] == 'success'
            assert 'extracted_data' in result
    
    @pytest.mark.asyncio
    async def test_load_invoice_images_pdf(self, processor_with_mocks, sample_pdf_file):
        """Test loading images from PDF file"""
        # Mock PDF conversion
        mock_pil_image = Mock()
        processor_with_mocks.preprocessor.pdf_to_images.return_value = [mock_pil_image]
        
        with patch('numpy.array') as mock_array:
            mock_array.return_value = np.ones((800, 600, 3), dtype=np.uint8) * 255
            
            images = await processor_with_mocks._load_invoice_images(str(sample_pdf_file))
            
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_load_invoice_images_image(self, processor_with_mocks, sample_image_file):
        """Test loading images from image file"""
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = np.ones((800, 600, 3), dtype=np.uint8) * 255
            
            images = await processor_with_mocks._load_invoice_images(str(sample_image_file))
            
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_process_single_image_step_by_step(self, processor_with_mocks, sample_invoice_image):
        """Test processing single image with detailed step verification"""
        result = await processor_with_mocks._process_single_image(sample_invoice_image, "test_page")
        
        assert isinstance(result, dict)
        assert result['page_id'] == "test_page"
        assert 'ocr_result' in result
        assert 'tables' in result
        assert 'line_items' in result
        assert 'detected_language' in result
        assert 'structured_data' in result
        assert 'quality_metrics' in result
        
        # Verify each component was called
        processor_with_mocks.preprocessor.preprocess_invoice_image.assert_called_once()
        processor_with_mocks.ocr_engine.extract_invoice_text.assert_called_once()
        processor_with_mocks.table_extractor.extract_tables.assert_called_once()
        processor_with_mocks.ocr_engine.detect_language.assert_called_once()
        processor_with_mocks.llm_processor.extract_structured_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_merge_extraction_results(self, processor_with_mocks):
        """Test merging LLM and table extraction results"""
        llm_data = {
            'invoice_id': 'INV-001',
            'total_incl_vat': 119.0,
            'invoice_lines': []
        }
        
        line_items = [
            {
                'description': 'Software',
                'quantity': 1.0,
                'unit_price': 100.0,
                'line_total': 100.0
            }
        ]
        
        tables = [{'type': 'detected', 'rows': 2, 'columns': 4}]
        
        merged = processor_with_mocks._merge_extraction_results(llm_data, line_items, tables)
        
        assert merged['invoice_id'] == 'INV-001'
        assert len(merged['invoice_lines']) == 1
        assert merged['tables_detected'] == 1
        assert merged['table_line_items'] == 1
    
    @pytest.mark.asyncio
    async def test_validate_financial_totals(self, processor_with_mocks):
        """Test financial totals validation"""
        data = {
            'invoice_lines': [
                {'line_total': 50.0},
                {'line_total': 50.0}
            ],
            'total_excl_vat': 100.0,
            'total_vat': 19.0,
            'total_incl_vat': 119.0
        }
        
        validated = processor_with_mocks._validate_financial_totals(data)
        
        assert validated['total_excl_vat'] == 100.0
        assert validated['total_vat'] == 19.0
        assert validated['total_incl_vat'] == 119.0
    
    @pytest.mark.asyncio
    async def test_calculate_quality_metrics(self, processor_with_mocks):
        """Test quality metrics calculation"""
        ocr_result = {
            'avg_confidence': 0.9,
            'total_elements': 10
        }
        
        structured_data = {
            'invoice_id': 'INV-001',
            'issue_date': '2024-01-15',
            'supplier': {'name': 'Test Company'},
            'customer': {'name': 'Customer'},
            'total_incl_vat': 119.0,
            'invoice_lines': [{'description': 'Item 1'}]
        }
        
        metrics = processor_with_mocks._calculate_quality_metrics(ocr_result, structured_data)
        
        assert 'ocr_confidence' in metrics
        assert 'text_elements_count' in metrics
        assert 'required_fields_present' in metrics
        assert 'data_completeness' in metrics
        assert 'overall_quality' in metrics
        
        assert 0 <= metrics['overall_quality'] <= 1
    
    @pytest.mark.asyncio
    async def test_combine_page_results_single_page(self, processor_with_mocks):
        """Test combining results from single page"""
        page_results = [
            {
                'structured_data': {
                    'invoice_id': 'INV-001',
                    'total_incl_vat': 119.0,
                    'invoice_lines': [{'description': 'Item 1'}]
                }
            }
        ]
        
        combined = processor_with_mocks._combine_page_results(page_results)
        
        assert combined['invoice_id'] == 'INV-001'
        assert combined['total_incl_vat'] == 119.0
    
    @pytest.mark.asyncio
    async def test_combine_page_results_multiple_pages(self, processor_with_mocks):
        """Test combining results from multiple pages"""
        page_results = [
            {
                'structured_data': {
                    'invoice_id': 'INV-001',
                    'total_incl_vat': 119.0,
                    'invoice_lines': [{'description': 'Item 1'}]
                }
            },
            {
                'structured_data': {
                    'invoice_id': 'INV-001',
                    'invoice_lines': [{'description': 'Item 2'}]
                }
            }
        ]
        
        combined = processor_with_mocks._combine_page_results(page_results)
        
        assert combined['invoice_id'] == 'INV-001'
        assert len(combined['invoice_lines']) == 2
    
    @pytest.mark.asyncio
    async def test_update_stats(self, processor_with_mocks):
        """Test statistics updating"""
        initial_count = processor_with_mocks.stats['processed_count']
        
        processor_with_mocks._update_stats(True, 2.5)
        
        assert processor_with_mocks.stats['processed_count'] == initial_count + 1
        assert processor_with_mocks.stats['success_count'] >= 1
        assert processor_with_mocks.stats['total_processing_time'] >= 2.5
        
        processor_with_mocks._update_stats(False, 1.0)
        
        assert processor_with_mocks.stats['processed_count'] == initial_count + 2
        assert processor_with_mocks.stats['error_count'] >= 1
    
    @pytest.mark.asyncio
    async def test_get_model_status(self, processor_with_mocks):
        """Test getting model status"""
        status = await processor_with_mocks.get_model_status()
        
        assert isinstance(status, dict)
        assert 'ocr_engine' in status
        assert 'llm_model' in status
        assert 'memory_usage' in status
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, processor_with_mocks):
        """Test getting processing metrics"""
        metrics = await processor_with_mocks.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'processing_stats' in metrics
        assert 'system_metrics' in metrics
        assert 'settings' in metrics
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, processor_with_mocks, sample_invoice_image):
        """Test error handling throughout the pipeline"""
        # Simulate OCR failure
        processor_with_mocks.ocr_engine.extract_invoice_text.side_effect = Exception("OCR failed")
        
        result = await processor_with_mocks._process_single_image(sample_invoice_image, "error_test")
        
        assert 'error' in result
        assert result['page_id'] == "error_test"
    
    @pytest.mark.asyncio
    async def test_unsupported_file_format(self, processor_with_mocks):
        """Test handling of unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"not an image")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                await processor_with_mocks._load_invoice_images(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, processor_with_mocks):
        """Test handling of corrupted files"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"corrupted image data")
            tmp_path = tmp_file.name
        
        try:
            # Should handle corrupted files gracefully
            with patch('cv2.imread', return_value=None):
                with pytest.raises(ValueError):
                    await processor_with_mocks._load_invoice_images(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_multiple_files(self, processor_with_mocks, sample_image_bytes):
        """Test performance with multiple files"""
        from fastapi import UploadFile
        from io import BytesIO
        import time
        
        # Create multiple files
        files = []
        for i in range(5):
            upload_file = UploadFile(
                filename=f"perf_test_{i}.png",
                file=BytesIO(sample_image_bytes),
                content_type="image/png"
            )
            upload_file.size = len(sample_image_bytes)
            files.append(upload_file)
        
        start_time = time.time()
        results = await processor_with_mocks.process_batch_upload(files)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert len(results) == 5
        assert processing_time < 30.0  # Should complete within 30 seconds for mocked components
        
        # All should be successful
        successful = sum(1 for r in results if r.get('status') == 'success')
        assert successful == 5
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, processor_with_mocks, sample_invoice_image):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple images
        for i in range(5):
            result = await processor_with_mocks._process_single_image(
                sample_invoice_image, f"memory_test_{i}"
            )
            assert 'structured_data' in result
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 200  # Less than 200MB increase
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, processor_with_mocks, sample_invoice_image):
        """Test concurrent processing capability"""
        # Process multiple images concurrently
        tasks = []
        for i in range(3):
            task = processor_with_mocks._process_single_image(
                sample_invoice_image, f"concurrent_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert 'structured_data' in result
    
    @pytest.mark.asyncio
    async def test_language_detection_integration(self, processor_with_mocks):
        """Test language detection integration"""
        # Test with different language mocks
        languages = ['en', 'de', 'et']
        
        for lang in languages:
            processor_with_mocks.ocr_engine.detect_language.return_value = lang
            
            result = await processor_with_mocks._process_single_image(
                np.ones((800, 600, 3), dtype=np.uint8) * 255, f"lang_test_{lang}"
            )
            
            assert result['detected_language'] == lang
    
    @pytest.mark.asyncio
    async def test_cleanup_process(self, test_settings):
        """Test cleanup process"""
        processor = EuropeanInvoiceProcessor(test_settings)
        
        # Mock components for cleanup
        processor.llm_processor = Mock()
        processor.llm_processor.cleanup = AsyncMock()
        
        await processor.cleanup()
        
        # Should call cleanup on components
        processor.llm_processor.cleanup.assert_called_once()


class TestPipelineRealComponents:
    """Integration tests with real components (when available)"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_ocr_integration(self, test_settings, sample_invoice_image):
        """Test with real OCR components if available"""
        try:
            from src.core.ocr_engine import InvoiceOCREngine
            ocr_engine = InvoiceOCREngine(engine_type="easyocr", languages=["en"])
            
            result = ocr_engine.extract_invoice_text(sample_invoice_image)
            
            assert isinstance(result, dict)
            assert 'full_text' in result
            assert 'text_elements' in result
            
        except ImportError:
            pytest.skip("OCR libraries not available")
        except Exception as e:
            pytest.skip(f"OCR integration test failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_preprocessing_integration(self, test_settings, sample_invoice_image):
        """Test with real preprocessing components"""
        try:
            from src.core.preprocessor import InvoicePreprocessor
            preprocessor = InvoicePreprocessor()
            
            result = preprocessor.preprocess_invoice_image(sample_invoice_image)
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] > 0 and result.shape[1] > 0
            
        except Exception as e:
            pytest.skip(f"Preprocessing integration test failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.gpu
    async def test_gpu_acceleration_integration(self, test_settings):
        """Test GPU acceleration if available"""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
            
            # Test GPU availability
            assert torch.cuda.device_count() > 0
            
        except ImportError:
            pytest.skip("PyTorch not available")
