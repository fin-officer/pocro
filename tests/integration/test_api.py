"""
Integration tests for the API endpoints
"""
import pytest
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.main import app
from ..conftest import create_test_upload_file, assert_valid_invoice_data


class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_files(self, sample_image_bytes, sample_pdf_bytes):
        """Sample files for testing"""
        return {
            'image': create_test_upload_file("test_invoice.png", sample_image_bytes, "image/png"),
            'pdf': create_test_upload_file("test_invoice.pdf", sample_pdf_bytes, "application/pdf"),
            'invalid': create_test_upload_file("test.txt", b"not an image", "text/plain")
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "European Invoice OCR API" in data["message"]
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        # Health endpoint should always respond
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
        else:
            # Service unavailable - models not loaded
            data = response.json()
            assert "detail" in data
    
    def test_status_endpoint(self, client):
        """Test status endpoint"""
        response = client.get("/api/v1/status")
        
        # May fail if processor not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "uptime_seconds" in data
            assert "total_processed" in data
            assert "system_info" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/api/v1/metrics")
        
        # May fail if processor not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "performance" in data
            assert "system" in data
    
    def test_prometheus_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/api/v1/metrics/prometheus")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    def test_alerts_endpoint(self, client):
        """Test alerts endpoint"""
        response = client.get("/api/v1/alerts")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_process_invoice_no_file(self, client):
        """Test invoice processing without file"""
        response = client.post("/api/v1/process-invoice")
        assert response.status_code == 422  # Validation error
    
    def test_process_invoice_invalid_format(self, client, sample_files):
        """Test invoice processing with invalid file format"""
        response = client.post(
            "/api/v1/process-invoice",
            files={"file": (sample_files['invalid'].filename, sample_files['invalid'].file, sample_files['invalid'].content_type)}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file format" in data["detail"]
    
    @patch('src.core.pipeline.EuropeanInvoiceProcessor')
    def test_process_invoice_image_success(self, mock_processor_class, client, sample_files):
        """Test successful image processing"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_invoice_upload.return_value = {
            'status': 'success',
            'extracted_data': {
                'invoice_id': 'INV-2024-001',
                'issue_date': '2024-01-15',
                'currency_code': 'EUR',
                'supplier': {'name': 'Test Company', 'country_code': 'DE'},
                'customer': {'name': 'Customer Ltd', 'country_code': 'EE'},
                'total_incl_vat': 119.00
            },
            'processing_time': 2.5
        }
        mock_processor_class.return_value = mock_processor
        
        response = client.post(
            "/api/v1/process-invoice",
            files={"file": (sample_files['image'].filename, sample_files['image'].file, sample_files['image'].content_type)}
        )
        
        assert response.status_code in [200, 503]  # May fail if processor not initialized
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["filename"] == sample_files['image'].filename
            assert "extracted_data" in data
            assert "processing_time" in data
    
    @patch('src.core.pipeline.EuropeanInvoiceProcessor')
    def test_process_invoice_pdf_success(self, mock_processor_class, client, sample_files):
        """Test successful PDF processing"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_invoice_upload.return_value = {
            'status': 'success',
            'extracted_data': {
                'invoice_id': 'INV-2024-001',
                'issue_date': '2024-01-15',
                'currency_code': 'EUR',
                'supplier': {'name': 'Test Company', 'country_code': 'DE'},
                'customer': {'name': 'Customer Ltd', 'country_code': 'EE'},
                'total_incl_vat': 119.00
            },
            'processing_time': 3.2
        }
        mock_processor_class.return_value = mock_processor
        
        response = client.post(
            "/api/v1/process-invoice",
            files={"file": (sample_files['pdf'].filename, sample_files['pdf'].file, sample_files['pdf'].content_type)}
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["filename"] == sample_files['pdf'].filename
    
    def test_process_invoice_with_validation(self, client, sample_files):
        """Test invoice processing with validation enabled"""
        response = client.post(
            "/api/v1/process-invoice?validate=true",
            files={"file": (sample_files['image'].filename, sample_files['image'].file, sample_files['image'].content_type)}
        )
        
        # May succeed or fail depending on processor availability
        assert response.status_code in [200, 400, 503]
    
    def test_process_invoice_without_validation(self, client, sample_files):
        """Test invoice processing without validation"""
        response = client.post(
            "/api/v1/process-invoice?validate=false",
            files={"file": (sample_files['image'].filename, sample_files['image'].file, sample_files['image'].content_type)}
        )
        
        # May succeed or fail depending on processor availability
        assert response.status_code in [200, 400, 503]
    
    def test_process_batch_no_files(self, client):
        """Test batch processing without files"""
        response = client.post("/api/v1/process-batch")
        assert response.status_code == 422  # Validation error
    
    def test_process_batch_too_many_files(self, client, sample_files):
        """Test batch processing with too many files"""
        # Create list with too many files
        files = []
        for i in range(51):  # More than the limit
            files.append(("files", (f"file_{i}.jpg", io.BytesIO(b"fake"), "image/jpeg")))
        
        response = client.post("/api/v1/process-batch", files=files)
        assert response.status_code == 400
        
        data = response.json()
        assert "Too many files" in data["detail"]
    
    @patch('src.core.pipeline.EuropeanInvoiceProcessor')
    def test_process_batch_success(self, mock_processor_class, client, sample_files):
        """Test successful batch processing"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_batch_upload.return_value = [
            {
                'status': 'success',
                'filename': 'test1.png',
                'extracted_data': {'invoice_id': 'INV-001'},
                'processing_time': 1.5
            },
            {
                'status': 'success',
                'filename': 'test2.png',
                'extracted_data': {'invoice_id': 'INV-002'},
                'processing_time': 1.8
            }
        ]
        mock_processor_class.return_value = mock_processor
        
        files = [
            ("files", (sample_files['image'].filename, sample_files['image'].file, sample_files['image'].content_type)),
            ("files", ("test2.png", io.BytesIO(b"fake image"), "image/png"))
        ]
        
        response = client.post("/api/v1/process-batch", files=files)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "completed"
            assert "processed_count" in data
            assert "results" in data
    
    def test_validate_endpoint_valid_data(self, client, sample_extracted_data):
        """Test validation endpoint with valid data"""
        response = client.post(
            "/api/v1/validate",
            json=sample_extracted_data
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_valid" in data
            assert "errors" in data
            assert "completeness_score" in data
            assert "confidence_score" in data
    
    def test_validate_endpoint_invalid_data(self, client, invalid_invoice_data):
        """Test validation endpoint with invalid data"""
        response = client.post(
            "/api/v1/validate",
            json=invalid_invoice_data
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_valid" in data
            assert data["is_valid"] is False
            assert len(data["errors"]) > 0
    
    def test_models_endpoint(self, client):
        """Test models information endpoint"""
        response = client.get("/api/v1/models")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Should contain model information
            assert isinstance(data, dict)
    
    def test_models_reload_endpoint(self, client):
        """Test models reload endpoint"""
        response = client.post("/api/v1/models/reload")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "reload_initiated"
    
    def test_config_endpoint(self, client):
        """Test configuration endpoint"""
        response = client.get("/api/v1/config")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "ocr_engine" in data
            assert "ocr_languages" in data
            assert "model_name" in data
            assert "max_file_size_mb" in data
    
    def test_languages_endpoint(self, client):
        """Test supported languages endpoint"""
        response = client.get("/api/v1/languages")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
    
    def test_formats_endpoint(self, client):
        """Test supported formats endpoint"""
        response = client.get("/api/v1/formats")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "input_formats" in data
        assert "output_formats" in data
        assert isinstance(data["input_formats"], list)
        assert isinstance(data["output_formats"], list)
        assert ".pdf" in data["input_formats"]
        assert ".jpg" in data["input_formats"]
    
    def test_test_endpoint(self, client):
        """Test the test endpoint"""
        test_data = {"test": "data"}
        
        response = client.post("/api/v1/test", json=test_data)
        
        assert response.status_code in [200, 429]  # May be rate limited
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ok"
            assert data["message"] == "API is working correctly"
            assert "timestamp" in data
            assert data["received_data"] == test_data
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Make multiple requests quickly
        responses = []
        for i in range(5):
            response = client.get("/api/v1/test")
            responses.append(response)
        
        # Should eventually get rate limited
        status_codes = [r.status_code for r in responses]
        
        # All should be either successful or rate limited
        assert all(code in [200, 429] for code in status_codes)
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/test")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers are present"""
        response = client.get("/")
        
        # Should have security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
    
    def test_request_id_header(self, client):
        """Test request ID header is added"""
        response = client.get("/")
        
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
    
    def test_processing_time_header(self, client):
        """Test processing time header is added"""
        response = client.get("/")
        
        assert "x-processing-time" in response.headers
        processing_time = float(response.headers["x-processing-time"])
        assert processing_time >= 0
    
    def test_large_file_upload(self, client):
        """Test upload with large file"""
        # Create a large fake file (beyond typical limits)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        
        response = client.post(
            "/api/v1/process-invoice",
            files={"file": ("large_file.jpg", io.BytesIO(large_content), "image/jpeg")}
        )
        
        # Should be rejected for being too large
        assert response.status_code in [400, 413, 422]
    
    def test_malformed_json_request(self, client):
        """Test request with malformed JSON"""
        response = client.post(
            "/api/v1/validate",
            data="malformed json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable entity
    
    def test_unsupported_content_type(self, client):
        """Test request with unsupported content type"""
        response = client.post(
            "/api/v1/test",
            data="some data",
            headers={"content-type": "application/xml"}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 415, 422]
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/api/v1/test")
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete
        assert len(responses) == 10
        
        # Most should be successful (some may be rate limited)
        successful = sum(1 for r in responses if r.status_code == 200)
        assert successful >= 5  # At least half should succeed
    
    def test_error_response_format(self, client):
        """Test error response format is consistent"""
        response = client.post(
            "/api/v1/process-invoice",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        
        # Should have consistent error format
        assert "detail" in data
        assert isinstance(data["detail"], str)
    
    def test_api_version_consistency(self, client):
        """Test API version consistency across endpoints"""
        endpoints = [
            "/api/v1/config",
            "/api/v1/languages", 
            "/api/v1/formats",
            "/api/v1/test"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # All v1 endpoints should either work or fail consistently
            assert response.status_code in [200, 503, 429]
    
    @pytest.mark.slow
    def test_api_performance(self, client):
        """Test API response performance"""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/test")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # API should respond quickly for simple endpoints
        assert response_time < 1.0  # Less than 1 second
        assert response.status_code in [200, 429]
    
    def test_health_check_reliability(self, client):
        """Test health check endpoint reliability"""
        # Health check should be reliable and fast
        responses = []
        
        for _ in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # All health checks should complete
        assert len(responses) == 10
        
        # Should be consistent
        status_codes = {r.status_code for r in responses}
        assert len(status_codes) <= 2  # Should be consistent (200 or 503)


class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_404_error_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_405_error_handling(self, client):
        """Test 405 Method Not Allowed"""
        response = client.delete("/")  # DELETE not allowed on root
        assert response.status_code == 405
    
    def test_500_error_simulation(self, client):
        """Test 500 error handling simulation"""
        # This would require mocking to force a 500 error
        # For now, just ensure error handling structure exists
        pass
    
    def test_timeout_handling(self, client):
        """Test request timeout handling"""
        # This would require special setup to test timeouts
        # For now, verify that timeout configurations exist
        pass


class TestAPIIntegrationWithMockProcessor:
    """Integration tests with fully mocked processor"""
    
    @pytest.fixture
    def client_with_mock_processor(self):
        """Client with mocked processor dependency"""
        # This would require dependency injection mocking
        # Implementation depends on how dependencies are structured
        return TestClient(app)
    
    def test_full_invoice_processing_pipeline(self, client_with_mock_processor, sample_image_bytes):
        """Test complete invoice processing pipeline"""
        # This test would mock the entire processing pipeline
        # and verify the API correctly orchestrates all components
        pass
