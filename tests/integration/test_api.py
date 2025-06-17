"""
Integration tests for the API endpoints
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class TestAPIIntegration:

    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "European Invoice OCR API" in response.json()["message"]

    def test_health_endpoint(self, client: TestClient):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May fail if models not loaded

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "timestamp" in data

    def test_process_invoice_no_file(self, client: TestClient):
        """Test invoice processing without file"""
        response = client.post("/process-invoice")
        assert response.status_code == 422  # Validation error

    def test_process_invoice_invalid_format(self, client: TestClient):
        """Test invoice processing with invalid file format"""
        # Create a fake text file
        fake_file = io.BytesIO(b"not an image or pdf")

        response = client.post(
            "/process-invoice",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    @pytest.mark.skipif(
        not Path("tests/fixtures/sample_invoices/german_invoice.pdf").exists(),
        reason="Sample invoice file not found"
    )
    def test_process_invoice_pdf(self, client: TestClient):
        """Test invoice processing with PDF file"""
        pdf_path = Path("tests/fixtures/sample_invoices/german_invoice.pdf")

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/process-invoice",
                files={"file": ("german_invoice.pdf", f, "application/pdf")}
            )

        # May fail if models not loaded, but should not crash
        assert response.status_code in [200, 503]

    def test_process_invoice_image(self, client: TestClient):
        """Test invoice processing with image file"""
        # Create a simple test image
        image = Image.new('RGB', (800, 600), color='white')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)

        response = client.post(
            "/process-invoice",
            files={"file": ("test_invoice.jpg", img_buffer, "image/jpeg")}
        )

        # May fail if models not loaded, but should not crash
        assert response.status_code in [200, 503]

    def test_process_batch_too_many_files(self, client: TestClient):
        """Test batch processing with too many files"""
        # Create multiple fake files
        files = []
        for i in range(51):  # More than the limit
            fake_file = io.BytesIO(b"fake image data")
            files.append(("files", (f"file_{i}.jpg", fake_file, "image/jpeg")))

        response = client.post("/process-batch", files=files)
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]

    def test_model_status_endpoint(self, client: TestClient):
        """Test model status endpoint"""
        response = client.get("/models/status")
        # May fail if processor not initialized
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        # May fail if processor not initialized
        assert response.status_code in [200, 503]