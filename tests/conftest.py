"""
Pytest configuration and fixtures
"""
import asyncio
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.config.settings import Settings
from src.core.pipeline import EuropeanInvoiceProcessor


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with temporary directories"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Settings(
            environment="test",
            debug=True,
            model_cache_dir=f"{temp_dir}/models",
            temp_dir=f"{temp_dir}/temp",
            output_dir=f"{temp_dir}/output",
            cuda_visible_devices="",  # No GPU for tests
            enable_metrics=False,
            log_level="DEBUG"
        )


@pytest.fixture
async def processor(test_settings: Settings) -> EuropeanInvoiceProcessor:
    """Invoice processor for testing"""
    processor = EuropeanInvoiceProcessor(test_settings)
    await processor.initialize()
    yield processor
    await processor.cleanup()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def sample_invoice_pdf() -> Path:
    """Path to sample invoice PDF"""
    return Path(__file__).parent / "fixtures" / "sample_invoices" / "german_invoice.pdf"


@pytest.fixture
def sample_invoice_image() -> Path:
    """Path to sample invoice image"""
    return Path(__file__).parent / "fixtures" / "sample_invoices" / "english_invoice.jpg"


@pytest.fixture
def expected_german_output() -> dict:
    """Expected output for German invoice"""
    return {
        "invoice_id": "RE-2024-001",
        "issue_date": "2024-01-15",
        "currency_code": "EUR",
        "supplier": {
            "name": "Musterfirma GmbH",
            "vat_id": "DE123456789",
            "country_code": "DE"
        },
        "total_incl_vat": 119.00,
        "total_vat": 19.00
    }