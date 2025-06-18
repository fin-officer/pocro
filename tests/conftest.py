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
def sample_invoice_text() -> str:
    """Sample invoice text for testing"""
    return """
    RECHNUNG
    
    Musterfirma GmbH
    Musterstraße 1
    10115 Berlin
    USt-ID: DE123456789
    
    Rechnungsnummer: RE-2024-001
    Datum: 15.01.2024
    
    Kunde AG
    Kundenstraße 10
    20095 Hamburg
    
    Pos. Beschreibung           Menge  Preis    Gesamt
    1    Beratungsleistung        10    100,00   1.000,00
    2    Software-Lizenz           1    500,00     500,00
    
    Netto:                                     1.500,00 EUR
    MwSt. 19%:                                   285,00 EUR
    Gesamt:                                    1.785,00 EUR
    """


@pytest.fixture
def expected_german_output() -> dict:
    """Expected output for German invoice"""
    return {
        "invoice_id": "RE-2024-001",
        "issue_date": "2024-01-15",
        "currency_code": "EUR",
        "supplier": {
            "name": "Musterfirma GmbH",
            "vat_id": "DE123456789"
        },
        "customer": {
            "name": "Kunde AG"
        },
        "total_excl_vat": 1500.00,
        "total_vat": 285.00,
        "total_incl_vat": 1785.00
    }