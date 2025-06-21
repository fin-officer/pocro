"""
Pytest configuration and fixtures for European Invoice OCR tests
"""

import asyncio
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config.settings import AppSettings
from src.core.ocr_engine import InvoiceOCREngine
from src.core.pipeline import EuropeanInvoiceProcessor
from src.core.preprocessor import InvoicePreprocessor
from src.core.table_extractor import InvoiceTableExtractor

# Import application modules
from src.main import app
from src.models.invoice_schema import InvoiceData


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests (session-scoped)"""
    temp_path = Path(tempfile.mkdtemp(prefix="pocro_test_"))
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def test_settings(temp_dir: Path, monkeypatch_session) -> Settings:
    """Test settings with temporary directories"""
    # Set environment variables for settings that require special handling
    monkeypatch_session.setenv("OCR_LANGUAGES", "en,de,et")

    # Import AppSettings here after setting the environment variables
    from src.config.settings import AppSettings

    # Create settings instance with test-specific values
    settings = AppSettings(
        environment="test",
        debug=True,
        model_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "temp"),
        output_dir=str(temp_dir / "output"),
        cuda_visible_devices="",  # No GPU for tests
        enable_metrics=False,
        log_level="DEBUG",
        ocr_engine="easyocr",
        max_file_size=10 * 1024 * 1024,  # 10MB for tests
        max_batch_size=5,
    )

    # Verify ocr_languages was parsed correctly
    assert settings.ocr_languages == ["en", "de", "et"], f"Failed to parse OCR_LANGUAGES. Got: {settings.ocr_languages}"

    return settings


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
async def mock_processor(test_settings: Settings) -> EuropeanInvoiceProcessor:
    """Mock invoice processor for testing"""
    processor = EuropeanInvoiceProcessor(test_settings)

    # Mock initialization to avoid loading real models
    processor.preprocessor = InvoicePreprocessor()
    processor.ocr_engine = None  # Will be mocked in individual tests
    processor.table_extractor = None  # Will be mocked in individual tests
    processor.llm_processor = None  # Will be mocked in individual tests

    yield processor

    # Cleanup
    if processor:
        await processor.cleanup()


@pytest.fixture
def sample_invoice_image() -> np.ndarray:
    """Create a sample invoice image for testing"""
    # Create a simple test image with invoice-like content
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background

    # Add some black rectangles to simulate text areas
    image[50:100, 50:550] = 0  # Header area
    image[150:200, 50:300] = 0  # Address area
    image[250:500, 50:550] = 0  # Table area
    image[550:600, 400:550] = 0  # Total area

    return image


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create sample PDF bytes for testing"""
    # Create a simple PDF-like content (not a real PDF)
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing"""
    # Create a simple PNG image
    image = Image.new("RGB", (800, 600), color="white")

    # Add some basic content
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)

    try:
        # Try to use default font
        font = ImageFont.load_default()
    except:
        font = None

    draw.text((50, 50), "INVOICE #001", fill="black", font=font)
    draw.text((50, 100), "Date: 2024-01-15", fill="black", font=font)
    draw.text((50, 150), "Amount: €100.00", fill="black", font=font)

    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_invoice_text() -> str:
    """Sample invoice text for testing"""
    return """
    INVOICE #INV-2024-001
    
    Date: 15.01.2024
    
    From:
    Test Company GmbH
    Test Street 123
    12345 Test City
    VAT ID: DE123456789
    
    To:
    Customer Ltd
    Customer Road 456
    54321 Customer City
    VAT ID: EE987654321
    
    Description         Qty    Price    Total
    Software License    1      100.00   100.00
    Support Services    1       50.00    50.00
    
    Subtotal:                           150.00
    VAT (19%):                           28.50
    Total:                              178.50
    """


@pytest.fixture
def sample_extracted_data() -> Dict[str, Any]:
    """Sample extracted invoice data"""
    return {
        "invoice_id": "INV-2024-001",
        "issue_date": "2024-01-15",
        "currency_code": "EUR",
        "supplier": {
            "name": "Test Company GmbH",
            "vat_id": "DE123456789",
            "country_code": "DE",
            "address_line": "Test Street 123",
            "city": "Test City",
            "postal_code": "12345",
        },
        "customer": {
            "name": "Customer Ltd",
            "vat_id": "EE987654321",
            "country_code": "EE",
            "address_line": "Customer Road 456",
            "city": "Customer City",
            "postal_code": "54321",
        },
        "invoice_lines": [
            {
                "line_id": "1",
                "description": "Software License",
                "quantity": 1.0,
                "unit_price": 100.00,
                "line_total": 100.00,
                "vat_rate": 0.19,
            },
            {
                "line_id": "2",
                "description": "Support Services",
                "quantity": 1.0,
                "unit_price": 50.00,
                "line_total": 50.00,
                "vat_rate": 0.19,
            },
        ],
        "total_excl_vat": 150.00,
        "total_vat": 28.50,
        "total_incl_vat": 178.50,
        "payable_amount": 178.50,
    }


@pytest.fixture
def sample_ocr_results() -> list:
    """Sample OCR results for testing"""
    return [
        {"text": "INVOICE #INV-2024-001", "confidence": 0.95, "bbox": [[50, 50], [300, 50], [300, 80], [50, 80]]},
        {"text": "Date: 15.01.2024", "confidence": 0.92, "bbox": [[50, 100], [200, 100], [200, 120], [50, 120]]},
        {"text": "Test Company GmbH", "confidence": 0.88, "bbox": [[50, 150], [250, 150], [250, 170], [50, 170]]},
        {"text": "VAT ID: DE123456789", "confidence": 0.90, "bbox": [[50, 200], [220, 200], [220, 220], [50, 220]]},
        {"text": "Total: €178.50", "confidence": 0.93, "bbox": [[400, 500], [550, 500], [550, 520], [400, 520]]},
    ]


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for testing"""
    return """
    {
        "invoice_id": "INV-2024-001",
        "issue_date": "2024-01-15",
        "currency_code": "EUR",
        "supplier": {
            "name": "Test Company GmbH",
            "vat_id": "DE123456789",
            "country_code": "DE"
        },
        "customer": {
            "name": "Customer Ltd",
            "vat_id": "EE987654321",
            "country_code": "EE"
        },
        "total_excl_vat": 150.00,
        "total_vat": 28.50,
        "total_incl_vat": 178.50
    }
    """


@pytest.fixture
def invalid_invoice_data() -> Dict[str, Any]:
    """Invalid invoice data for testing validation"""
    return {
        "invoice_id": "",  # Invalid: empty
        "issue_date": "invalid-date",  # Invalid: bad format
        "currency_code": "INVALID",  # Invalid: not supported
        "supplier": {
            "name": "",  # Invalid: empty
            "vat_id": "INVALID123",  # Invalid: bad format
            "country_code": "XX",  # Invalid: not supported
        },
        "total_excl_vat": -100,  # Invalid: negative
        "total_vat": "not_a_number",  # Invalid: not numeric
        "total_incl_vat": 50,  # Invalid: doesn't match calculation
    }


# Test data files
@pytest.fixture
def test_files_dir() -> Path:
    """Directory containing test files"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_file(test_files_dir: Path, temp_dir: Path) -> Path:
    """Path to sample PDF file"""
    pdf_path = temp_dir / "sample_invoice.pdf"

    # Create a simple PDF file for testing
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Sample Invoice) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000203 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
295
%%EOF"""

    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    return pdf_path


@pytest.fixture
def sample_image_file(temp_dir: Path, sample_image_bytes: bytes) -> Path:
    """Path to sample image file"""
    image_path = temp_dir / "sample_invoice.png"

    with open(image_path, "wb") as f:
        f.write(sample_image_bytes)

    return image_path


# Mock configurations
@pytest.fixture
def mock_easyocr():
    """Mock EasyOCR for testing"""

    class MockEasyOCR:
        def __init__(self, languages):
            self.languages = languages

        def readtext(self, image, detail=1, paragraph=False):
            return [
                ([[50, 50], [300, 50], [300, 80], [50, 80]], "INVOICE #INV-2024-001", 0.95),
                ([[50, 100], [200, 100], [200, 120], [50, 120]], "Date: 15.01.2024", 0.92),
                ([[400, 500], [550, 500], [550, 520], [400, 520]], "Total: €178.50", 0.93),
            ]

    return MockEasyOCR


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""

    class MockLLM:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompts, sampling_params):
            class MockOutput:
                def __init__(self):
                    self.outputs = [MockOutputText()]

            class MockOutputText:
                def __init__(self):
                    self.text = """
                    {
                        "invoice_id": "INV-2024-001",
                        "issue_date": "2024-01-15",
                        "currency_code": "EUR",
                        "supplier": {
                            "name": "Test Company GmbH",
                            "vat_id": "DE123456789",
                            "country_code": "DE"
                        },
                        "customer": {
                            "name": "Customer Ltd",
                            "vat_id": "EE987654321",
                            "country_code": "EE"
                        },
                        "total_excl_vat": 150.00,
                        "total_vat": 28.50,
                        "total_incl_vat": 178.50
                    }
                    """

            return [MockOutput()]

    return MockLLM


# Async test helpers
@pytest.fixture
def async_test():
    """Helper for async tests"""

    def _async_test(coro):
        def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro(*args, **kwargs))

        return wrapper

    return _async_test


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests"""
    return {
        "max_processing_time": 30.0,  # seconds
        "max_memory_increase": 500,  # MB
        "min_success_rate": 0.95,  # 95%
        "test_file_count": 10,
    }


# Database fixtures (if needed for future extensions)
@pytest.fixture
def test_database_url():
    """Test database URL"""
    return "sqlite:///test.db"


# API testing fixtures
@pytest.fixture
def api_headers():
    """Standard API headers for testing"""
    return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "pytest-client"}


@pytest.fixture
def multipart_headers():
    """Headers for multipart form data"""
    return {"Accept": "application/json", "User-Agent": "pytest-client"}


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield

    # Clean up any temporary files, clear caches, etc.
    import gc

    gc.collect()


# Configuration for different test environments
@pytest.fixture(params=["cpu", "gpu"])
def test_environment(request):
    """Test environment configuration"""
    if request.param == "gpu":
        pytest.skip("GPU tests disabled in CI")

    return {"device": request.param, "use_quantization": request.param == "gpu"}


# Parametrized fixtures for different languages
@pytest.fixture(params=["en", "de", "et"])
def test_language(request):
    """Test different languages"""
    return request.param


@pytest.fixture(params=["easyocr", "paddleocr"])
def ocr_engine(request):
    """Test different OCR engines"""
    return request.param


# Utility functions for tests
@pytest.fixture
def create_test_upload_file(monkeypatch):
    """Create a test upload file"""
    from io import BytesIO
    from unittest.mock import MagicMock

    from fastapi import UploadFile

    def _create_test_upload_file(
        filename: str = "test_file.pdf", content: bytes = b"test content", content_type: str = "application/pdf"
    ) -> UploadFile:
        # Create a simple mock that behaves like UploadFile
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.file = BytesIO(content)
        mock_file.content_type = content_type

        # Mock the required methods
        async def async_read(size: int = -1) -> bytes:
            return content[:size] if size > 0 else content

        mock_file.read = async_read
        mock_file.__aenter__.return_value = mock_file
        mock_file.__aexit__.return_value = None

        return mock_file

    return _create_test_upload_file


def assert_valid_invoice_data(data: Dict[str, Any]):
    """Assert that invoice data is valid"""
    required_fields = ["invoice_id", "issue_date", "currency_code", "supplier", "customer"]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Validate supplier
    assert "name" in data["supplier"], "Supplier name is required"
    assert "country_code" in data["supplier"], "Supplier country code is required"

    # Validate financial data
    if "total_incl_vat" in data and "total_excl_vat" in data and "total_vat" in data:
        expected_total = data["total_excl_vat"] + data["total_vat"]
        actual_total = data["total_incl_vat"]
        assert abs(expected_total - actual_total) < 0.01, "Total calculation mismatch"


@pytest.fixture(scope="session", autouse=True)
def download_ocr_models():
    """Download OCR models before running tests"""
    try:
        # Download EasyOCR models
        import easyocr

        reader = easyocr.Reader(["en", "de", "et"])
        del reader
    except Exception as e:
        print(f"Warning: Could not download EasyOCR models: {e}")

    try:
        # Download PaddleOCR models
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        del ocr
    except Exception as e:
        print(f"Warning: Could not download PaddleOCR models: {e}")


# Mark slow tests
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: mark test as slow to run (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
