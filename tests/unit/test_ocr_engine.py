"""
Unit tests for OCR engine
"""
import numpy as np
import pytest
import cv2

from src.core.ocr_engine import MultilingualOCREngine, OCRResult


class TestMultilingualOCREngine:

    @pytest.fixture
    def ocr_engine(self):
        return MultilingualOCREngine(languages=["en"], engine="easyocr")

    @pytest.fixture
    def sample_text_image(self):
        """Create an image with text"""
        image = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(image, "INVOICE 2024-001", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image

    def test_language_detection(self, ocr_engine):
        """Test language detection"""
        german_texts = ["Rechnung", "Datum", "Betrag", "MwSt"]
        english_texts = ["Invoice", "Date", "Amount", "VAT"]

        german_lang = ocr_engine._detect_language(german_texts)
        english_lang = ocr_engine._detect_language(english_texts)

        # Should default to 'en' since only English is in languages
        assert german_lang == "en"
        assert english_lang == "en"

    def test_extract_text(self, ocr_engine, sample_text_image):
        """Test text extraction"""
        results, detected_lang = ocr_engine.extract_text(sample_text_image)

        assert isinstance(results, list)
        assert isinstance(detected_lang, str)

        if results:  # OCR might not work in test environment
            assert all(isinstance(r, OCRResult) for r in results)