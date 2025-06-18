"""
OCR Engine with multiple backend support
"""
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Dict, Optional
import logging

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class OCRResult:
    """OCR result container"""

    def __init__(self, text: str, confidence: float, bbox: List[List[int]]):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox

    def __repr__(self):
        return f"OCRResult(text='{self.text}', confidence={self.confidence:.2f})"


class MultilingualOCREngine:
    """OCR engine with multiple backend support"""

    def __init__(self, languages: List[str] = ["en", "de", "et"], engine: str = "easyocr"):
        """
        Initialize OCR engine

        Args:
            languages: List of language codes
            engine: OCR engine to use ('easyocr' or 'paddleocr')
        """
        self.languages = languages
        self.engine_type = engine
        self.engine = None

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the OCR engine"""
        if self.engine_type == "easyocr":
            self._initialize_easyocr()
        elif self.engine_type == "paddleocr":
            self._initialize_paddleocr()
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine_type}")

    def _initialize_easyocr(self):
        """Initialize EasyOCR"""
        try:
            self.engine = easyocr.Reader(
                self.languages,
                gpu=True,  # Will fallback to CPU if GPU not available
                verbose=False
            )
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def _initialize_paddleocr(self):
        """Initialize PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not available. Install with: pip install paddleocr")

        try:
            # PaddleOCR uses different language codes
            lang_map = {"en": "en", "de": "german", "et": "en"}  # Estonian falls back to English

            self.engines = {}
            for lang in self.languages:
                paddle_lang = lang_map.get(lang, "en")
                self.engines[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    show_log=False
                )

            logger.info(f"PaddleOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def extract_text(self, image: np.ndarray, detect_language: bool = True) -> Tuple[List[OCRResult], str]:
        """
        Extract text from image

        Args:
            image: Input image as numpy array
            detect_language: Whether to detect language automatically

        Returns:
            Tuple of (OCR results, detected language)
        """
        if self.engine_type == "easyocr":
            return self._extract_with_easyocr(image, detect_language)
        elif self.engine_type == "paddleocr":
            return self._extract_with_paddleocr(image, detect_language)

    def _extract_with_easyocr(self, image: np.ndarray, detect_language: bool) -> Tuple[List[OCRResult], str]:
        """Extract text using EasyOCR"""
        try:
            # Run OCR
            results = self.engine.readtext(image, detail=1, paragraph=False)

            # Convert to OCRResult objects
            ocr_results = []
            for bbox, text, confidence in results:
                if confidence > 0.3:  # Filter low confidence results
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox
                    ))

            # Detect language if requested
            detected_lang = "en"
            if detect_language and ocr_results:
                detected_lang = self._detect_language([r.text for r in ocr_results])

            return ocr_results, detected_lang

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return [], "en"

    def _extract_with_paddleocr(self, image: np.ndarray, detect_language: bool) -> Tuple[List[OCRResult], str]:
        """Extract text using PaddleOCR"""
        try:
            # Detect language first
            detected_lang = "en"
            if detect_language:
                # Use English engine for language detection
                temp_results = self.engines["en"].ocr(image, cls=True)
                if temp_results and temp_results[0]:
                    texts = [item[1][0] for item in temp_results[0] if item[1][1] > 0.3]
                    detected_lang = self._detect_language(texts)

            # Use appropriate engine
            engine_lang = detected_lang if detected_lang in self.engines else "en"
            results = self.engines[engine_lang].ocr(image, cls=True)

            # Convert to OCRResult objects
            ocr_results = []
            if results and results[0]:
                for item in results[0]:
                    bbox, (text, confidence) = item
                    if confidence > 0.3:
                        # Convert bbox format
                        bbox_points = [[int(x), int(y)] for x, y in bbox]
                        ocr_results.append(OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=bbox_points
                        ))

            return ocr_results, detected_lang

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return [], "en"

    def _detect_language(self, texts: List[str]) -> str:
        """Simple language detection based on keywords"""
        combined_text = " ".join(texts).lower()

        # Language-specific keywords
        language_keywords = {
            "de": ["rechnung", "datum", "betrag", "mwst", "ust", "gesamt", "steuernummer", "euro"],
            "et": ["arve", "kuupäev", "summa", "käibemaks", "kokku", "euro"],
            "en": ["invoice", "date", "amount", "vat", "total", "tax", "number"]
        }

        # Count keyword matches
        scores = {}
        for lang, keywords in language_keywords.items():
            if lang in self.languages:
                score = sum(1 for keyword in keywords if keyword in combined_text)
                scores[lang] = score

        # Return language with highest score
        if scores:
            detected = max(scores, key=scores.get)
            if scores[detected] > 0:
                return detected

        return "en"  # Default fallback

    def get_full_text(self, ocr_results: List[OCRResult]) -> str:
        """Combine OCR results into full text"""
        return " ".join([result.text for result in ocr_results])

    def filter_by_confidence(self, ocr_results: List[OCRResult], min_confidence: float = 0.5) -> List[OCRResult]:
        """Filter OCR results by confidence threshold"""
        return [result for result in ocr_results if result.confidence >= min_confidence]