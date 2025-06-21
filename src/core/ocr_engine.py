"""
OCR Engine for invoice text extraction
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """Abstract base class for OCR engines"""

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image"""
        pass

    @abstractmethod
    def extract_text_with_boxes(self, image: np.ndarray) -> List[Dict]:
        """Extract text with bounding boxes"""
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR implementation"""

    def __init__(self, languages: List[str] = None):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR is not installed")

        self.languages = languages or ["en", "de", "et"]
        self.reader = easyocr.Reader(self.languages, gpu=True)
        logger.info(f"EasyOCR initialized with languages: {self.languages}")

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image using EasyOCR"""
        try:
            results = self.reader.readtext(image, detail=1, paragraph=False)

            extracted_text = []
            for bbox, text, confidence in results:
                if confidence > 0.3:  # Filter low confidence detections
                    extracted_text.append({"text": text.strip(), "confidence": confidence, "bbox": bbox})

            logger.debug(f"EasyOCR extracted {len(extracted_text)} text elements")
            return extracted_text

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []

    def extract_text_with_boxes(self, image: np.ndarray) -> List[Dict]:
        """Extract text with detailed bounding box information"""
        return self.extract_text(image)


class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation - better for tables"""

    def __init__(self, languages: List[str] = None):
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed")

        # PaddleOCR language mapping
        lang_map = {"en": "en", "de": "german", "et": "es"}  # Estonian not directly supported, use Spanish as fallback

        self.languages = languages or ["en"]
        self.primary_lang = lang_map.get(self.languages[0], "en")

        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.primary_lang, use_gpu=True, show_log=False)

        logger.info(f"PaddleOCR initialized with language: {self.primary_lang}")

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image using PaddleOCR"""
        try:
            results = self.ocr.ocr(image, cls=True)

            extracted_text = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    if confidence > 0.3:
                        extracted_text.append({"text": text.strip(), "confidence": confidence, "bbox": bbox})

            logger.debug(f"PaddleOCR extracted {len(extracted_text)} text elements")
            return extracted_text

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return []

    def extract_text_with_boxes(self, image: np.ndarray) -> List[Dict]:
        """Extract text with detailed bounding box information"""
        return self.extract_text(image)


class InvoiceOCREngine:
    """Main OCR engine for invoice processing"""

    def __init__(self, engine_type: str = "easyocr", languages: List[str] = None):
        self.engine_type = engine_type.lower()
        self.languages = languages or ["en", "de", "et"]

        # Initialize OCR engine
        if self.engine_type == "easyocr":
            self.engine = EasyOCREngine(self.languages)
        elif self.engine_type == "paddleocr":
            self.engine = PaddleOCREngine(self.languages)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine_type}")

        logger.info(f"Invoice OCR engine initialized: {engine_type}")

    def extract_invoice_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from invoice image

        Args:
            image: Preprocessed invoice image

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Extract text with bounding boxes
            text_elements = self.engine.extract_text_with_boxes(image)

            # Sort by reading order (top to bottom, left to right)
            sorted_elements = self._sort_reading_order(text_elements)

            # Combine into full text
            full_text = self._combine_text_elements(sorted_elements)

            # Extract structured data
            structured_data = self._structure_text_data(sorted_elements)

            return {
                "full_text": full_text,
                "text_elements": sorted_elements,
                "structured_data": structured_data,
                "total_elements": len(text_elements),
                "avg_confidence": np.mean([elem["confidence"] for elem in text_elements]) if text_elements else 0,
            }

        except Exception as e:
            logger.error(f"Invoice text extraction failed: {e}")
            return {
                "full_text": "",
                "text_elements": [],
                "structured_data": {},
                "total_elements": 0,
                "avg_confidence": 0,
            }

    def _sort_reading_order(self, text_elements: List[Dict]) -> List[Dict]:
        """Sort text elements in reading order"""
        # Sort by Y coordinate first (top to bottom), then X coordinate (left to right)
        return sorted(text_elements, key=lambda x: (self._get_y_center(x["bbox"]), self._get_x_center(x["bbox"])))

    def _get_y_center(self, bbox) -> float:
        """Get Y center of bounding box"""
        if isinstance(bbox[0], list):
            # EasyOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            return sum(point[1] for point in bbox) / 4
        else:
            # PaddleOCR format: [x1, y1, x2, y2]
            return (bbox[1] + bbox[3]) / 2

    def _get_x_center(self, bbox) -> float:
        """Get X center of bounding box"""
        if isinstance(bbox[0], list):
            # EasyOCR format
            return sum(point[0] for point in bbox) / 4
        else:
            # PaddleOCR format
            return (bbox[0] + bbox[2]) / 2

    def _combine_text_elements(self, text_elements: List[Dict]) -> str:
        """Combine text elements into full text with proper spacing"""
        if not text_elements:
            return ""

        full_text = ""
        last_y = None

        for element in text_elements:
            text = element["text"]
            y_pos = self._get_y_center(element["bbox"])

            # Add newline if we're on a new line (significant Y difference)
            if last_y is not None and abs(y_pos - last_y) > 10:
                full_text += "\n"
            elif full_text and not full_text.endswith(" "):
                full_text += " "

            full_text += text
            last_y = y_pos

        return full_text.strip()

    def _structure_text_data(self, text_elements: List[Dict]) -> Dict:
        """Extract structured data from text elements"""
        structured = {"lines": [], "potential_tables": [], "headers": [], "amounts": []}

        current_line = []
        last_y = None
        line_tolerance = 15  # pixels

        for element in text_elements:
            y_pos = self._get_y_center(element["bbox"])

            # Group elements into lines
            if last_y is None or abs(y_pos - last_y) <= line_tolerance:
                current_line.append(element)
            else:
                if current_line:
                    structured["lines"].append(current_line)
                current_line = [element]

            last_y = y_pos

            # Identify potential amounts
            text = element["text"]
            if self._is_amount(text):
                structured["amounts"].append(element)

            # Identify potential headers (high confidence, short text)
            if element["confidence"] > 0.9 and len(text.split()) <= 3:
                structured["headers"].append(element)

        # Add last line
        if current_line:
            structured["lines"].append(current_line)

        return structured

    def _is_amount(self, text: str) -> bool:
        """Check if text represents a monetary amount"""
        import re

        # Patterns for amounts in different formats
        amount_patterns = [
            r"[€$£]\s*\d+[.,]\d{2}",  # €123.45, $123,45
            r"\d+[.,]\d{2}\s*[€$£]",  # 123.45€, 123,45$
            r"\d{1,3}(?:[.,]\d{3})*[.,]\d{2}",  # 1,234.56 or 1.234,56
        ]

        for pattern in amount_patterns:
            if re.search(pattern, text):
                return True

        return False

    def detect_language(self, text_elements: List[Dict]) -> str:
        """Detect primary language from extracted text"""
        if not text_elements:
            return "en"

        # Combine all text
        all_text = " ".join([elem["text"] for elem in text_elements]).lower()

        # Language keywords
        language_keywords = {
            "de": ["rechnung", "datum", "betrag", "mwst", "ust", "gesamt", "summe", "netto", "brutto"],
            "et": ["arve", "kuupäev", "summa", "käibemaks", "kokku", "maksukohustuslane"],
            "en": ["invoice", "date", "amount", "vat", "total", "subtotal", "tax"],
        }

        # Count matches for each language
        scores = {lang: 0 for lang in language_keywords}
        for lang, keywords in language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            scores[lang] = score

        # Get the language with the highest score
        max_score = max(scores.values())

        # If no keywords were found, default to English
        if max_score == 0:
            logger.info("No language keywords found, defaulting to English")
            return "en"

        # Return the language with the highest score
        detected_lang = max(scores, key=scores.get)
        logger.info(f"Detected language: {detected_lang} (scores: {scores})")

        return detected_lang
