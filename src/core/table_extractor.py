"""
Table extraction from invoices
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import re
import logging

try:
    from paddleocr import PPStructure
    PPSTRUCTURE_AVAILABLE = True
except ImportError:
    PPSTRUCTURE_AVAILABLE = False

from .ocr_engine import OCRResult

logger = logging.getLogger(__name__)


class InvoiceTable:
    """Represents an extracted table from invoice"""

    def __init__(self, rows: List[List[str]], bbox: List[int], confidence: float = 1.0):
        self.rows = rows
        self.bbox = bbox
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary"""
        return {
            "rows": self.rows,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "num_rows": len(self.rows),
            "num_cols": len(self.rows[0]) if self.rows else 0
        }


class InvoiceLineItem:
    """Represents a line item from invoice table"""

    def __init__(self, description: str = "", quantity: Optional[float] = None,
                 unit_price: Optional[float] = None, total_price: Optional[float] = None,
                 confidence: float = 1.0):
        self.description = description
        self.quantity = quantity
        self.unit_price = unit_price
        self.total_price = total_price
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert line item to dictionary"""
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "confidence": self.confidence
        }


class TableExtractor:
    """Extract tables and line items from invoices"""

    def __init__(self, use_ppstructure: bool = True):
        """
        Initialize table extractor

        Args:
            use_ppstructure: Whether to use PaddleOCR's PP-Structure for table detection
        """
        self.use_ppstructure = use_ppstructure and PPSTRUCTURE_AVAILABLE
        self.table_engine = None

        if self.use_ppstructure:
            self._initialize_ppstructure()

    def _initialize_ppstructure(self):
        """Initialize PP-Structure for table detection"""
        try:
            self.table_engine = PPStructure(
                table=True,
                ocr=True,
                show_log=False,
                lang="en"  # Will be overridden per image
            )
            logger.info("PP-Structure initialized for table detection")
        except Exception as e:
            logger.warning(f"Failed to initialize PP-Structure: {e}")
            self.use_ppstructure = False

    def extract_tables(self, image: np.ndarray) -> List[InvoiceTable]:
        """Extract tables from image"""
        if self.use_ppstructure:
            return self._extract_tables_ppstructure(image)
        else:
            return self._extract_tables_heuristic(image)

    def _extract_tables_ppstructure(self, image: np.ndarray) -> List[InvoiceTable]:
        """Extract tables using PP-Structure"""
        try:
            # Run PP-Structure
            result = self.table_engine(image)

            tables = []
            for region in result:
                if region['type'] == 'table' and 'res' in region:
                    # Parse HTML table if available
                    if 'html' in region['res']:
                        table_data = self._parse_html_table(region['res']['html'])
                        if table_data:
                            tables.append(InvoiceTable(
                                rows=table_data,
                                bbox=region['bbox'],
                                confidence=0.8  # Default confidence for PP-Structure
                            ))

            return tables

        except Exception as e:
            logger.error(f"PP-Structure table extraction failed: {e}")
            return self._extract_tables_heuristic(image)

    def _extract_tables_heuristic(self, image: np.ndarray) -> List[InvoiceTable]:
        """Extract tables using heuristic methods"""
        # This is a simplified table detection
        # In practice, you'd implement more sophisticated table detection

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

        # Find contours (potential table regions)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Extract table region (this would need OCR to get actual content)
                tables.append(InvoiceTable(
                    rows=[["Placeholder table data"]],  # Would need actual OCR
                    bbox=[x, y, x+w, y+h],
                    confidence=0.5
                ))

        return tables

    def _parse_html_table(self, html_content: str) -> List[List[str]]:
        """Parse HTML table content"""
        try:
            # Simple HTML table parsing (in production, use proper HTML parser)
            import re

            # Extract table rows
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'

            rows = []
            for row_match in re.finditer(row_pattern, html_content, re.DOTALL | re.IGNORECASE):
                row_content = row_match.group(1)
                cells = []
                for cell_match in re.finditer(cell_pattern, row_content, re.DOTALL | re.IGNORECASE):
                    cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)

            return rows

        except Exception as e:
            logger.error(f"Failed to parse HTML table: {e}")
            return []

    def extract_line_items_from_ocr(self, ocr_results: List[OCRResult], image_height: int) -> List[InvoiceLineItem]:
        """Extract line items from OCR results using position analysis"""

        # Sort OCR results by Y position (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x.bbox[0][1])

        # Group results by rows (similar Y positions)
        rows = self._group_by_rows(sorted_results, tolerance=10)

        # Parse each row as potential line item
        line_items = []
        for row in rows:
            line_item = self._parse_row_as_line_item(row)
            if line_item and line_item.description:  # Only add if we have at least description
                line_items.append(line_item)

        return line_items

    def _group_by_rows(self, ocr_results: List[OCRResult], tolerance: int = 10) -> List[List[OCRResult]]:
        """Group OCR results by rows based on Y position"""
        if not ocr_results:
            return []

        rows = []
        current_row = [ocr_results[0]]
        current_y = ocr_results[0].bbox[0][1]

        for result in ocr_results[1:]:
            result_y = result.bbox[0][1]

            if abs(result_y - current_y) <= tolerance:
                current_row.append(result)
            else:
                # Sort current row by X position (left to right)
                current_row.sort(key=lambda x: x.bbox[0][0])
                rows.append(current_row)
                current_row = [result]
                current_y = result_y

        # Add last row
        if current_row:
            current_row.sort(key=lambda x: x.bbox[0][0])
            rows.append(current_row)

        return rows

    def _parse_row_as_line_item(self, row: List[OCRResult]) -> Optional[InvoiceLineItem]:
        """Parse a row of OCR results as a line item"""
        if len(row) < 2:  # Need at least 2 elements for a meaningful line item
            return None

        description_parts = []
        quantity = None
        unit_price = None
        total_price = None
        confidence_sum = 0

        for result in row:
            text = result.text.strip()
            confidence_sum += result.confidence

            # Try to parse as number/price
            if self._is_price(text):
                price_value = self._extract_price(text)
                if price_value is not None:
                    if total_price is None:
                        total_price = price_value
                    elif unit_price is None:
                        unit_price = total_price  # Move previous total to unit price
                        total_price = price_value

            # Try to parse as quantity
            elif self._is_quantity(text):
                qty = self._extract_quantity(text)
                if qty is not None and quantity is None:
                    quantity = qty

            # Otherwise, treat as description
            else:
                description_parts.append(text)

        # Build description
        description = " ".join(description_parts).strip()

        # Calculate average confidence
        avg_confidence = confidence_sum / len(row) if row else 0

        return InvoiceLineItem(
            description=description,
            quantity=quantity,
            unit_price=unit_price,
            total_price=total_price,
            confidence=avg_confidence
        )

    def _is_price(self, text: str) -> bool:
        """Check if text looks like a price"""
        # Look for currency symbols or decimal patterns
        price_pattern = r'[\€\$\£]?\s*\d+[.,]\d{2}|\d+[.,]\d{2}\s*[\€\$\£]?'
        return bool(re.search(price_pattern, text))

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract numeric price value"""
        try:
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[\€\$\£\s]', '', text)
            # Replace comma with dot for decimal
            cleaned = cleaned.replace(',', '.')
            # Extract number
            number_match = re.search(r'\d+\.?\d*', cleaned)
            if number_match:
                return float(number_match.group())
        except ValueError:
            pass
        return None

    def _is_quantity(self, text: str) -> bool:
        """Check if text looks like a quantity"""
        # Simple quantity pattern - just numbers, possibly with units
        return bool(re.match(r'^\d+(\.\d+)?\s*(st|stk|pcs|pc|x)?\.?$', text.lower().strip()))

    def _extract_quantity(self, text: str) -> Optional[float]:
        """Extract numeric quantity value"""
        try:
            # Extract just the number part
            number_match = re.search(r'\d+(\.\d+)?', text)
            if number_match:
                return float(number_match.group())
        except ValueError:
            pass
        return None