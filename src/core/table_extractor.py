"""
Table extraction for invoice line items
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from paddleocr import PPStructure

    PPSTRUCTURE_AVAILABLE = True
except ImportError:
    PPSTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class InvoiceTableExtractor:
    """Extract and parse tables from invoice images"""

    def __init__(self, use_pp_structure: bool = True):
        self.use_pp_structure = use_pp_structure and PPSTRUCTURE_AVAILABLE

        if self.use_pp_structure:
            try:
                self.table_engine = PPStructure(table=True, ocr=True, show_log=False, lang="en")
                logger.info("PPStructure table engine initialized")
            except Exception as e:
                logger.warning(f"PPStructure initialization failed: {e}")
                self.use_pp_structure = False

        if not self.use_pp_structure:
            logger.info("Using fallback table detection method")

    def extract_tables(self, image: np.ndarray, ocr_results: List[Dict] = None) -> List[Dict]:
        """
        Extract tables from invoice image

        Args:
            image: Input image
            ocr_results: OCR results for fallback method

        Returns:
            List of extracted tables
        """
        try:
            if self.use_pp_structure:
                return self._extract_with_ppstructure(image)
            else:
                return self._extract_with_fallback(image, ocr_results or [])

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

    def _extract_with_ppstructure(self, image: np.ndarray) -> List[Dict]:
        """Extract tables using PP-Structure"""
        try:
            result = self.table_engine(image)

            tables = []
            for region in result:
                if region["type"] == "table":
                    table_data = self._parse_ppstructure_table(region)
                    if table_data:
                        tables.append(table_data)

            logger.info(f"PP-Structure extracted {len(tables)} tables")
            return tables

        except Exception as e:
            logger.error(f"PP-Structure table extraction failed: {e}")
            return []

    def _parse_ppstructure_table(self, table_region: Dict) -> Optional[Dict]:
        """Parse PP-Structure table result"""
        try:
            html = table_region["res"]["html"]
            bbox = table_region["bbox"]

            # Parse HTML table to extract data
            table_data = self._parse_html_table(html)

            return {
                "type": "ppstructure",
                "bbox": bbox,
                "html": html,
                "data": table_data,
                "rows": len(table_data),
                "columns": len(table_data[0]) if table_data else 0,
            }

        except Exception as e:
            logger.error(f"Failed to parse PP-Structure table: {e}")
            return None

    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Parse HTML table to extract cell data"""
        try:
            # Simple HTML table parsing
            # In production, you might want to use BeautifulSoup
            import re

            # Remove HTML tags and extract cell content
            cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.DOTALL)
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)

            table_data = []
            for row in rows:
                row_cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                row_data = [re.sub(r"<[^>]+>", "", cell).strip() for cell in row_cells]
                if row_data:
                    table_data.append(row_data)

            return table_data

        except Exception as e:
            logger.error(f"HTML table parsing failed: {e}")
            return []

    def _extract_with_fallback(self, image: np.ndarray, ocr_results: List[Dict]) -> List[Dict]:
        """Fallback table extraction using line detection and OCR results"""
        try:
            # Detect table structure using line detection
            tables = self._detect_table_structure(image)

            # Match OCR results to table cells
            for table in tables:
                table["data"] = self._match_ocr_to_cells(table, ocr_results)

            logger.info(f"Fallback method extracted {len(tables)} tables")
            return tables

        except Exception as e:
            logger.error(f"Fallback table extraction failed: {e}")
            return []

    def _detect_table_structure(self, image: np.ndarray) -> List[Dict]:
        """Detect table structure using line detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Apply threshold
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            tables = []
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Extract table region
                    table_region = image[y : y + h, x : x + w]

                    # Detect cells in this region
                    cells = self._detect_cells(table_region, x, y)

                    if len(cells) > 4:  # Minimum cells for a valid table
                        tables.append({"type": "detected", "bbox": [x, y, x + w, y + h], "cells": cells, "data": []})

            return tables

        except Exception as e:
            logger.error(f"Table structure detection failed: {e}")
            return []

    def _detect_cells(self, table_region: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """Detect individual cells in table region"""
        try:
            gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY) if len(table_region.shape) == 3 else table_region

            # Find grid intersections
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Find contours for cells
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filter cell-sized areas
                    x, y, w, h = cv2.boundingRect(contour)

                    cells.append(
                        {"bbox": [offset_x + x, offset_y + y, offset_x + x + w, offset_y + y + h], "area": area}
                    )

            # Sort cells by position (top-left to bottom-right)
            cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))

            return cells

        except Exception as e:
            logger.error(f"Cell detection failed: {e}")
            return []

    def _match_ocr_to_cells(self, table: Dict, ocr_results: List[Dict]) -> List[List[str]]:
        """Match OCR results to table cells"""
        try:
            cells = table.get("cells", [])
            if not cells:
                return []

            # Create grid structure
            rows = self._organize_cells_to_grid(cells)

            # Match OCR text to cells
            for row in rows:
                for cell in row:
                    cell["text"] = self._find_text_in_cell(cell["bbox"], ocr_results)

            # Convert to data array
            table_data = []
            for row in rows:
                row_data = [cell.get("text", "") for cell in row]
                table_data.append(row_data)

            return table_data

        except Exception as e:
            logger.error(f"OCR to cell matching failed: {e}")
            return []

    def _organize_cells_to_grid(self, cells: List[Dict]) -> List[List[Dict]]:
        """Organize cells into a grid structure"""
        if not cells:
            return []

        # Group cells by row (similar Y coordinates)
        tolerance = 10
        rows = []
        current_row = [cells[0]]

        for cell in cells[1:]:
            # Check if cell belongs to current row
            if abs(cell["bbox"][1] - current_row[0]["bbox"][1]) <= tolerance:
                current_row.append(cell)
            else:
                # Sort current row by X coordinate
                current_row.sort(key=lambda c: c["bbox"][0])
                rows.append(current_row)
                current_row = [cell]

        # Add last row
        if current_row:
            current_row.sort(key=lambda c: c["bbox"][0])
            rows.append(current_row)

        return rows

    def _find_text_in_cell(self, cell_bbox: List[int], ocr_results: List[Dict]) -> str:
        """Find OCR text that falls within cell boundaries"""
        x1, y1, x2, y2 = cell_bbox
        cell_texts = []

        for ocr_result in ocr_results:
            # Check if OCR text overlaps with cell
            ocr_bbox = ocr_result["bbox"]
            ocr_center_x = self._get_bbox_center_x(ocr_bbox)
            ocr_center_y = self._get_bbox_center_y(ocr_bbox)

            if x1 <= ocr_center_x <= x2 and y1 <= ocr_center_y <= y2:
                cell_texts.append(ocr_result["text"])

        return " ".join(cell_texts).strip()

    def _get_bbox_center_x(self, bbox) -> float:
        """Get X center of bounding box"""
        if isinstance(bbox[0], list):
            # EasyOCR format
            return sum(point[0] for point in bbox) / 4
        else:
            # Standard format [x1, y1, x2, y2]
            return (bbox[0] + bbox[2]) / 2

    def _get_bbox_center_y(self, bbox) -> float:
        """Get Y center of bounding box"""
        if isinstance(bbox[0], list):
            # EasyOCR format
            return sum(point[1] for point in bbox) / 4
        else:
            # Standard format [x1, y1, x2, y2]
            return (bbox[1] + bbox[3]) / 2

    def parse_invoice_line_items(self, tables: List[Dict]) -> List[Dict]:
        """
        Parse table data to extract invoice line items

        Args:
            tables: Extracted table data

        Returns:
            List of parsed line items
        """
        line_items = []

        for table in tables:
            table_data = table.get("data", [])
            if not table_data:
                continue

            # Try to identify the line items table
            if self._is_line_items_table(table_data):
                items = self._parse_line_items_from_table(table_data)
                line_items.extend(items)

        logger.info(f"Parsed {len(line_items)} line items from tables")
        return line_items

    def _is_line_items_table(self, table_data: List[List[str]]) -> bool:
        """Check if table contains line items"""
        if len(table_data) < 2:  # Need at least header + 1 row
            return False

        # Check for common line item headers
        header_row = " ".join(table_data[0]).lower()
        line_item_indicators = [
            "description",
            "quantity",
            "price",
            "amount",
            "total",
            "beschreibung",
            "menge",
            "preis",
            "betrag",
            "summe",
            "kirjeldus",
            "kogus",
            "hind",
            "summa",
        ]

        return any(indicator in header_row for indicator in line_item_indicators)

    def _parse_line_items_from_table(self, table_data: List[List[str]]) -> List[Dict]:
        """Parse line items from table data"""
        line_items = []

        if len(table_data) < 2:
            return line_items

        # Identify column types
        headers = [h.lower().strip() for h in table_data[0]]
        column_mapping = self._identify_columns(headers)

        # Parse data rows
        for row in table_data[1:]:
            if len(row) < len(headers):
                continue

            line_item = self._parse_single_line_item(row, column_mapping)
            if line_item:
                line_items.append(line_item)

        return line_items

    def _identify_columns(self, headers: List[str]) -> Dict[str, int]:
        """Identify column types from headers"""
        column_mapping = {}

        # Column patterns for different languages
        patterns = {
            "description": [
                "description",
                "item",
                "product",
                "beschreibung",
                "artikel",
                "produkt",
                "kirjeldus",
                "toode",
            ],
            "quantity": ["quantity", "qty", "menge", "anzahl", "kogus"],
            "unit_price": ["unit price", "price", "einzelpreis", "preis", "ühiku hind", "hind"],
            "total_price": ["total", "amount", "gesamt", "betrag", "summa", "kokku"],
            "vat_rate": ["vat", "tax", "mwst", "ust", "steuer", "käibemaks"],
        }

        for col_type, pattern_list in patterns.items():
            for i, header in enumerate(headers):
                if any(pattern in header for pattern in pattern_list):
                    column_mapping[col_type] = i
                    break

        return column_mapping

    def _parse_single_line_item(self, row: List[str], column_mapping: Dict[str, int]) -> Optional[Dict]:
        """Parse a single line item row"""
        try:
            line_item = {}

            # Extract description
            if "description" in column_mapping:
                line_item["description"] = row[column_mapping["description"]].strip()

            # Extract quantity
            if "quantity" in column_mapping:
                qty_text = row[column_mapping["quantity"]].strip()
                line_item["quantity"] = self._parse_number(qty_text)

            # Extract unit price
            if "unit_price" in column_mapping:
                price_text = row[column_mapping["unit_price"]].strip()
                line_item["unit_price"] = self._parse_amount(price_text)

            # Extract total price
            if "total_price" in column_mapping:
                total_text = row[column_mapping["total_price"]].strip()
                line_item["total_price"] = self._parse_amount(total_text)

            # Extract VAT rate
            if "vat_rate" in column_mapping:
                vat_text = row[column_mapping["vat_rate"]].strip()
                line_item["vat_rate"] = self._parse_percentage(vat_text)

            # Only return if we have at least description
            if line_item.get("description"):
                return line_item

            return None

        except Exception as e:
            logger.error(f"Failed to parse line item: {e}")
            return None

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text"""
        try:
            # Remove non-numeric characters except decimal separators
            cleaned = re.sub(r"[^\d.,]", "", text)
            if not cleaned:
                return None

            # Handle different decimal separators
            if "," in cleaned and "." in cleaned:
                # Both present, assume European format (1.234,56)
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned:
                # Only comma, could be decimal or thousands separator
                if cleaned.count(",") == 1 and len(cleaned.split(",")[1]) <= 2:
                    # Likely decimal separator
                    cleaned = cleaned.replace(",", ".")
                else:
                    # Likely thousands separator
                    cleaned = cleaned.replace(",", "")

            return float(cleaned)

        except (ValueError, AttributeError):
            return None

    def _parse_amount(self, text: str) -> Optional[float]:
        """Parse monetary amount from text"""
        return self._parse_number(text)

    def _parse_percentage(self, text: str) -> Optional[float]:
        """Parse percentage from text"""
        try:
            # Remove % symbol and parse as number
            cleaned = text.replace("%", "").strip()
            number = self._parse_number(cleaned)
            return number / 100 if number is not None else None

        except Exception:
            return None
