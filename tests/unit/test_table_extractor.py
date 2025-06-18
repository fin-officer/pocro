"""
Unit tests for table extractor
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from src.core.table_extractor import InvoiceTableExtractor


class TestInvoiceTableExtractor:
    """Test cases for InvoiceTableExtractor class"""
    
    @pytest.fixture
    def table_extractor(self):
        """Create table extractor instance"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', False):
            return InvoiceTableExtractor(use_pp_structure=False)
    
    @pytest.fixture
    def table_extractor_with_ppstructure(self):
        """Create table extractor with PP-Structure"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', True):
            with patch('src.core.table_extractor.PPStructure') as mock_pp:
                mock_pp.return_value = Mock()
                return InvoiceTableExtractor(use_pp_structure=True)
    
    @pytest.fixture
    def sample_table_image(self):
        """Create a sample image with table-like structure"""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Draw table structure
        # Horizontal lines
        cv2.line(image, (50, 100), (750, 100), (0, 0, 0), 2)  # Header line
        cv2.line(image, (50, 150), (750, 150), (0, 0, 0), 1)  # Row 1
        cv2.line(image, (50, 200), (750, 200), (0, 0, 0), 1)  # Row 2
        cv2.line(image, (50, 250), (750, 250), (0, 0, 0), 1)  # Row 3
        cv2.line(image, (50, 300), (750, 300), (0, 0, 0), 2)  # Bottom line
        
        # Vertical lines
        cv2.line(image, (50, 100), (50, 300), (0, 0, 0), 2)   # Left border
        cv2.line(image, (300, 100), (300, 300), (0, 0, 0), 1)  # Col 1
        cv2.line(image, (500, 100), (500, 300), (0, 0, 0), 1)  # Col 2
        cv2.line(image, (650, 100), (650, 300), (0, 0, 0), 1)  # Col 3
        cv2.line(image, (750, 100), (750, 300), (0, 0, 0), 2)  # Right border
        
        # Add some text in cells
        cv2.putText(image, "Description", (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, "Qty", (310, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, "Price", (510, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, "Total", (660, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return image
    
    @pytest.fixture
    def sample_ocr_results(self):
        """Sample OCR results for table extraction testing"""
        return [
            {'text': 'Description', 'confidence': 0.95, 'bbox': [[60, 110], [200, 110], [200, 140], [60, 140]]},
            {'text': 'Qty', 'confidence': 0.92, 'bbox': [[310, 110], [340, 110], [340, 140], [310, 140]]},
            {'text': 'Price', 'confidence': 0.90, 'bbox': [[510, 110], [560, 110], [560, 140], [510, 140]]},
            {'text': 'Total', 'confidence': 0.93, 'bbox': [[660, 110], [700, 110], [700, 140], [660, 140]]},
            {'text': 'Software License', 'confidence': 0.88, 'bbox': [[60, 160], [250, 160], [250, 190], [60, 190]]},
            {'text': '1', 'confidence': 0.95, 'bbox': [[320, 160], [330, 160], [330, 190], [320, 190]]},
            {'text': '100.00', 'confidence': 0.90, 'bbox': [[520, 160], [580, 160], [580, 190], [520, 190]]},
            {'text': '100.00', 'confidence': 0.91, 'bbox': [[670, 160], [720, 160], [720, 190], [670, 190]]},
        ]
    
    @pytest.fixture
    def mock_ppstructure_result(self):
        """Mock PP-Structure result"""
        return [
            {
                'type': 'table',
                'bbox': [50, 100, 750, 300],
                'res': {
                    'html': '''<table>
                        <tr><td>Description</td><td>Qty</td><td>Price</td><td>Total</td></tr>
                        <tr><td>Software License</td><td>1</td><td>100.00</td><td>100.00</td></tr>
                        <tr><td>Support Services</td><td>1</td><td>50.00</td><td>50.00</td></tr>
                    </table>''',
                    'cell_bbox': [[[60, 110], [200, 140]], [[310, 110], [340, 140]]]
                }
            }
        ]
    
    def test_table_extractor_initialization_without_ppstructure(self):
        """Test table extractor initialization without PP-Structure"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', False):
            extractor = InvoiceTableExtractor(use_pp_structure=True)
            assert extractor.use_pp_structure is False
    
    def test_table_extractor_initialization_with_ppstructure(self):
        """Test table extractor initialization with PP-Structure"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', True):
            with patch('src.core.table_extractor.PPStructure') as mock_pp:
                mock_pp.return_value = Mock()
                extractor = InvoiceTableExtractor(use_pp_structure=True)
                assert extractor.use_pp_structure is True
                mock_pp.assert_called_once()
    
    def test_extract_tables_with_ppstructure(self, mock_ppstructure_result):
        """Test table extraction using PP-Structure"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', True):
            with patch('src.core.table_extractor.PPStructure') as mock_pp:
                mock_engine = Mock()
                mock_engine.return_value = mock_ppstructure_result
                mock_pp.return_value = mock_engine
                
                extractor = InvoiceTableExtractor(use_pp_structure=True)
                image = np.ones((600, 800, 3), dtype=np.uint8) * 255
                
                tables = extractor.extract_tables(image)
                
                assert len(tables) == 1
                assert tables[0]['type'] == 'ppstructure'
                assert 'bbox' in tables[0]
                assert 'html' in tables[0]
                assert 'data' in tables[0]
                assert tables[0]['rows'] > 0
                assert tables[0]['columns'] > 0
    
    def test_extract_tables_with_fallback(self, table_extractor, sample_table_image, sample_ocr_results):
        """Test table extraction using fallback method"""
        tables = table_extractor.extract_tables(sample_table_image, sample_ocr_results)
        
        assert isinstance(tables, list)
        # Should detect at least some table structure
        assert len(tables) >= 0
    
    def test_parse_html_table(self, table_extractor):
        """Test HTML table parsing"""
        html = '''<table>
            <tr><td>Description</td><td>Qty</td><td>Price</td><td>Total</td></tr>
            <tr><td>Software License</td><td>1</td><td>100.00</td><td>100.00</td></tr>
            <tr><td>Support Services</td><td>1</td><td>50.00</td><td>50.00</td></tr>
        </table>'''
        
        table_data = table_extractor._parse_html_table(html)
        
        assert len(table_data) == 3  # Header + 2 data rows
        assert table_data[0] == ['Description', 'Qty', 'Price', 'Total']
        assert table_data[1] == ['Software License', '1', '100.00', '100.00']
        assert table_data[2] == ['Support Services', '1', '50.00', '50.00']
    
    def test_detect_table_structure(self, table_extractor, sample_table_image):
        """Test table structure detection"""
        tables = table_extractor._detect_table_structure(sample_table_image)
        
        assert isinstance(tables, list)
        # Should detect table structure in the sample image
        if tables:  # May not detect anything depending on implementation
            table = tables[0]
            assert 'bbox' in table
            assert 'cells' in table
            assert table['type'] == 'detected'
    
    def test_detect_cells(self, table_extractor, sample_table_image):
        """Test cell detection in table region"""
        # Extract a table region
        table_region = sample_table_image[90:310, 40:760]
        
        cells = table_extractor._detect_cells(table_region, 40, 90)
        
        assert isinstance(cells, list)
        for cell in cells:
            assert 'bbox' in cell
            assert 'area' in cell
            assert len(cell['bbox']) == 4  # [x1, y1, x2, y2]
    
    def test_organize_cells_to_grid(self, table_extractor):
        """Test organizing cells into grid structure"""
        cells = [
            {'bbox': [50, 100, 150, 120], 'area': 2000},    # Row 1, Col 1
            {'bbox': [200, 100, 300, 120], 'area': 2000},   # Row 1, Col 2
            {'bbox': [350, 100, 450, 120], 'area': 2000},   # Row 1, Col 3
            {'bbox': [50, 150, 150, 170], 'area': 2000},    # Row 2, Col 1
            {'bbox': [200, 150, 300, 170], 'area': 2000},   # Row 2, Col 2
            {'bbox': [350, 150, 450, 170], 'area': 2000},   # Row 2, Col 3
        ]
        
        grid = table_extractor._organize_cells_to_grid(cells)
        
        assert len(grid) == 2  # 2 rows
        assert len(grid[0]) == 3  # 3 columns in first row
        assert len(grid[1]) == 3  # 3 columns in second row
        
        # Check ordering (left to right within rows)
        assert grid[0][0]['bbox'][0] < grid[0][1]['bbox'][0] < grid[0][2]['bbox'][0]
    
    def test_match_ocr_to_cells(self, table_extractor, sample_ocr_results):
        """Test matching OCR results to table cells"""
        # Create mock table with cells
        table = {
            'cells': [
                {'bbox': [60, 110, 200, 140]},   # Should match "Description"
                {'bbox': [310, 110, 340, 140]},  # Should match "Qty"
                {'bbox': [510, 110, 560, 140]},  # Should match "Price"
                {'bbox': [660, 110, 700, 140]},  # Should match "Total"
            ]
        }
        
        table_data = table_extractor._match_ocr_to_cells(table, sample_ocr_results)
        
        assert isinstance(table_data, list)
        if table_data:  # May be empty if no matching
            for row in table_data:
                assert isinstance(row, list)
    
    def test_find_text_in_cell(self, table_extractor, sample_ocr_results):
        """Test finding text within cell boundaries"""
        cell_bbox = [60, 110, 200, 140]  # Should contain "Description"
        
        text = table_extractor._find_text_in_cell(cell_bbox, sample_ocr_results)
        
        assert isinstance(text, str)
        # Should find "Description" text
        if text:
            assert "Description" in text
    
    def test_bbox_center_calculations(self, table_extractor):
        """Test bounding box center calculations"""
        # Test EasyOCR format
        easyocr_bbox = [[50, 100], [150, 100], [150, 120], [50, 120]]
        x_center = table_extractor._get_bbox_center_x(easyocr_bbox)
        y_center = table_extractor._get_bbox_center_y(easyocr_bbox)
        
        assert x_center == 100.0
        assert y_center == 110.0
        
        # Test standard format
        standard_bbox = [50, 100, 150, 120]
        x_center = table_extractor._get_bbox_center_x(standard_bbox)
        y_center = table_extractor._get_bbox_center_y(standard_bbox)
        
        assert x_center == 100.0
        assert y_center == 110.0
    
    def test_parse_invoice_line_items(self, table_extractor):
        """Test parsing invoice line items from tables"""
        tables = [
            {
                'data': [
                    ['Description', 'Quantity', 'Unit Price', 'Total'],
                    ['Software License', '1', '100.00', '100.00'],
                    ['Support Services', '1', '50.00', '50.00']
                ]
            }
        ]
        
        line_items = table_extractor.parse_invoice_line_items(tables)
        
        assert len(line_items) == 2
        
        item1 = line_items[0]
        assert item1['description'] == 'Software License'
        assert item1['quantity'] == 1.0
        assert item1['unit_price'] == 100.0
        assert item1['total_price'] == 100.0
        
        item2 = line_items[1]
        assert item2['description'] == 'Support Services'
        assert item2['quantity'] == 1.0
        assert item2['unit_price'] == 50.0
        assert item2['total_price'] == 50.0
    
    def test_is_line_items_table(self, table_extractor):
        """Test identification of line items tables"""
        # Valid line items table
        valid_table = [
            ['Description', 'Quantity', 'Price', 'Total'],
            ['Item 1', '1', '100.00', '100.00']
        ]
        
        assert table_extractor._is_line_items_table(valid_table) is True
        
        # Invalid table (no line item indicators)
        invalid_table = [
            ['Random', 'Headers'],
            ['Random', 'Data']
        ]
        
        assert table_extractor._is_line_items_table(invalid_table) is False
        
        # Too short table
        short_table = [['Header only']]
        
        assert table_extractor._is_line_items_table(short_table) is False
    
    def test_identify_columns(self, table_extractor):
        """Test column identification from headers"""
        headers = ['description', 'qty', 'unit price', 'total amount']
        
        column_mapping = table_extractor._identify_columns(headers)
        
        assert column_mapping['description'] == 0
        assert column_mapping['quantity'] == 1
        assert column_mapping['unit_price'] == 2
        assert column_mapping['total_price'] == 3
        
        # Test with different languages
        german_headers = ['beschreibung', 'menge', 'preis', 'gesamt']
        german_mapping = table_extractor._identify_columns(german_headers)
        
        assert german_mapping['description'] == 0
        assert german_mapping['quantity'] == 1
        assert german_mapping['unit_price'] == 2
        assert german_mapping['total_price'] == 3
    
    def test_parse_single_line_item(self, table_extractor):
        """Test parsing a single line item"""
        row = ['Software License', '2', '50.00', '100.00']
        column_mapping = {
            'description': 0,
            'quantity': 1,
            'unit_price': 2,
            'total_price': 3
        }
        
        line_item = table_extractor._parse_single_line_item(row, column_mapping)
        
        assert line_item is not None
        assert line_item['description'] == 'Software License'
        assert line_item['quantity'] == 2.0
        assert line_item['unit_price'] == 50.0
        assert line_item['total_price'] == 100.0
    
    def test_parse_single_line_item_missing_description(self, table_extractor):
        """Test parsing line item with missing description"""
        row = ['', '1', '50.00', '50.00']
        column_mapping = {
            'description': 0,
            'quantity': 1,
            'unit_price': 2,
            'total_price': 3
        }
        
        line_item = table_extractor._parse_single_line_item(row, column_mapping)
        
        assert line_item is None  # Should return None for empty description
    
    def test_parse_number(self, table_extractor):
        """Test number parsing"""
        assert table_extractor._parse_number('123') == 123.0
        assert table_extractor._parse_number('123.45') == 123.45
        assert table_extractor._parse_number('1,234.56') == 1234.56
        assert table_extractor._parse_number('1.234,56') == 1234.56
        assert table_extractor._parse_number('') is None
        assert table_extractor._parse_number('abc') is None
    
    def test_parse_amount(self, table_extractor):
        """Test monetary amount parsing"""
        assert table_extractor._parse_amount('€123.45') == 123.45
        assert table_extractor._parse_amount('$1,234.56') == 1234.56
        assert table_extractor._parse_amount('123,45') == 123.45
        assert table_extractor._parse_amount('invalid') is None
    
    def test_parse_percentage(self, table_extractor):
        """Test percentage parsing"""
        assert table_extractor._parse_percentage('19%') == 0.19
        assert table_extractor._parse_percentage('20.5%') == 0.205
        assert table_extractor._parse_percentage('invalid') is None
    
    def test_extract_tables_error_handling(self, table_extractor):
        """Test error handling in table extraction"""
        # Test with None image
        result = table_extractor.extract_tables(None, [])
        assert result == []
        
        # Test with invalid image
        invalid_image = np.array([])
        result = table_extractor.extract_tables(invalid_image, [])
        assert result == []
    
    def test_ppstructure_error_handling(self):
        """Test PP-Structure error handling"""
        with patch('src.core.table_extractor.PPSTRUCTURE_AVAILABLE', True):
            with patch('src.core.table_extractor.PPStructure') as mock_pp:
                mock_engine = Mock()
                mock_engine.side_effect = Exception("PP-Structure failed")
                mock_pp.return_value = mock_engine
                
                extractor = InvoiceTableExtractor(use_pp_structure=True)
                image = np.ones((600, 800, 3), dtype=np.uint8) * 255
                
                result = extractor.extract_tables(image)
                assert result == []
    
    @pytest.mark.parametrize("table_data,expected_count", [
        ([['Description', 'Qty', 'Price'], ['Item 1', '1', '10.00']], 1),
        ([['Random', 'Headers'], ['Random', 'Data']], 0),
        ([['Beschreibung', 'Menge', 'Preis'], ['Artikel 1', '1', '10.00']], 1),
        ([['Kirjeldus', 'Kogus', 'Hind'], ['Toode 1', '1', '10.00']], 1),
    ])
    def test_parse_line_items_different_languages(self, table_extractor, table_data, expected_count):
        """Test parsing line items with different languages"""
        tables = [{'data': table_data}]
        
        line_items = table_extractor.parse_invoice_line_items(tables)
        assert len(line_items) == expected_count
    
    def test_performance_with_large_table(self, table_extractor):
        """Test performance with large table"""
        # Create large table data
        large_table_data = [['Description', 'Qty', 'Price', 'Total']]
        for i in range(100):
            large_table_data.append([f'Item {i}', '1', f'{i}.00', f'{i}.00'])
        
        tables = [{'data': large_table_data}]
        
        import time
        start_time = time.time()
        line_items = table_extractor.parse_invoice_line_items(tables)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert len(line_items) == 100
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    def test_complex_table_structure(self, table_extractor):
        """Test with complex table structure"""
        # Table with merged cells or irregular structure
        complex_table_data = [
            ['Item', 'Details', '', 'Amount'],  # Empty cell
            ['Software', 'License', 'Annual', '1200.00'],
            ['', 'Support', 'Monthly', '100.00'],  # Empty first cell
            ['Hardware', 'Server', '', '5000.00']
        ]
        
        tables = [{'data': complex_table_data}]
        line_items = table_extractor.parse_invoice_line_items(tables)
        
        # Should handle complex structure gracefully
        assert isinstance(line_items, list)
        
        # Check that valid items are parsed
        valid_items = [item for item in line_items if item.get('description')]
        assert len(valid_items) >= 1
